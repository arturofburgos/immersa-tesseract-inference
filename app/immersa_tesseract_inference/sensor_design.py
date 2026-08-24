"""Differentiable sensor-array design over the WakeSurrogate observable map.

Composition, all through the Tesseract API -- no surrogate weights are ever
imported into this process:

    sensor coordinates s = [x1, y1, x2, y2]
      -> WakeSurrogate.apply, once per design AoA
      -> M, shape (N_alpha, N_sensors, 5, 2)
      -> SensorArrayDesign.apply
      -> D_tau(s)

and in reverse, for the gradient:

    seed dD/dD = 1
      -> SensorArrayDesign.vector_jacobian_product
      -> G = dD/dM, shape (N_alpha, N_sensors, 5, 2)
      -> WakeSurrogate.vector_jacobian_product, once per AoA, seeded with G[a]
      -> dD/d(sensor_x), dD/d(sensor_y)
      -> accumulate over AoA
      -> dD/ds, shape (4,)

Reverse mode is used on both hops. The alternative -- materializing T3's full
sensor Jacobian and contracting it by hand -- would depend on that Jacobian
being block-diagonal in the sensor index, and grows as N_sensors^2 rather than
N_sensors.
"""

from typing import Any

import numpy as np
from tesseract_core import Tesseract

WAKE_SURROGATE_IMAGE = "immersa_tesseract_inference_wake_surrogate"
SENSOR_ARRAY_DESIGN_IMAGE = "immersa_tesseract_inference_sensor_array_design"

# Design AoA grid: 20 to 85 degrees inclusive in 2.5 degree steps.
#
# This deliberately does not contain 63 degrees. The sealed physical truth must
# never enter the design prior, otherwise the optimized layout would be tuned
# to the very angle it is later validated against.
ALPHA_GRID_DEG = np.arange(20.0, 85.0 + 1.0e-9, 2.5, dtype=np.float64)

DELTA_ALPHA_MIN_DEG = 7.5

# Baseline two-sensor layout, as [x1, y1, x2, y2].
BASELINE_LAYOUT = np.array([1.0, -0.4, 1.0, 0.4], dtype=np.float64)

# Box constraints, matching the WakeSurrogate training region.
DESIGN_BOUNDS = ((1.0, 3.0), (-1.0, 1.0), (1.0, 3.0), (-1.0, 1.0))

# Minimum sensor separation: four CFD cells at h = 0.05, which keeps the two
# probes from becoming effectively colocated.
MIN_SENSOR_DISTANCE = 0.2

# Guards the derivative of r at coincident sensors.
_DISTANCE_EPSILON = 1.0e-12


def unpack_layout(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split [x1, y1, x2, y2] into sensor_x and sensor_y arrays."""
    design = np.asarray(design, dtype=np.float64)

    if design.shape != (4,):
        raise ValueError(f"design must have shape (4,); got {design.shape}.")

    return design[[0, 2]].copy(), design[[1, 3]].copy()


def canonicalize_layout(design: np.ndarray) -> np.ndarray:
    """Order the two sensors by (x, y) so layouts can be compared.

    The design score is invariant to sensor relabelling, so the optimizer sees
    two equivalent optima. This is only for storing and comparing results; it
    must never be applied inside the objective, where it would introduce a
    discontinuity as the two sensors exchange order.
    """
    sensor_x, sensor_y = unpack_layout(design)

    order = np.lexsort((sensor_y, sensor_x))

    return np.array(
        [
            sensor_x[order[0]],
            sensor_y[order[0]],
            sensor_x[order[1]],
            sensor_y[order[1]],
        ],
        dtype=np.float64,
    )


def retained_pair_mask(
    alpha_deg: np.ndarray,
    delta_alpha_min_deg: float,
) -> np.ndarray:
    """Strict upper-triangular mask of sufficiently separated AoA pairs.

    Mirrors the mask inside SensorArrayDesign so the application can select
    retained entries out of ``pair_distances``. Callers should cross-check the
    count against the Tesseract's ``n_pairs`` to catch any drift between the
    two implementations.
    """
    alpha_deg = np.asarray(alpha_deg, dtype=np.float64)

    separation = np.abs(alpha_deg[:, None] - alpha_deg[None, :])

    upper = np.triu(
        np.ones((alpha_deg.size, alpha_deg.size), dtype=bool),
        k=1,
    )

    return upper & (separation >= delta_alpha_min_deg)


def softmin_weights(distances: np.ndarray, tau: float) -> np.ndarray:
    """Normalized soft-minimum weights over a set of pair distances.

        w_p = exp(-d_p / tau) / sum_q exp(-d_q / tau)

    Computed relative to the smallest distance so the exponentials stay in
    range for small tau.
    """
    distances = np.asarray(distances, dtype=np.float64)

    shifted = -(distances - distances.min()) / tau

    weights = np.exp(shifted)

    return weights / weights.sum()


def effective_pair_count(distances: np.ndarray, tau: float) -> float:
    """Perplexity of the soft-minimum weights: how many pairs actually count.

        N_eff(tau) = exp( -sum_p w_p log w_p )

    Ranges from 1 (all weight on the closest pair, tau -> 0) to the number of
    retained pairs (uniform weights, tau -> infinity), and increases
    monotonically with tau -- which is what makes it safe to invert by
    bisection.

    Evaluated through the log-sum-exp identity

        entropy = logsumexp(z) - sum_p w_p z_p,   z_p = -d_p / tau,

    which avoids taking the logarithm of an underflowed weight.
    """
    distances = np.asarray(distances, dtype=np.float64)

    shifted = -(distances - distances.min()) / tau

    max_shifted = shifted.max()

    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))

    weights = np.exp(shifted - log_sum_exp)

    entropy = log_sum_exp - float(np.sum(weights * shifted))

    return float(np.exp(entropy))


def solve_tau_for_effective_pairs(
    distances: np.ndarray,
    target: float = 10.0,
) -> float:
    """Find tau whose soft-minimum weights spread over ``target`` pairs.

    ``effective_pair_count`` is monotone in tau, so this brackets the target by
    geometric expansion and then bisects. Bisection on log(tau) is used rather
    than a derivative-based solve because tau spans orders of magnitude and the
    objective is only needed to a few significant figures.
    """
    distances = np.asarray(distances, dtype=np.float64)

    if not 1.0 < target < distances.size:
        raise ValueError(
            f"target must lie strictly between 1 and {distances.size}; got {target}."
        )

    spread = float(distances.max() - distances.min())

    if spread <= 0.0:
        raise ValueError("Pair distances are degenerate; cannot calibrate tau.")

    # Expand a bracket outwards until it straddles the target.
    low, high = 1.0e-6 * spread, spread

    for _ in range(60):
        if effective_pair_count(distances, low) <= target:
            break
        low /= 2.0
    else:
        raise RuntimeError("Failed to bracket tau from below.")

    for _ in range(60):
        if effective_pair_count(distances, high) >= target:
            break
        high *= 2.0
    else:
        raise RuntimeError("Failed to bracket tau from above.")

    for _ in range(200):
        middle = float(np.sqrt(low * high))

        if effective_pair_count(distances, middle) < target:
            low = middle
        else:
            high = middle

        if high / low < 1.0 + 1.0e-10:
            break

    return float(np.sqrt(low * high))


def separation_penalty(
    design: np.ndarray,
    *,
    lambda_separation: float,
    min_distance: float = MIN_SENSOR_DISTANCE,
) -> tuple[float, np.ndarray]:
    """One-sided quadratic penalty keeping the two sensors apart.

        P = lambda * relu(min_distance - r)^2,
        r = ||s1 - s2||.

    Returns the penalty and its gradient with respect to [x1, y1, x2, y2]. The
    penalty and its first derivative are both zero once the constraint holds,
    so it is inactive at any layout that respects the separation.
    """
    design = np.asarray(design, dtype=np.float64)

    dx = design[0] - design[2]
    dy = design[1] - design[3]

    r = float(np.sqrt(dx * dx + dy * dy + _DISTANCE_EPSILON))

    violation = min_distance - r

    if violation <= 0.0:
        return 0.0, np.zeros(4, dtype=np.float64)

    penalty = lambda_separation * violation**2

    # dP/dr = -2 * lambda * violation, and dr/dx1 = dx / r.
    scale = -2.0 * lambda_separation * violation / r

    gradient = np.array(
        [scale * dx, scale * dy, -scale * dx, -scale * dy],
        dtype=np.float64,
    )

    return penalty, gradient


class SensorDesignPipeline:
    """Compose WakeSurrogate and SensorArrayDesign into an objective.

    Both Tesseracts stay alive for the lifetime of the context manager, so the
    per-call cost is one HTTP round trip rather than a container start.
    """

    def __init__(
        self,
        surrogate_image: str = WAKE_SURROGATE_IMAGE,
        design_image: str = SENSOR_ARRAY_DESIGN_IMAGE,
        *,
        alpha_grid_deg: np.ndarray | None = None,
        delta_alpha_min_deg: float = DELTA_ALPHA_MIN_DEG,
    ) -> None:
        """Configure the design grid and the two component images."""
        self.surrogate_image = surrogate_image
        self.design_image = design_image

        self.alpha_grid_deg = np.asarray(
            ALPHA_GRID_DEG if alpha_grid_deg is None else alpha_grid_deg,
            dtype=np.float64,
        )

        self.delta_alpha_min_deg = float(delta_alpha_min_deg)

        self._stack: Any = None
        self._surrogate: Any = None
        self._design: Any = None

        self.n_surrogate_calls = 0
        self.n_design_calls = 0

    def __enter__(self) -> "SensorDesignPipeline":
        """Start both Tesseracts."""
        from contextlib import ExitStack

        self._stack = ExitStack()

        self._surrogate = self._stack.enter_context(
            Tesseract.from_image(self.surrogate_image)
        )

        self._design = self._stack.enter_context(
            Tesseract.from_image(self.design_image)
        )

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Stop both Tesseracts."""
        if self._stack is not None:
            self._stack.close()

        self._stack = None
        self._surrogate = None
        self._design = None

    def _require_active(self) -> None:
        if self._surrogate is None or self._design is None:
            raise RuntimeError(
                "Pipeline is not active. Use "
                "'with SensorDesignPipeline() as pipeline:'."
            )

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------

    def measurements(self, design: np.ndarray) -> np.ndarray:
        """Evaluate WakeSurrogate across the design AoA grid.

        WakeSurrogate takes a scalar angle of attack, so this costs one call
        per grid angle. Returns shape (N_alpha, N_sensors, 5, 2).
        """
        self._require_active()

        sensor_x, sensor_y = unpack_layout(design)

        sensor_x = sensor_x.astype(np.float32)
        sensor_y = sensor_y.astype(np.float32)

        batch = []

        for alpha in self.alpha_grid_deg:
            outputs = self._surrogate.apply(
                {
                    "angle_of_attack_deg": float(alpha),
                    "sensor_x": sensor_x,
                    "sensor_y": sensor_y,
                }
            )

            self.n_surrogate_calls += 1

            batch.append(np.asarray(outputs["measurements"], dtype=np.float32))

        return np.stack(batch, axis=0)

    def discrimination(
        self,
        measurements: np.ndarray,
        tau: float,
    ) -> dict[str, Any]:
        """Score a measurement batch with SensorArrayDesign."""
        self._require_active()

        outputs = self._design.apply(
            {
                "measurements": np.asarray(measurements, dtype=np.float32),
                "alpha_deg": self.alpha_grid_deg.astype(np.float32),
                "delta_alpha_min_deg": float(self.delta_alpha_min_deg),
                "tau": float(tau),
            }
        )

        self.n_design_calls += 1

        return {
            "discrimination": float(np.asarray(outputs["discrimination"])),
            "pair_distances": np.asarray(outputs["pair_distances"], dtype=np.float64),
            "min_pair_distance": float(np.asarray(outputs["min_pair_distance"])),
            "n_pairs": int(np.asarray(outputs["n_pairs"])),
        }

    # ----------------------------------------------------------------
    # Reverse
    # ----------------------------------------------------------------

    def discrimination_cotangent(
        self,
        measurements: np.ndarray,
        tau: float,
    ) -> np.ndarray:
        """Pull dD/dD = 1 back onto the measurements via T4's VJP.

        Returns dD/d(measurements) with the shape of ``measurements``.
        """
        self._require_active()

        outputs = self._design.vector_jacobian_product(
            {
                "measurements": np.asarray(measurements, dtype=np.float32),
                "alpha_deg": self.alpha_grid_deg.astype(np.float32),
                "delta_alpha_min_deg": float(self.delta_alpha_min_deg),
                "tau": float(tau),
            },
            vjp_inputs=["measurements"],
            vjp_outputs=["discrimination"],
            cotangent_vector={"discrimination": 1.0},
        )

        self.n_design_calls += 1

        return np.asarray(outputs["measurements"], dtype=np.float64)

    def design_gradient(
        self,
        design: np.ndarray,
        cotangent: np.ndarray,
    ) -> np.ndarray:
        """Pull a measurement cotangent back onto the sensor coordinates.

        One WakeSurrogate VJP per design angle, seeded with that angle's slice
        of the cotangent, accumulated over the grid.
        """
        self._require_active()

        sensor_x, sensor_y = unpack_layout(design)

        sensor_x = sensor_x.astype(np.float32)
        sensor_y = sensor_y.astype(np.float32)

        grad_x = np.zeros(sensor_x.size, dtype=np.float64)
        grad_y = np.zeros(sensor_y.size, dtype=np.float64)

        for index, alpha in enumerate(self.alpha_grid_deg):
            outputs = self._surrogate.vector_jacobian_product(
                {
                    "angle_of_attack_deg": float(alpha),
                    "sensor_x": sensor_x,
                    "sensor_y": sensor_y,
                },
                vjp_inputs=["sensor_x", "sensor_y"],
                vjp_outputs=["measurements"],
                cotangent_vector={"measurements": cotangent[index].astype(np.float32)},
            )

            self.n_surrogate_calls += 1

            grad_x += np.asarray(outputs["sensor_x"], dtype=np.float64)
            grad_y += np.asarray(outputs["sensor_y"], dtype=np.float64)

        return np.array(
            [grad_x[0], grad_y[0], grad_x[1], grad_y[1]],
            dtype=np.float64,
        )

    # ----------------------------------------------------------------
    # Calibration
    # ----------------------------------------------------------------

    def calibrate(
        self,
        design: np.ndarray = BASELINE_LAYOUT,
        *,
        target_effective_pairs: float = 10.0,
        min_distance: float = MIN_SENSOR_DISTANCE,
    ) -> dict[str, float]:
        """Fix tau and the separation weight from the baseline layout.

        tau is chosen so the soft-minimum weights spread over exactly
        ``target_effective_pairs`` pairs, measured by the weight perplexity.
        This states the intent directly, unlike a rank-k distance gap, whose
        effective spread also depends on how many pairs sit in the tail and so
        drifts when the AoA grid or the separation threshold changes.

        Both constants are computed once and then frozen. An adaptive tau would
        make the objective non-stationary and invalidate the quasi-Newton
        curvature estimates.

        The superseded rank-based value is returned alongside for comparison.
        """
        measurements = self.measurements(design)

        # tau only sets the scale of the reported score here, so any positive
        # placeholder works; the pair distances it returns are tau-independent.
        scored = self.discrimination(measurements, tau=1.0)

        mask = retained_pair_mask(self.alpha_grid_deg, self.delta_alpha_min_deg)

        if int(mask.sum()) != scored["n_pairs"]:
            raise RuntimeError(
                "Retained pair count disagrees with SensorArrayDesign: "
                f"application mask has {int(mask.sum())}, "
                f"component reported {scored['n_pairs']}."
            )

        distances = np.sort(scored["pair_distances"][mask])

        median_distance = float(np.median(distances))

        tau = solve_tau_for_effective_pairs(
            distances,
            target=target_effective_pairs,
        )

        # Superseded rank-10 gap heuristic, kept only for the calibration report.
        rank = 10
        legacy_tau = max(
            float(distances[rank - 1] - distances[0]),
            1.0e-3 * median_distance,
        )

        weights = np.sort(softmin_weights(distances, tau))[::-1]

        lambda_separation = 10.0 * median_distance / min_distance**2

        return {
            "tau": tau,
            "legacy_tau": legacy_tau,
            "effective_pairs": effective_pair_count(distances, tau),
            "legacy_effective_pairs": effective_pair_count(distances, legacy_tau),
            "top1_weight": float(weights[0]),
            "top10_weight": float(weights[:10].sum()),
            "lambda_separation": lambda_separation,
            "min_distance": float(distances[0]),
            "rank_distance": float(distances[rank - 1]),
            "median_distance": median_distance,
            "n_pairs": scored["n_pairs"],
        }

    # ----------------------------------------------------------------
    # Objective
    # ----------------------------------------------------------------

    def objective_gradient_and_diagnostics(
        self,
        design: np.ndarray,
        *,
        tau: float,
        lambda_separation: float,
        min_distance: float = MIN_SENSOR_DISTANCE,
    ) -> tuple[float, np.ndarray, dict[str, float]]:
        """Objective, gradient, and the diagnostics that fall out for free.

        The diagnostics are byproducts of quantities the objective already
        computes, so recording an optimizer trajectory costs no additional
        Tesseract calls.
        """
        design = np.asarray(design, dtype=np.float64)

        measurements = self.measurements(design)

        scored = self.discrimination(measurements, tau)

        cotangent = self.discrimination_cotangent(measurements, tau)

        grad_discrimination = self.design_gradient(design, cotangent)

        penalty, grad_penalty = separation_penalty(
            design,
            lambda_separation=lambda_separation,
            min_distance=min_distance,
        )

        objective = -scored["discrimination"] + penalty

        gradient = -grad_discrimination + grad_penalty

        mask = retained_pair_mask(self.alpha_grid_deg, self.delta_alpha_min_deg)

        distances = scored["pair_distances"][mask]

        weights = np.sort(softmin_weights(distances, tau))[::-1]

        separation = float(np.hypot(design[0] - design[2], design[1] - design[3]))

        diagnostics = {
            "discrimination": scored["discrimination"],
            "hard_min_distance": scored["min_pair_distance"],
            "effective_pairs": effective_pair_count(distances, tau),
            "top1_weight": float(weights[0]),
            "top10_weight": float(weights[:10].sum()),
            "separation": separation,
            "penalty": penalty,
            "objective": float(objective),
        }

        return float(objective), gradient, diagnostics

    def objective_and_gradient(
        self,
        design: np.ndarray,
        *,
        tau: float,
        lambda_separation: float,
        min_distance: float = MIN_SENSOR_DISTANCE,
    ) -> tuple[float, np.ndarray]:
        """Minimized objective and its gradient in [x1, y1, x2, y2].

        L(s) = -D_tau(s) + lambda * relu(min_distance - r)^2
        """
        objective, gradient, _ = self.objective_gradient_and_diagnostics(
            design,
            tau=tau,
            lambda_separation=lambda_separation,
            min_distance=min_distance,
        )

        return objective, gradient
