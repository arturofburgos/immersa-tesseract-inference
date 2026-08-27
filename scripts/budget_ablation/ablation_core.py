"""General N-sensor machinery for the budget-versus-placement ablation.

The frozen application helpers assume exactly two probes: ``unpack_layout``
rejects anything but a length-4 vector and ``separation_penalty`` differences a
single pair. This module generalises both to N sensors without touching the
frozen code or any Tesseract mathematics.

A layout is a flat vector [x1, y1, x2, y2, ...] of length 2*Ns.
"""

from contextlib import ExitStack
from typing import Any

import numpy as np
from immersa_tesseract_inference.cfd_bank import (
    FLOW_KEYS,
    OBSERVATION_TIMES,
    load_flow,
)
from immersa_tesseract_inference.sensor_design import (
    effective_pair_count,
    retained_pair_mask,
    softmin_weights,
    solve_tau_for_effective_pairs,
)
from tesseract_core import Tesseract

SURROGATE_IMAGE = "immersa_tesseract_inference_wake_surrogate"
OBSERVATION_IMAGE = "immersa_tesseract_inference_wake_observation"
DESIGN_IMAGE = "immersa_tesseract_inference_sensor_array_design"

DESIGN_GRID_DEG = np.arange(20.0, 85.0 + 1.0e-9, 1.0, dtype=np.float64)
DELTA_ALPHA_MIN_DEG = 7.5

X_BOUNDS = (1.0, 3.0)
Y_BOUNDS = (-1.0, 1.0)
MIN_SEPARATION = 0.2

_DISTANCE_EPSILON = 1.0e-12


def unpack(layout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a flat layout into sensor_x and sensor_y."""
    layout = np.asarray(layout, dtype=np.float64)

    if layout.ndim != 1 or layout.size % 2 != 0:
        raise ValueError(f"layout must be flat with even length; got {layout.shape}.")

    return layout[0::2].copy(), layout[1::2].copy()


def pack(sensor_x: np.ndarray, sensor_y: np.ndarray) -> np.ndarray:
    """Interleave sensor coordinates back into a flat layout."""
    layout = np.empty(2 * len(sensor_x), dtype=np.float64)
    layout[0::2] = sensor_x
    layout[1::2] = sensor_y
    return layout


def naive_layout(n_sensors: int, y_positions: list[float]) -> np.ndarray:
    """Vertically aligned probes at x = 1."""
    if len(y_positions) != n_sensors:
        raise ValueError("y_positions length must match n_sensors.")
    return pack(np.ones(n_sensors), np.asarray(y_positions, dtype=np.float64))


def canonicalize(layout: np.ndarray) -> np.ndarray:
    """Order probes by (x, y) so layouts can be compared.

    Probes are unlabelled, so this is a relabelling only. Never apply it inside
    an objective, where it would introduce a discontinuity as probes reorder.
    """
    sensor_x, sensor_y = unpack(layout)
    order = np.lexsort((sensor_y, sensor_x))
    return pack(sensor_x[order], sensor_y[order])


def bounds_for(n_sensors: int) -> list[tuple[float, float]]:
    """Box bounds for L-BFGS-B, alternating x and y."""
    return [X_BOUNDS if i % 2 == 0 else Y_BOUNDS for i in range(2 * n_sensors)]


def separation_penalty(
    layout: np.ndarray,
    *,
    lambda_separation: float,
    min_distance: float = MIN_SEPARATION,
) -> tuple[float, np.ndarray]:
    """One-sided quadratic penalty over EVERY probe pair.

        P = lambda * sum_{i<j} relu(min_distance - r_ij)^2

    Returns the penalty and its gradient. Both vanish once every pair is far
    enough apart, so the term is inactive at any well-separated layout.
    """
    layout = np.asarray(layout, dtype=np.float64)
    sensor_x, sensor_y = unpack(layout)

    n = len(sensor_x)
    penalty = 0.0
    gradient = np.zeros_like(layout)

    for i in range(n):
        for j in range(i + 1, n):
            dx = sensor_x[i] - sensor_x[j]
            dy = sensor_y[i] - sensor_y[j]
            r = float(np.sqrt(dx * dx + dy * dy + _DISTANCE_EPSILON))

            violation = min_distance - r
            if violation <= 0.0:
                continue

            penalty += lambda_separation * violation**2

            scale = -2.0 * lambda_separation * violation / r

            gradient[2 * i] += scale * dx
            gradient[2 * i + 1] += scale * dy
            gradient[2 * j] -= scale * dx
            gradient[2 * j + 1] -= scale * dy

    return penalty, gradient


def min_pairwise_separation(layout: np.ndarray) -> float:
    """Smallest distance between any two probes."""
    sensor_x, sensor_y = unpack(layout)
    if len(sensor_x) < 2:
        return float("inf")
    return min(
        float(np.hypot(sensor_x[i] - sensor_x[j], sensor_y[i] - sensor_y[j]))
        for i in range(len(sensor_x))
        for j in range(i + 1, len(sensor_x))
    )


class AblationPipeline:
    """Surrogate and physical discrimination objectives for any sensor count.

    Both routes feed the same frozen SensorArrayDesign component; only the
    source of the measurements differs.

        surrogate   s -> T3 -> T4
        physical    s -> T2[T1 bank] -> T4

    and in reverse, T4 VJP followed by the matching sensor VJP.
    """

    def __init__(self, alphas: np.ndarray = DESIGN_GRID_DEG) -> None:
        """Configure the AoA grid and preload the physical bank lazily."""
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.mask = retained_pair_mask(self.alphas, DELTA_ALPHA_MIN_DEG)

        self._stack: ExitStack | None = None
        self._surrogate: Any = None
        self._observation: Any = None
        self._design: Any = None
        self._flows: list[dict] | None = None

        self.n_surrogate_calls = 0
        self.n_observation_calls = 0
        self.n_design_calls = 0

    def __enter__(self) -> "AblationPipeline":
        """Start the three Tesseracts."""
        self._stack = ExitStack()
        self._surrogate = self._stack.enter_context(
            Tesseract.from_image(SURROGATE_IMAGE)
        )
        self._observation = self._stack.enter_context(
            Tesseract.from_image(OBSERVATION_IMAGE)
        )
        self._design = self._stack.enter_context(Tesseract.from_image(DESIGN_IMAGE))
        return self

    def __exit__(self, *_: object) -> None:
        """Stop the Tesseracts."""
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._surrogate = self._observation = self._design = None

    @property
    def flows(self) -> list[dict]:
        """Bank flows, loaded once on first physical use."""
        if self._flows is None:
            self._flows = [load_flow(a) for a in self.alphas]
        return self._flows

    # ------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------

    def surrogate_measurements(self, layout: np.ndarray) -> np.ndarray:
        """T3 predictions over the AoA grid, (n_alpha, Ns, 5, 2)."""
        sensor_x, sensor_y = unpack(layout)

        batch = []
        for alpha in self.alphas:
            outputs = self._surrogate.apply(
                {
                    "angle_of_attack_deg": float(alpha),
                    "sensor_x": sensor_x.astype(np.float32),
                    "sensor_y": sensor_y.astype(np.float32),
                }
            )
            self.n_surrogate_calls += 1
            batch.append(np.asarray(outputs["measurements"], dtype=np.float64))

        return np.stack(batch)

    def _observation_inputs(
        self,
        flow: dict,
        sensor_x: np.ndarray,
        sensor_y: np.ndarray,
    ) -> dict:
        return {
            **{key: flow[key] for key in FLOW_KEYS},
            "times": flow["times"],
            "sensor_x": sensor_x,
            "sensor_y": sensor_y,
            "sensor_times": OBSERVATION_TIMES,
        }

    def physical_measurements(self, layout: np.ndarray) -> np.ndarray:
        """Real-CFD measurements over the AoA grid, (n_alpha, Ns, 5, 2)."""
        sensor_x, sensor_y = unpack(layout)

        batch = []
        for flow in self.flows:
            outputs = self._observation.apply(
                self._observation_inputs(flow, sensor_x, sensor_y)
            )
            self.n_observation_calls += 1
            batch.append(np.asarray(outputs["measurements"], dtype=np.float64))

        return np.stack(batch)

    # ------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------

    def score(self, measurements: np.ndarray, tau: float) -> dict:
        """Evaluate the frozen T4 criterion on a measurement batch."""
        outputs = self._design.apply(
            {
                "measurements": measurements.astype(np.float32),
                "alpha_deg": self.alphas.astype(np.float32),
                "delta_alpha_min_deg": float(DELTA_ALPHA_MIN_DEG),
                "tau": float(tau),
            }
        )
        self.n_design_calls += 1

        distances = np.asarray(outputs["pair_distances"], dtype=np.float64)
        retained = distances[self.mask]
        weights = np.sort(softmin_weights(retained, tau))[::-1]

        masked = np.where(self.mask, distances, np.inf)
        i, j = np.unravel_index(np.argmin(masked), masked.shape)

        return {
            "D_tau": float(np.asarray(outputs["discrimination"])),
            "hard_min": float(np.asarray(outputs["min_pair_distance"])),
            "n_pairs": int(np.asarray(outputs["n_pairs"])),
            "n_eff": effective_pair_count(retained, tau),
            "top1_weight": float(weights[0]),
            "hardest_pair_deg": [float(self.alphas[i]), float(self.alphas[j])],
            "pair_distances": distances,
        }

    def calibrate_tau(
        self,
        measurements: np.ndarray,
        target: float = 10.0,
    ) -> float:
        """Solve for tau giving the requested effective-pair count."""
        distances = np.sort(
            self.score(measurements, tau=1.0)["pair_distances"][self.mask]
        )
        return solve_tau_for_effective_pairs(distances, target=target)

    # ------------------------------------------------------------
    # Surrogate objective and gradient
    # ------------------------------------------------------------

    def surrogate_objective(
        self,
        layout: np.ndarray,
        *,
        tau: float,
        lambda_separation: float,
    ) -> tuple[float, np.ndarray]:
        """Minimized surrogate objective and its gradient over the layout."""
        layout = np.asarray(layout, dtype=np.float64)
        sensor_x, sensor_y = unpack(layout)

        batch = self.surrogate_measurements(layout)
        scored = self.score(batch, tau)

        cotangent = np.asarray(
            self._design.vector_jacobian_product(
                {
                    "measurements": batch.astype(np.float32),
                    "alpha_deg": self.alphas.astype(np.float32),
                    "delta_alpha_min_deg": float(DELTA_ALPHA_MIN_DEG),
                    "tau": float(tau),
                },
                vjp_inputs=["measurements"],
                vjp_outputs=["discrimination"],
                cotangent_vector={"discrimination": 1.0},
            )["measurements"],
            dtype=np.float64,
        )
        self.n_design_calls += 1

        grad_x = np.zeros(len(sensor_x))
        grad_y = np.zeros(len(sensor_y))

        for index, alpha in enumerate(self.alphas):
            outputs = self._surrogate.vector_jacobian_product(
                {
                    "angle_of_attack_deg": float(alpha),
                    "sensor_x": sensor_x.astype(np.float32),
                    "sensor_y": sensor_y.astype(np.float32),
                },
                vjp_inputs=["sensor_x", "sensor_y"],
                vjp_outputs=["measurements"],
                cotangent_vector={"measurements": cotangent[index].astype(np.float32)},
            )
            self.n_surrogate_calls += 1
            grad_x += np.asarray(outputs["sensor_x"], dtype=np.float64)
            grad_y += np.asarray(outputs["sensor_y"], dtype=np.float64)

        penalty, grad_penalty = separation_penalty(
            layout, lambda_separation=lambda_separation
        )

        return (
            float(-scored["D_tau"] + penalty),
            -pack(grad_x, grad_y) + grad_penalty,
        )
