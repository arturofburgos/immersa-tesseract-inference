"""Check final WakeSurrogate gradients using central finite differences."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from immersa_tesseract_inference.wake_surrogate_config import (
    SENSOR_TIMES as CONFIG_SENSOR_TIMES,
)

# ============================================================
# Configuration
# ============================================================

MODEL_ROOT = Path("models/wake_surrogate/v3_broad_aoa_refinement")

RESULTS_ROOT = Path("results/wake_surrogate/final_validation/gradient_check")

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Final model
# ============================================================

MODEL_VERSION = "v3"

MODEL_LABEL = "V3 Broad AoA Refinement"


# ============================================================
# Architecture
# ============================================================

WIDTH = 128

DEPTH = 3

FOURIER_FREQUENCIES = jnp.array(
    [
        1.0,
        2.0,
        4.0,
        8.0,
    ],
    dtype=jnp.float32,
)

SENSOR_TIMES = np.asarray(
    CONFIG_SENSOR_TIMES,
    dtype=np.float64,
)


# ============================================================
# Gradient-check configuration
# ============================================================

# Representative interior points.
#
# We avoid x = 1 exactly because a central difference
# would step slightly outside the sensor-design domain.

CHECK_POINTS = [
    (
        40.0,
        1.25,
        -0.40,
    ),
    (
        63.0,
        1.50,
        0.00,
    ),
    (
        80.0,
        2.50,
        0.50,
    ),
]


VARIABLE_NAMES = [
    "alpha_deg",
    "x",
    "y",
]


# Test several finite-difference step sizes.
#
# alpha is measured in degrees.
#
# x and y are nondimensional spatial coordinates.

FD_EPSILONS = {
    "alpha_deg": [
        1.0e-1,
        1.0e-2,
        1.0e-3,
    ],
    "x": [
        1.0e-2,
        1.0e-3,
        1.0e-4,
    ],
    "y": [
        1.0e-2,
        1.0e-3,
        1.0e-4,
    ],
}


# ============================================================
# WakeSurrogate architecture
#
# Must match the architecture used during V3 training.
# ============================================================


class WakeSurrogate(eqx.Module):
    """Final differentiable WakeSurrogate.

    Physical input:
        [alpha_deg, x, y]

    Fourier encoding:
        x and y only

    Output:
        [ux(t1), uy(t1), ..., ux(t5), uy(t5)]
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        key: jax.Array,
    ) -> None:
        n_original = 3

        n_spatial = 2

        n_fourier = 2 * n_spatial * len(FOURIER_FREQUENCIES)

        feature_size = n_original + n_fourier

        self.mlp = eqx.nn.MLP(
            in_size=feature_size,
            out_size=10,
            width_size=WIDTH,
            depth=DEPTH,
            activation=jax.nn.silu,
            final_activation=lambda x: x,
            key=key,
        )

    def fourier_features(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """Fourier-encode only spatial x and y."""
        # x[0] = normalized alpha
        # x[1] = normalized spatial x
        # x[2] = normalized spatial y

        spatial = x[1:]

        angles = (
            jnp.pi
            * spatial[:, None]
            * FOURIER_FREQUENCIES[
                None,
                :,
            ]
        )

        sin_features = jnp.sin(angles).reshape(-1)

        cos_features = jnp.cos(angles).reshape(-1)

        return jnp.concatenate(
            [
                x,
                sin_features,
                cos_features,
            ]
        )

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """Return ten normalized velocity outputs."""
        return self.mlp(self.fourier_features(x))


# ============================================================
# Load differentiable physical-space predictor
# ============================================================


def load_predictor() -> Callable[[jax.Array], jax.Array]:
    """Load the final V3 model and construct a differentiable physical-space map.

    Input
    -----

    z = [alpha_deg, x, y]

    Output
    ------

    [ux(t1), uy(t1), ..., ux(t5), uy(t5)]

    Input normalization is performed INSIDE the JAX function.
    Therefore JAX differentiates with respect to the physical
    variables:

        alpha_deg,
        x,
        y.
    """
    model_path = MODEL_ROOT / "best_model.eqx"

    normalization_path = MODEL_ROOT / "normalization.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing final model: {model_path}")

    if not normalization_path.exists():
        raise FileNotFoundError(f"Missing normalization metadata: {normalization_path}")

    # --------------------------------------------------------
    # Load output-normalization metadata.
    # --------------------------------------------------------

    with np.load(normalization_path) as normalization:
        y_mean = jnp.asarray(
            normalization["y_mean"].reshape(-1),
            dtype=jnp.float32,
        )

        y_std = jnp.asarray(
            normalization["y_std"].reshape(-1),
            dtype=jnp.float32,
        )

        stored_times = np.asarray(
            normalization["sensor_times"],
            dtype=np.float64,
        )

    if not np.allclose(
        stored_times,
        SENSOR_TIMES,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Final model observation times do not "
            "match wake_surrogate_config.py.\n"
            f"Stored:   {stored_times.tolist()}\n"
            f"Expected: {SENSOR_TIMES.tolist()}"
        )

    # --------------------------------------------------------
    # Reconstruct architecture and load final weights.
    # --------------------------------------------------------

    model = WakeSurrogate(jax.random.PRNGKey(0))

    model = eqx.tree_deserialise_leaves(
        model_path,
        model,
    )

    # --------------------------------------------------------
    # Physical-space prediction map.
    # --------------------------------------------------------

    def predict_physical(
        z: jax.Array,
    ) -> jax.Array:
        """Predict physical velocities from [alpha_deg, x, y]."""
        alpha_deg = z[0]

        x_coord = z[1]

        y_coord = z[2]

        # ----------------------------------------------------
        # Same input normalization used during training.
        # ----------------------------------------------------

        alpha_normalized = 2.0 * alpha_deg / 90.0 - 1.0

        x_normalized = x_coord - 2.0

        y_normalized = y_coord

        normalized_input = jnp.stack(
            [
                alpha_normalized,
                x_normalized,
                y_normalized,
            ]
        )

        # ----------------------------------------------------
        # Neural-network prediction in standardized space.
        # ----------------------------------------------------

        prediction_normalized = model(normalized_input)

        # ----------------------------------------------------
        # Return to physical velocity space.
        # ----------------------------------------------------

        prediction_physical = prediction_normalized * y_std + y_mean

        return prediction_physical

    return jax.jit(predict_physical)


# ============================================================
# Finite differences
# ============================================================


def central_difference(
    predict: Callable[[jax.Array], jax.Array],
    point: jax.Array,
    variable_index: int,
    epsilon: float,
) -> np.ndarray:
    """Compute a central finite-difference derivative of all ten model outputs."""
    plus = point.at[variable_index].add(epsilon)

    minus = point.at[variable_index].add(-epsilon)

    derivative = (predict(plus) - predict(minus)) / (2.0 * epsilon)

    return np.asarray(derivative)


# ============================================================
# Error metrics
# ============================================================


def relative_l2_error(
    reference: np.ndarray,
    approximation: np.ndarray,
) -> float:
    """Compute relative L2 error."""
    numerator = np.linalg.norm(approximation - reference)

    denominator = np.linalg.norm(reference)

    return float(
        numerator
        / max(
            denominator,
            1.0e-12,
        )
    )


def max_absolute_error(
    reference: np.ndarray,
    approximation: np.ndarray,
) -> float:
    """Compute maximum componentwise absolute error."""
    return float(np.max(np.abs(approximation - reference)))


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Measure directional agreement between derivative vectors."""
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < 1.0e-14:
        return float("nan")

    return float(
        np.dot(
            a,
            b,
        )
        / denominator
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Run JAX-AD versus central-FD gradient checks."""
    print("=" * 78)

    print("WAKE SURROGATE GRADIENT CHECK")

    print("=" * 78)

    print(f"Model version    : {MODEL_VERSION}")

    print(f"Model            : {MODEL_LABEL}")

    print(f"JAX devices      : {jax.devices()}")

    print("Precision        : float32")

    print("Fourier encoding : x, y only")

    print("AoA encoding     : raw normalized alpha only")

    print(f"Observation times: {SENSOR_TIMES.tolist()}")

    print("\nChecking physical-space derivatives:")

    print("  d m / d alpha_deg")

    print("  d m / d x")

    print("  d m / d y")

    # ========================================================
    # Differentiable predictor
    # ========================================================

    predict = load_predictor()

    # Full Jacobian:
    #
    # output dimension = 10
    # input dimension  = 3
    #
    # shape = (10, 3)

    jacobian_fn = jax.jit(jax.jacfwd(predict))

    rows = []

    # ========================================================
    # Check all representative points
    # ========================================================

    for (
        alpha_deg,
        x_coord,
        y_coord,
    ) in CHECK_POINTS:
        point = jnp.array(
            [
                alpha_deg,
                x_coord,
                y_coord,
            ],
            dtype=jnp.float32,
        )

        prediction = np.asarray(predict(point))

        jacobian_ad = np.asarray(jacobian_fn(point))

        if jacobian_ad.shape != (
            10,
            3,
        ):
            raise RuntimeError(f"Unexpected Jacobian shape: {jacobian_ad.shape}")

        print("\n" + "-" * 78)

        print(f"Point: alpha={alpha_deg:5.1f} deg, x={x_coord:.3f}, y={y_coord:.3f}")

        print(f"Prediction RMS: {np.sqrt(np.mean(prediction**2)):.6e}")

        # ====================================================
        # Check one physical variable at a time
        # ====================================================

        for (
            variable_index,
            variable_name,
        ) in enumerate(VARIABLE_NAMES):
            derivative_ad = jacobian_ad[
                :,
                variable_index,
            ]

            derivative_norm = float(np.linalg.norm(derivative_ad))

            print(f"\n  d m / d {variable_name}")

            print(f"  JAX derivative norm: {derivative_norm:.6e}")

            best_relative_error = np.inf

            best_epsilon = None

            for epsilon in FD_EPSILONS[variable_name]:
                derivative_fd = central_difference(
                    predict,
                    point,
                    variable_index,
                    epsilon,
                )

                relative_error = relative_l2_error(
                    derivative_ad,
                    derivative_fd,
                )

                maximum_error = max_absolute_error(
                    derivative_ad,
                    derivative_fd,
                )

                cosine = cosine_similarity(
                    derivative_ad,
                    derivative_fd,
                )

                print(
                    f"    eps={epsilon:.1e} "
                    f"| relL2="
                    f"{relative_error:.6e} "
                    f"| maxAbs="
                    f"{maximum_error:.6e} "
                    f"| cos="
                    f"{cosine:.9f}"
                )

                rows.append(
                    {
                        "version": MODEL_VERSION,
                        "alpha_deg": alpha_deg,
                        "x": x_coord,
                        "y": y_coord,
                        "variable": variable_name,
                        "epsilon": epsilon,
                        "ad_derivative_norm": (derivative_norm),
                        "relative_l2_error": (relative_error),
                        "max_absolute_error": (maximum_error),
                        "cosine_similarity": (cosine),
                    }
                )

                if relative_error < best_relative_error:
                    best_relative_error = relative_error

                    best_epsilon = epsilon

            print(f"  BEST: eps={best_epsilon:.1e} | relL2={best_relative_error:.6e}")

    # ========================================================
    # Save detailed results
    # ========================================================

    output_csv = RESULTS_ROOT / "gradient_check.csv"

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()

        writer.writerows(rows)

    # ========================================================
    # Find best FD agreement for each point/variable pair
    # ========================================================

    best_rows = []

    for (
        alpha_deg,
        x_coord,
        y_coord,
    ) in CHECK_POINTS:
        for variable_name in VARIABLE_NAMES:
            matching = [
                row
                for row in rows
                if (
                    row["alpha_deg"] == alpha_deg
                    and row["x"] == x_coord
                    and row["y"] == y_coord
                    and row["variable"] == variable_name
                )
            ]

            best_row = min(
                matching,
                key=lambda row: row["relative_l2_error"],
            )

            best_rows.append(best_row)

    best_errors = np.asarray(
        [row["relative_l2_error"] for row in best_rows],
        dtype=np.float64,
    )

    best_cosines = np.asarray(
        [row["cosine_similarity"] for row in best_rows],
        dtype=np.float64,
    )

    # ========================================================
    # Save best-step summary
    # ========================================================

    summary_csv = RESULTS_ROOT / "gradient_check_best.csv"

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(best_rows[0].keys()),
        )

        writer.writeheader()

        writer.writerows(best_rows)

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 78)

    print("GRADIENT CHECK SUMMARY")

    print("=" * 78)

    print(f"Best relative error, mean: {np.mean(best_errors):.6e}")

    print(f"Best relative error, max:  {np.max(best_errors):.6e}")

    print(f"Best cosine similarity, min: {np.nanmin(best_cosines):.9f}")

    print(f"\nSaved detailed results: {output_csv}")

    print(f"Saved best-step summary: {summary_csv}")

    print("\nGradient check complete.")


if __name__ == "__main__":
    main()
