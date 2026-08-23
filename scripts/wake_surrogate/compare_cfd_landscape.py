"""Compare final WakeSurrogate and real-CFD AoA objective landscapes."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from immersa_tesseract_inference.wake_surrogate_config import (
    SENSOR_TIMES as CONFIG_SENSOR_TIMES,
)
from immersa_tesseract_inference.wake_surrogate_config import (
    TEST_ALPHAS,
)

# ============================================================
# Final model
# ============================================================

MODEL_VERSION = "v3"

MODEL_LABEL = "V3 Broad AoA Refinement"


# ============================================================
# Paths
# ============================================================

MODEL_ROOT = Path("models/wake_surrogate/v3_broad_aoa_refinement")

CFD_CSV = Path(
    "results/identifiability/"
    "sensor_count/"
    "nonuniform_sensor_count_landscape/"
    "sensor_count_loss_landscapes.csv"
)

RESULTS_ROOT = Path("results/wake_surrogate/final_validation/inverse_landscape")

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Observation configuration
# ============================================================

if len(TEST_ALPHAS) != 1:
    raise RuntimeError(
        f"Expected exactly one configured test AoA, found {len(TEST_ALPHAS)}."
    )

TRUTH_ALPHA = float(TEST_ALPHAS[0])

SENSOR_TIMES = np.asarray(
    CONFIG_SENSOR_TIMES,
    dtype=np.float64,
)

TEST_FILE = Path("data/wake_surrogate/test") / (
    "alpha_"
    + f"{TRUTH_ALPHA:05.1f}".replace(
        ".",
        "p",
    )
    + ".npz"
)


# Same Ns = 2 configuration used in the previous
# real-CFD identifiability experiment.

SENSOR_X = np.array(
    [
        1.0,
        1.0,
    ],
    dtype=np.float64,
)

SENSOR_Y = np.array(
    [
        -0.4,
        0.4,
    ],
    dtype=np.float64,
)


# Dense surrogate scan.

ALPHA_DENSE = np.arange(
    20.0,
    85.0 + 0.125,
    0.25,
    dtype=np.float64,
)


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
# Input normalization
# ============================================================


def normalize_inputs(
    x: np.ndarray,
) -> np.ndarray:
    """Normalize physical inputs exactly as during training."""
    x = x.copy()

    # alpha:
    # [0, 90] -> [-1, 1]

    x[:, 0] = 2.0 * x[:, 0] / 90.0 - 1.0

    # spatial x:
    # [1, 3] -> [-1, 1]

    x[:, 1] = x[:, 1] - 2.0

    # y already lies in [-1, 1].

    return x


# ============================================================
# CFD landscape loading
# ============================================================


def load_cfd_ns2_landscape(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Load the existing real-CFD Ns=2 objective landscape.

    The CSV is stored in long format with one row per
    (alpha, sensor-count) configuration.

    Returns:
    -------
    alpha:
        CFD AoA samples for Ns=2.

    j_per_scalar:
        Normalized inverse objective J / N_scalar.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing CFD landscape CSV: {path}")

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise RuntimeError(f"No CSV header found in {path}.")

        rows = list(reader)

    print(
        "CFD CSV columns:",
        fieldnames,
    )

    required_columns = {
        "alpha_deg",
        "n_sensors",
        "n_times",
        "n_scalar",
        "objective_per_scalar",
    }

    missing = required_columns - set(fieldnames)

    if missing:
        raise RuntimeError(f"Missing required CFD columns: {sorted(missing)}")

    # --------------------------------------------------------
    # Select exactly the Ns = 2 experiment.
    # --------------------------------------------------------

    ns2_rows = [row for row in rows if (int(float(row["n_sensors"])) == len(SENSOR_X))]

    if not ns2_rows:
        raise RuntimeError("No Ns=2 rows found in CFD landscape CSV.")

    print(f"Found {len(ns2_rows)} CFD rows for Ns=2.")

    # --------------------------------------------------------
    # Sanity-check observation configuration.
    #
    # Ns = 2 sensors
    # Nt = 5 times
    # 2 velocity components
    #
    # N_scalar = 2 * 5 * 2 = 20
    # --------------------------------------------------------

    expected_n_times = len(SENSOR_TIMES)

    expected_n_scalar = len(SENSOR_X) * expected_n_times * 2

    for row in ns2_rows:
        n_times = int(float(row["n_times"]))

        n_scalar = int(float(row["n_scalar"]))

        if n_times != expected_n_times:
            raise RuntimeError(
                "Unexpected number of times "
                f"for alpha="
                f"{row['alpha_deg']}: "
                f"{n_times}; "
                f"expected "
                f"{expected_n_times}."
            )

        if n_scalar != expected_n_scalar:
            raise RuntimeError(
                "Unexpected number of scalar "
                "observations for alpha="
                f"{row['alpha_deg']}: "
                f"{n_scalar}; "
                f"expected "
                f"{expected_n_scalar}."
            )

    # --------------------------------------------------------
    # Extract alpha and normalized objective.
    # --------------------------------------------------------

    alpha = np.array(
        [float(row["alpha_deg"]) for row in ns2_rows],
        dtype=np.float64,
    )

    j_per_scalar = np.array(
        [float(row["objective_per_scalar"]) for row in ns2_rows],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Sort by AoA.
    # --------------------------------------------------------

    order = np.argsort(alpha)

    alpha = alpha[order]

    j_per_scalar = j_per_scalar[order]

    print(f"CFD Ns=2 alpha range: {alpha.min():.1f} to {alpha.max():.1f} deg")

    print(f"CFD Ns=2 samples: {len(alpha)}")

    return (
        alpha,
        j_per_scalar,
    )


# ============================================================
# Extract real-CFD truth measurements at the test AoA
# ============================================================


def load_cfd_truth_measurements() -> np.ndarray:
    """Extract real-CFD measurements at the two Ns=2 sensor positions.

    Shape:
        (2 sensors, Nt times, 2 velocity components)
    """
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing sealed test dataset: {TEST_FILE}")

    with np.load(TEST_FILE) as data:
        alpha_deg = float(np.asarray(data["alpha_deg"]).reshape(-1)[0])

        sensor_x_all = np.asarray(data["sensor_x"])

        sensor_y_all = np.asarray(data["sensor_y"])

        sensor_times = np.asarray(data["sensor_times"])

        measurements_all = np.asarray(data["measurements"])

    if not np.isclose(
        alpha_deg,
        TRUTH_ALPHA,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise RuntimeError(
            f"Unexpected test AoA.\nExpected: {TRUTH_ALPHA}\nLoaded:   {alpha_deg}"
        )

    if not np.allclose(
        sensor_times,
        SENSOR_TIMES,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Test-file observation times do not match wake_surrogate_config.py."
        )

    measurements = []

    for (
        sensor_x,
        sensor_y,
    ) in zip(
        SENSOR_X,
        SENSOR_Y,
        strict=True,
    ):
        matches = np.where(
            np.isclose(
                sensor_x_all,
                sensor_x,
                rtol=0.0,
                atol=1.0e-12,
            )
            & np.isclose(
                sensor_y_all,
                sensor_y,
                rtol=0.0,
                atol=1.0e-12,
            )
        )[0]

        if len(matches) != 1:
            raise RuntimeError(
                "Expected exactly one dataset "
                "point at "
                f"({sensor_x}, {sensor_y}), "
                f"found {len(matches)}."
            )

        measurements.append(measurements_all[matches[0]])

    result = np.stack(
        measurements,
        axis=0,
    )

    expected_shape = (
        len(SENSOR_X),
        len(SENSOR_TIMES),
        2,
    )

    if result.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected truth shape {result.shape}; expected {expected_shape}."
        )

    return result


# ============================================================
# Surrogate prediction
# ============================================================


def make_predictor() -> Callable[[float], np.ndarray]:
    """Load the final V3 surrogate and return an Ns=2 prediction function."""
    model_path = MODEL_ROOT / "best_model.eqx"

    normalization_path = MODEL_ROOT / "normalization.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing final model: {model_path}")

    if not normalization_path.exists():
        raise FileNotFoundError(f"Missing normalization metadata: {normalization_path}")

    with np.load(normalization_path) as normalization:
        y_mean = np.asarray(normalization["y_mean"])

        y_std = np.asarray(normalization["y_std"])

        stored_times = np.asarray(normalization["sensor_times"])

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

    model = WakeSurrogate(jax.random.PRNGKey(0))

    model = eqx.tree_deserialise_leaves(
        model_path,
        model,
    )

    predict_batch = jax.jit(jax.vmap(model))

    def predict(
        alpha_deg: float,
    ) -> np.ndarray:
        """Predict Ns=2 physical wake measurements.

        Returns:
            (2 sensors, Nt times, 2 components)
        """
        inputs = np.column_stack(
            [
                np.full(
                    len(SENSOR_X),
                    alpha_deg,
                    dtype=np.float64,
                ),
                SENSOR_X,
                SENSOR_Y,
            ]
        )

        inputs = normalize_inputs(inputs)

        prediction_normalized = np.asarray(
            predict_batch(
                jnp.asarray(
                    inputs,
                    dtype=jnp.float32,
                )
            )
        )

        prediction_physical = prediction_normalized * y_std + y_mean

        return prediction_physical.reshape(
            len(SENSOR_X),
            len(SENSOR_TIMES),
            2,
        )

    return predict


# ============================================================
# Objective
# ============================================================


def normalized_objective(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute J / N_scalar.

    J = 0.5 ||prediction - target||_2^2

    Therefore:

        J / N_scalar
        = 0.5 * mean(residual^2)
    """
    residual = prediction - target

    return float(0.5 * np.mean(residual**2))


# ============================================================
# Local minima
# ============================================================


def find_grid_local_minima(
    alpha: np.ndarray,
    objective: np.ndarray,
) -> list[
    tuple[
        float,
        float,
    ]
]:
    """Find simple interior grid-local minima."""
    minima = []

    for i in range(
        1,
        len(alpha) - 1,
    ):
        if objective[i] < objective[i - 1] and objective[i] < objective[i + 1]:
            minima.append(
                (
                    float(alpha[i]),
                    float(objective[i]),
                )
            )

    return minima


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Compare real-CFD and final-surrogate inverse landscapes."""
    print("=" * 78)

    print("WAKE SURROGATE VS REAL-CFD INVERSE LANDSCAPE")

    print("=" * 78)

    print(f"Model version : {MODEL_VERSION}")

    print(f"Model         : {MODEL_LABEL}")

    print(f"JAX devices   : {jax.devices()}")

    print(f"Sensors x     : {SENSOR_X.tolist()}")

    print(f"Sensors y     : {SENSOR_Y.tolist()}")

    print(f"Times         : {SENSOR_TIMES.tolist()}")

    print(f"Truth AoA     : {TRUTH_ALPHA} deg")

    # ========================================================
    # Load original real-CFD landscape
    # ========================================================

    (
        cfd_alpha,
        cfd_objective,
    ) = load_cfd_ns2_landscape(CFD_CSV)

    # ========================================================
    # Load REAL CFD target measurements at truth AoA
    # ========================================================

    cfd_truth = load_cfd_truth_measurements()

    # ========================================================
    # Load final V3 surrogate
    # ========================================================

    predict = make_predictor()

    surrogate_truth = predict(TRUTH_ALPHA)

    # ========================================================
    # Dense surrogate landscape
    #
    # Curve 1:
    #
    # surrogate candidate compared against REAL CFD truth.
    #
    # This is the strictest comparison.
    # ========================================================

    surrogate_cfd_truth = np.array(
        [
            normalized_objective(
                predict(alpha),
                cfd_truth,
            )
            for alpha in ALPHA_DENSE
        ],
        dtype=np.float64,
    )

    # ========================================================
    # Curve 2:
    #
    # surrogate candidate compared against surrogate truth
    # at alpha = TRUTH_ALPHA.
    #
    # This isolates the intrinsic topology of the learned map.
    # ========================================================

    surrogate_self_truth = np.array(
        [
            normalized_objective(
                predict(alpha),
                surrogate_truth,
            )
            for alpha in ALPHA_DENSE
        ],
        dtype=np.float64,
    )

    # ========================================================
    # Evaluate surrogate exactly at CFD AoAs for quantitative
    # curve comparison
    # ========================================================

    surrogate_at_cfd_alpha = np.array(
        [
            normalized_objective(
                predict(alpha),
                cfd_truth,
            )
            for alpha in cfd_alpha
        ],
        dtype=np.float64,
    )

    curve_rmse = float(np.sqrt(np.mean((surrogate_at_cfd_alpha - cfd_objective) ** 2)))

    cfd_range = float(np.max(cfd_objective) - np.min(cfd_objective))

    normalized_curve_rmse = curve_rmse / max(
        cfd_range,
        1.0e-12,
    )

    correlation = float(
        np.corrcoef(
            cfd_objective,
            surrogate_at_cfd_alpha,
        )[
            0,
            1,
        ]
    )

    # ========================================================
    # Local minima
    # ========================================================

    cfd_minima = find_grid_local_minima(
        cfd_alpha,
        cfd_objective,
    )

    surrogate_minima = find_grid_local_minima(
        ALPHA_DENSE,
        surrogate_cfd_truth,
    )

    surrogate_self_minima = find_grid_local_minima(
        ALPHA_DENSE,
        surrogate_self_truth,
    )

    print("\nCFD local minima:")

    for (
        alpha,
        value,
    ) in cfd_minima:
        print(f"  alpha={alpha:6.2f} deg | J/N={value:.8e}")

    print(f"\n{MODEL_VERSION.upper()} local minima (using REAL CFD truth):")

    for (
        alpha,
        value,
    ) in surrogate_minima:
        print(f"  alpha={alpha:6.2f} deg | J/N={value:.8e}")

    print(f"\n{MODEL_VERSION.upper()} local minima (using surrogate truth):")

    for (
        alpha,
        value,
    ) in surrogate_self_minima:
        print(f"  alpha={alpha:6.2f} deg | J/N={value:.8e}")

    print("\nCurve agreement:")

    print(f"  RMSE in J/N:       {curve_rmse:.8e}")

    print(f"  RMSE / CFD range:  {100.0 * normalized_curve_rmse:.2f}%")

    print(f"  Pearson corr.:     {correlation:.6f}")

    # ========================================================
    # Save dense surrogate curves
    # ========================================================

    output_csv = RESULTS_ROOT / "surrogate_landscape.csv"

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "alpha_deg",
                "surrogate_J_per_scalar_cfd_truth",
                "surrogate_J_per_scalar_self_truth",
            ]
        )

        for (
            alpha,
            j_cfd_truth,
            j_self_truth,
        ) in zip(
            ALPHA_DENSE,
            surrogate_cfd_truth,
            surrogate_self_truth,
            strict=True,
        ):
            writer.writerow(
                [
                    alpha,
                    j_cfd_truth,
                    j_self_truth,
                ]
            )

    # ========================================================
    # Save curve-comparison metrics
    # ========================================================

    metrics_csv = RESULTS_ROOT / "landscape_metrics.csv"

    with metrics_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "metric",
                "value",
            ]
        )

        writer.writerow(
            [
                "curve_rmse_J_per_scalar",
                curve_rmse,
            ]
        )

        writer.writerow(
            [
                "curve_rmse_percent_cfd_range",
                (100.0 * normalized_curve_rmse),
            ]
        )

        writer.writerow(
            [
                "pearson_correlation",
                correlation,
            ]
        )

    # ========================================================
    # Save local minima
    # ========================================================

    minima_csv = RESULTS_ROOT / "local_minima.csv"

    with minima_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "curve",
                "alpha_deg",
                "J_per_scalar",
            ]
        )

        for (
            alpha,
            value,
        ) in cfd_minima:
            writer.writerow(
                [
                    "real_cfd",
                    alpha,
                    value,
                ]
            )

        for (
            alpha,
            value,
        ) in surrogate_minima:
            writer.writerow(
                [
                    "surrogate_vs_cfd_truth",
                    alpha,
                    value,
                ]
            )

        for (
            alpha,
            value,
        ) in surrogate_self_minima:
            writer.writerow(
                [
                    "surrogate_self_truth",
                    alpha,
                    value,
                ]
            )

    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            9,
            5.5,
        )
    )

    ax.plot(
        cfd_alpha,
        cfd_objective,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Immersa CFD",
    )

    ax.plot(
        ALPHA_DENSE,
        surrogate_cfd_truth,
        linewidth=2.0,
        label=("V3 surrogate vs CFD truth"),
    )

    ax.plot(
        ALPHA_DENSE,
        surrogate_self_truth,
        linestyle="--",
        linewidth=1.5,
        label=("V3 surrogate self-reference"),
    )

    ax.axvline(
        TRUTH_ALPHA,
        linestyle="--",
        linewidth=1.2,
        label=(
            rf"True $\alpha="
            rf"{TRUTH_ALPHA:g}^\circ$"
        ),
    )

    ax.set_xlabel(
        r"Angle of attack, "
        r"$\alpha$ [deg]"
    )

    ax.set_ylabel(
        r"Normalized objective, "
        r"$J/N_{\mathrm{scalar}}$"
    )

    ax.set_title("Inverse-objective landscape: Immersa CFD vs WakeSurrogate V3")

    ax.grid(alpha=0.25)

    ax.legend()

    figure_png = RESULTS_ROOT / "cfd_vs_surrogate_landscape.png"

    figure_pdf = RESULTS_ROOT / "cfd_vs_surrogate_landscape.pdf"

    fig.savefig(
        figure_png,
        dpi=200,
        bbox_inches="tight",
    )

    fig.savefig(
        figure_pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # Final summary
    # ========================================================

    print(f"\nSaved surrogate landscape: {output_csv}")

    print(f"Saved landscape metrics:   {metrics_csv}")

    print(f"Saved local minima:        {minima_csv}")

    print(f"Saved PNG:                 {figure_png}")

    print(f"Saved PDF:                 {figure_pdf}")

    print("\n" + "=" * 78)

    print("LANDSCAPE COMPARISON COMPLETE")

    print("=" * 78)


if __name__ == "__main__":
    main()
