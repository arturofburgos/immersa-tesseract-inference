"""Evaluate WakeSurrogate models on the held-out validation AoAs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from immersa_tesseract_inference.wake_surrogate_config import (
    SENSOR_TIMES as CONFIG_SENSOR_TIMES,
    VALIDATION_ALPHAS,
)


# ============================================================
# Paths
# ============================================================

DATA_ROOT = Path(
    "data/wake_surrogate"
)

MODEL_ROOT = Path(
    "models/wake_surrogate"
)

RESULTS_ROOT = Path(
    "results/wake_surrogate/model_selection"
)


# ============================================================
# Model versions
# ============================================================

MODEL_CONFIGS = {
    "v1": {
        "label": "V1 Spatial Fourier",
        "directory": "v1_spatial_fourier",
    },
    "v2": {
        "label": "V2 High-AoA Refinement",
        "directory": "v2_high_aoa_refinement",
    },
    "v3": {
        "label": "V3 Broad AoA Refinement",
        "directory": "v3_broad_aoa_refinement",
    },
}


# ============================================================
# Model architecture
#
# V1, V2, and V3 all use the same architecture.
# They differ only in their training-data coverage.
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
# Command-line interface
# ============================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one WakeSurrogate version on "
            "the held-out validation AoAs."
        )
    )

    parser.add_argument(
        "--version",
        required=True,
        choices=(
            "v1",
            "v2",
            "v3",
        ),
        help=(
            "WakeSurrogate version to evaluate."
        ),
    )

    return parser.parse_args()


# ============================================================
# WakeSurrogate architecture
# ============================================================


class WakeSurrogate(eqx.Module):
    """
    MLP with Fourier features only on spatial coordinates.

    Physical input:
        [alpha, x, y]

    Fourier encoding:
        x and y only

    Output:
        [ux(t1), uy(t1), ..., ux(t5), uy(t5)]
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        key: jax.Array,
    ):
        n_original = 3

        n_spatial = 2

        n_fourier = (
            2
            * n_spatial
            * len(
                FOURIER_FREQUENCIES
            )
        )

        feature_size = (
            n_original
            + n_fourier
        )

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
        """
        Keep alpha smooth and Fourier-encode only x and y.
        """

        # x[0] = normalized alpha
        # x[1] = normalized spatial x
        # x[2] = normalized spatial y

        spatial = x[
            1:
        ]

        angles = (
            jnp.pi
            * spatial[:, None]
            * FOURIER_FREQUENCIES[
                None,
                :,
            ]
        )

        sin_features = (
            jnp.sin(
                angles
            ).reshape(-1)
        )

        cos_features = (
            jnp.cos(
                angles
            ).reshape(-1)
        )

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
        """Predict ten normalized velocity values."""

        features = (
            self.fourier_features(
                x
            )
        )

        return self.mlp(
            features
        )


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

    x[:, 0] = (
        2.0
        * x[:, 0]
        / 90.0
        - 1.0
    )

    # x:
    # [1, 3] -> [-1, 1]

    x[:, 1] = (
        x[:, 1]
        - 2.0
    )

    # y already lies in [-1, 1].

    return x


# ============================================================
# Metrics
# ============================================================


def relative_l2(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Compute relative L2 error."""

    numerator = np.linalg.norm(
        prediction
        - truth
    )

    denominator = np.linalg.norm(
        truth
    )

    return float(
        numerator
        / max(
            denominator,
            1.0e-12,
        )
    )


def rmse(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Compute root-mean-square error."""

    return float(
        np.sqrt(
            np.mean(
                (
                    prediction
                    - truth
                )
                ** 2
            )
        )
    )


# ============================================================
# Utilities
# ============================================================


def alpha_tag(
    alpha_deg: float,
) -> str:
    """Convert an AoA into a filesystem-safe tag."""

    return (
        f"{alpha_deg:05.1f}"
        .replace(
            ".",
            "p",
        )
    )


def validate_validation_files(
    validation_files: list[Path],
) -> None:
    """Verify exact validation-set membership."""

    loaded_alphas = []

    for path in validation_files:

        with np.load(
            path
        ) as data:

            alpha_deg = float(
                np.asarray(
                    data["alpha_deg"]
                ).reshape(-1)[0]
            )

        loaded_alphas.append(
            alpha_deg
        )

    loaded = np.asarray(
        sorted(
            loaded_alphas
        ),
        dtype=np.float64,
    )

    expected = np.asarray(
        sorted(
            VALIDATION_ALPHAS
        ),
        dtype=np.float64,
    )

    if (
        loaded.shape
        != expected.shape
    ):
        raise RuntimeError(
            "Incorrect number of validation AoAs.\n"
            f"Expected: {expected.tolist()}\n"
            f"Loaded:   {loaded.tolist()}"
        )

    if not np.allclose(
        loaded,
        expected,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise RuntimeError(
            "Validation AoAs do not match "
            "wake_surrogate_config.py.\n"
            f"Expected: {expected.tolist()}\n"
            f"Loaded:   {loaded.tolist()}"
        )


# ============================================================
# Plotting
# ============================================================


def plot_case(
    model_label: str,
    results_root: Path,
    alpha_deg: float,
    sensor_x: np.ndarray,
    sensor_y: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    """Plot CFD, surrogate, and error at the middle time."""

    x_values = np.unique(
        sensor_x
    )

    y_values = np.unique(
        sensor_y
    )

    nx = len(
        x_values
    )

    ny = len(
        y_values
    )

    # --------------------------------------------------------
    # Middle observation time:
    #
    # t = 15.1
    # --------------------------------------------------------

    time_index = 2

    time_value = (
        SENSOR_TIMES[
            time_index
        ]
    )

    true_ux = (
        truth[
            :,
            time_index,
            0,
        ]
        .reshape(
            ny,
            nx,
        )
    )

    pred_ux = (
        prediction[
            :,
            time_index,
            0,
        ]
        .reshape(
            ny,
            nx,
        )
    )

    true_uy = (
        truth[
            :,
            time_index,
            1,
        ]
        .reshape(
            ny,
            nx,
        )
    )

    pred_uy = (
        prediction[
            :,
            time_index,
            1,
        ]
        .reshape(
            ny,
            nx,
        )
    )

    err_ux = (
        pred_ux
        - true_ux
    )

    err_uy = (
        pred_uy
        - true_uy
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            13,
            7,
        ),
        constrained_layout=True,
    )

    arrays = [
        true_ux,
        pred_ux,
        err_ux,
        true_uy,
        pred_uy,
        err_uy,
    ]

    titles = [
        r"CFD $u_x$",
        r"Surrogate $\hat{u}_x$",
        r"Error $\hat{u}_x-u_x$",
        r"CFD $u_y$",
        r"Surrogate $\hat{u}_y$",
        r"Error $\hat{u}_y-u_y$",
    ]

    for (
        ax,
        field,
        title,
    ) in zip(
        axes.ravel(),
        arrays,
        titles,
    ):

        image = ax.pcolormesh(
            x_values,
            y_values,
            field,
            shading="auto",
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "x"
        )

        ax.set_ylabel(
            "y"
        )

        fig.colorbar(
            image,
            ax=ax,
        )

    fig.suptitle(
        rf"{model_label}: "
        rf"held-out $\alpha={alpha_deg:.1f}^\circ$, "
        rf"$t={time_value:g}$"
    )

    output_path = (
        results_root
        / (
            "validation_alpha_"
            f"{alpha_tag(alpha_deg)}"
            ".png"
        )
    )

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Evaluate one surrogate on the validation set."""

    args = parse_arguments()

    config = MODEL_CONFIGS[
        args.version
    ]

    model_label = config[
        "label"
    ]

    model_root = (
        MODEL_ROOT
        / config[
            "directory"
        ]
    )

    results_root = (
        RESULTS_ROOT
        / config[
            "directory"
        ]
    )

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_root
        / "best_model.eqx"
    )

    normalization_path = (
        model_root
        / "normalization.npz"
    )

    # --------------------------------------------------------
    # Check required model artifacts.
    # --------------------------------------------------------

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}"
        )

    if not normalization_path.exists():
        raise FileNotFoundError(
            "Missing normalization metadata: "
            f"{normalization_path}"
        )

    # ========================================================
    # Header
    # ========================================================

    print(
        "=" * 78
    )

    print(
        "WAKE SURROGATE VALIDATION"
    )

    print(
        "=" * 78
    )

    print(
        f"Version          : "
        f"{args.version}"
    )

    print(
        f"Model            : "
        f"{model_label}"
    )

    print(
        f"Model directory  : "
        f"{model_root}"
    )

    print(
        f"Results directory: "
        f"{results_root}"
    )

    print(
        f"JAX devices      : "
        f"{jax.devices()}"
    )

    print(
        "Fourier encoding : x, y only"
    )

    print(
        "AoA encoding     : "
        "raw normalized alpha only"
    )

    # ========================================================
    # Load normalization metadata
    # ========================================================

    with np.load(
        normalization_path
    ) as normalization:

        y_mean = np.asarray(
            normalization[
                "y_mean"
            ]
        )

        y_std = np.asarray(
            normalization[
                "y_std"
            ]
        )

        stored_times = np.asarray(
            normalization[
                "sensor_times"
            ]
        )

    if not np.allclose(
        stored_times,
        SENSOR_TIMES,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Model normalization metadata contains "
            "unexpected observation times.\n"
            f"Stored:   {stored_times.tolist()}\n"
            f"Expected: {SENSOR_TIMES.tolist()}"
        )

    # ========================================================
    # Reconstruct architecture and load weights
    # ========================================================

    model = WakeSurrogate(
        jax.random.PRNGKey(
            0
        )
    )

    model = (
        eqx.tree_deserialise_leaves(
            model_path,
            model,
        )
    )

    predict_batch = jax.jit(
        jax.vmap(
            model
        )
    )

    # ========================================================
    # Validation files
    # ========================================================

    validation_files = sorted(
        (
            DATA_ROOT
            / "validation"
        ).glob(
            "*.npz"
        )
    )

    if not validation_files:
        raise RuntimeError(
            "No validation files found."
        )

    validate_validation_files(
        validation_files
    )

    rows = []

    print(
        "\nPer-AoA validation:"
    )

    print(
        "-" * 78
    )

    # ========================================================
    # Evaluate each validation AoA
    # ========================================================

    for path in validation_files:

        with np.load(
            path
        ) as data:

            alpha_deg = float(
                np.asarray(
                    data[
                        "alpha_deg"
                    ]
                ).reshape(-1)[0]
            )

            inputs_physical = np.asarray(
                data[
                    "inputs"
                ]
            )

            targets_physical = np.asarray(
                data[
                    "targets"
                ]
            )

            sensor_x = np.asarray(
                data[
                    "sensor_x"
                ]
            )

            sensor_y = np.asarray(
                data[
                    "sensor_y"
                ]
            )

            file_times = np.asarray(
                data[
                    "sensor_times"
                ]
            )

        if not np.allclose(
            file_times,
            SENSOR_TIMES,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"Unexpected observation times in {path}."
            )

        # ----------------------------------------------------
        # Normalize inputs
        # ----------------------------------------------------

        inputs_normalized = (
            normalize_inputs(
                inputs_physical
            )
        )

        # ----------------------------------------------------
        # Predict in normalized target space
        # ----------------------------------------------------

        prediction_normalized = np.asarray(
            predict_batch(
                jnp.asarray(
                    inputs_normalized,
                    dtype=jnp.float32,
                )
            )
        )

        # ----------------------------------------------------
        # Return to physical velocity space
        # ----------------------------------------------------

        prediction_physical = (
            prediction_normalized
            * y_std
            + y_mean
        )

        truth = (
            targets_physical.reshape(
                -1,
                len(
                    SENSOR_TIMES
                ),
                2,
            )
        )

        prediction = (
            prediction_physical.reshape(
                -1,
                len(
                    SENSOR_TIMES
                ),
                2,
            )
        )

        # ====================================================
        # Metrics
        # ====================================================

        overall_rmse = rmse(
            prediction,
            truth,
        )

        ux_rmse = rmse(
            prediction[
                :,
                :,
                0,
            ],
            truth[
                :,
                :,
                0,
            ],
        )

        uy_rmse = rmse(
            prediction[
                :,
                :,
                1,
            ],
            truth[
                :,
                :,
                1,
            ],
        )

        rel_error = relative_l2(
            prediction,
            truth,
        )

        ux_rel_error = relative_l2(
            prediction[
                :,
                :,
                0,
            ],
            truth[
                :,
                :,
                0,
            ],
        )

        uy_rel_error = relative_l2(
            prediction[
                :,
                :,
                1,
            ],
            truth[
                :,
                :,
                1,
            ],
        )

        print(
            f"alpha={alpha_deg:5.1f} "
            f"| RMSE={overall_rmse:.6f} "
            f"| ux={ux_rmse:.6f} "
            f"| uy={uy_rmse:.6f} "
            f"| relL2="
            f"{100.0 * rel_error:6.2f}%"
        )

        row = {
            "version": args.version,
            "alpha_deg": alpha_deg,
            "rmse": overall_rmse,
            "ux_rmse": ux_rmse,
            "uy_rmse": uy_rmse,
            "relative_l2_percent": (
                100.0
                * rel_error
            ),
            "ux_relative_l2_percent": (
                100.0
                * ux_rel_error
            ),
            "uy_relative_l2_percent": (
                100.0
                * uy_rel_error
            ),
        }

        # ----------------------------------------------------
        # Error separately at each observation time
        # ----------------------------------------------------

        for (
            time_index,
            time_value,
        ) in enumerate(
            SENSOR_TIMES
        ):

            row[
                f"rmse_t_{time_value:.1f}"
            ] = rmse(
                prediction[
                    :,
                    time_index,
                    :,
                ],
                truth[
                    :,
                    time_index,
                    :,
                ],
            )

            row[
                f"relative_l2_percent_t_{time_value:.1f}"
            ] = (
                100.0
                * relative_l2(
                    prediction[
                        :,
                        time_index,
                        :,
                    ],
                    truth[
                        :,
                        time_index,
                        :,
                    ],
                )
            )

        rows.append(
            row
        )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        plot_case(
            model_label,
            results_root,
            alpha_deg,
            sensor_x,
            sensor_y,
            truth,
            prediction,
        )

    # ========================================================
    # Save detailed metrics
    # ========================================================

    output_csv = (
        results_root
        / "validation_metrics.csv"
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # Summary statistics
    # ========================================================

    mean_rmse = float(
        np.mean(
            [
                row[
                    "rmse"
                ]
                for row
                in rows
            ]
        )
    )

    mean_rel = float(
        np.mean(
            [
                row[
                    "relative_l2_percent"
                ]
                for row
                in rows
            ]
        )
    )

    mean_ux_rel = float(
        np.mean(
            [
                row[
                    "ux_relative_l2_percent"
                ]
                for row
                in rows
            ]
        )
    )

    mean_uy_rel = float(
        np.mean(
            [
                row[
                    "uy_relative_l2_percent"
                ]
                for row
                in rows
            ]
        )
    )

    print(
        "\n"
        + "-" * 78
    )

    print(
        f"Mean validation RMSE:        "
        f"{mean_rmse:.6f}"
    )

    print(
        f"Mean validation relative L2: "
        f"{mean_rel:.2f}%"
    )

    print(
        f"Mean ux relative L2:         "
        f"{mean_ux_rel:.2f}%"
    )

    print(
        f"Mean uy relative L2:         "
        f"{mean_uy_rel:.2f}%"
    )

    print(
        f"\nSaved metrics: "
        f"{output_csv}"
    )

    print(
        f"Saved figures: "
        f"{results_root}"
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "VALIDATION COMPLETE"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()