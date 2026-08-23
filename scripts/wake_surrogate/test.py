"""Evaluate the final WakeSurrogate on the sealed test AoA."""

from __future__ import annotations

import csv
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from immersa_tesseract_inference.wake_surrogate_config import (
    SENSOR_TIMES as CONFIG_SENSOR_TIMES,
    TEST_ALPHAS,
)


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path(
    "data/wake_surrogate"
)

MODEL_ROOT = Path(
    "models/wake_surrogate/"
    "v3_broad_aoa_refinement"
)

RESULTS_ROOT = Path(
    "results/wake_surrogate/"
    "final_validation/"
    "test_alpha63"
)

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Final model
# ============================================================

MODEL_VERSION = "v3"

MODEL_LABEL = (
    "V3 Broad AoA Refinement"
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

SENSOR_TIMES = np.asarray(
    CONFIG_SENSOR_TIMES,
    dtype=np.float64,
)


# Plot all observation times.
PLOT_TIME_INDICES = list(
    range(
        len(
            SENSOR_TIMES
        )
    )
)


# ============================================================
# WakeSurrogate architecture
#
# Must match the architecture used during V3 training.
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

    # spatial x:
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
# Utility
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


def time_tag(
    time_value: float,
) -> str:
    """Convert an observation time into a safe tag."""

    return (
        f"{time_value:04.1f}"
        .replace(
            ".",
            "p",
        )
    )


# ============================================================
# Plotting
# ============================================================


def plot_case_all_times(
    alpha_deg: float,
    sensor_x: np.ndarray,
    sensor_y: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    time_indices: list[int],
) -> None:
    """
    Plot CFD, surrogate, and error for requested times.
    """

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

    for time_index in time_indices:

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

        # ----------------------------------------------------
        # Identical color limits for CFD and surrogate.
        # ----------------------------------------------------

        ux_vmin = min(
            float(
                np.min(
                    true_ux
                )
            ),
            float(
                np.min(
                    pred_ux
                )
            ),
        )

        ux_vmax = max(
            float(
                np.max(
                    true_ux
                )
            ),
            float(
                np.max(
                    pred_ux
                )
            ),
        )

        uy_vmin = min(
            float(
                np.min(
                    true_uy
                )
            ),
            float(
                np.min(
                    pred_uy
                )
            ),
        )

        uy_vmax = max(
            float(
                np.max(
                    true_uy
                )
            ),
            float(
                np.max(
                    pred_uy
                )
            ),
        )

        # ----------------------------------------------------
        # Symmetric color limits for errors.
        # ----------------------------------------------------

        ux_error_limit = max(
            abs(
                float(
                    np.min(
                        err_ux
                    )
                )
            ),
            abs(
                float(
                    np.max(
                        err_ux
                    )
                )
            ),
        )

        uy_error_limit = max(
            abs(
                float(
                    np.min(
                        err_uy
                    )
                )
            ),
            abs(
                float(
                    np.max(
                        err_uy
                    )
                )
            ),
        )

        ux_error_limit = max(
            ux_error_limit,
            1.0e-12,
        )

        uy_error_limit = max(
            uy_error_limit,
            1.0e-12,
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

        # ====================================================
        # ux row
        # ====================================================

        image = axes[
            0,
            0,
        ].pcolormesh(
            x_values,
            y_values,
            true_ux,
            shading="auto",
            vmin=ux_vmin,
            vmax=ux_vmax,
        )

        axes[
            0,
            0,
        ].set_title(
            r"CFD $u_x$"
        )

        fig.colorbar(
            image,
            ax=axes[
                0,
                0,
            ],
        )

        image = axes[
            0,
            1,
        ].pcolormesh(
            x_values,
            y_values,
            pred_ux,
            shading="auto",
            vmin=ux_vmin,
            vmax=ux_vmax,
        )

        axes[
            0,
            1,
        ].set_title(
            r"Surrogate $\hat{u}_x$"
        )

        fig.colorbar(
            image,
            ax=axes[
                0,
                1,
            ],
        )

        image = axes[
            0,
            2,
        ].pcolormesh(
            x_values,
            y_values,
            err_ux,
            shading="auto",
            vmin=-ux_error_limit,
            vmax=ux_error_limit,
        )

        axes[
            0,
            2,
        ].set_title(
            r"Error $\hat{u}_x-u_x$"
        )

        fig.colorbar(
            image,
            ax=axes[
                0,
                2,
            ],
        )

        # ====================================================
        # uy row
        # ====================================================

        image = axes[
            1,
            0,
        ].pcolormesh(
            x_values,
            y_values,
            true_uy,
            shading="auto",
            vmin=uy_vmin,
            vmax=uy_vmax,
        )

        axes[
            1,
            0,
        ].set_title(
            r"CFD $u_y$"
        )

        fig.colorbar(
            image,
            ax=axes[
                1,
                0,
            ],
        )

        image = axes[
            1,
            1,
        ].pcolormesh(
            x_values,
            y_values,
            pred_uy,
            shading="auto",
            vmin=uy_vmin,
            vmax=uy_vmax,
        )

        axes[
            1,
            1,
        ].set_title(
            r"Surrogate $\hat{u}_y$"
        )

        fig.colorbar(
            image,
            ax=axes[
                1,
                1,
            ],
        )

        image = axes[
            1,
            2,
        ].pcolormesh(
            x_values,
            y_values,
            err_uy,
            shading="auto",
            vmin=-uy_error_limit,
            vmax=uy_error_limit,
        )

        axes[
            1,
            2,
        ].set_title(
            r"Error $\hat{u}_y-u_y$"
        )

        fig.colorbar(
            image,
            ax=axes[
                1,
                2,
            ],
        )

        # ----------------------------------------------------
        # Axis labels
        # ----------------------------------------------------

        for ax in axes.ravel():

            ax.set_xlabel(
                "x"
            )

            ax.set_ylabel(
                "y"
            )

        fig.suptitle(
            rf"{MODEL_LABEL}: unseen test "
            rf"$\alpha={alpha_deg:.1f}^\circ$, "
            rf"$t={time_value:.1f}$"
        )

        output_path = (
            RESULTS_ROOT
            / (
                f"test_alpha_"
                f"{alpha_tag(alpha_deg)}"
                f"_t_"
                f"{time_tag(time_value)}"
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

        print(
            f"Saved figure: "
            f"{output_path}"
        )


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Evaluate final V3 on the sealed test case."""

    # --------------------------------------------------------
    # The experiment intentionally has one sealed test AoA.
    # --------------------------------------------------------

    if len(
        TEST_ALPHAS
    ) != 1:
        raise RuntimeError(
            "Expected exactly one configured test AoA, "
            f"found {len(TEST_ALPHAS)}."
        )

    expected_test_alpha = float(
        TEST_ALPHAS[
            0
        ]
    )

    model_path = (
        MODEL_ROOT
        / "best_model.eqx"
    )

    normalization_path = (
        MODEL_ROOT
        / "normalization.npz"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing final model: "
            f"{model_path}"
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
        "WAKE SURROGATE FINAL TEST"
    )

    print(
        "=" * 78
    )

    print(
        f"Model version    : "
        f"{MODEL_VERSION}"
    )

    print(
        f"Model            : "
        f"{MODEL_LABEL}"
    )

    print(
        f"Test AoA         : "
        f"{expected_test_alpha}"
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

    print(
        f"Observation times: "
        f"{SENSOR_TIMES.tolist()}"
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
            "Final model observation times do not "
            "match wake_surrogate_config.py.\n"
            f"Stored:   {stored_times.tolist()}\n"
            f"Expected: {SENSOR_TIMES.tolist()}"
        )

    # ========================================================
    # Load final V3 weights
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
    # Load sealed test case
    # ========================================================

    test_files = sorted(
        (
            DATA_ROOT
            / "test"
        ).glob(
            "*.npz"
        )
    )

    if len(
        test_files
    ) != 1:
        raise RuntimeError(
            "Expected exactly one test file, "
            f"found {len(test_files)}."
        )

    path = test_files[
        0
    ]

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

    # --------------------------------------------------------
    # Verify sealed-test identity.
    # --------------------------------------------------------

    if not np.isclose(
        alpha_deg,
        expected_test_alpha,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise RuntimeError(
            "Unexpected test AoA.\n"
            f"Expected: {expected_test_alpha}\n"
            f"Loaded:   {alpha_deg}"
        )

    if not np.allclose(
        file_times,
        SENSOR_TIMES,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            f"Unexpected observation times in "
            f"{path}."
        )

    print(
        "\nSealed test case:"
    )

    print(
        "-" * 78
    )

    # ========================================================
    # Normalize model inputs
    # ========================================================

    inputs_normalized = (
        normalize_inputs(
            inputs_physical
        )
    )

    # ========================================================
    # Surrogate prediction
    # ========================================================

    prediction_normalized = np.asarray(
        predict_batch(
            jnp.asarray(
                inputs_normalized,
                dtype=jnp.float32,
            )
        )
    )

    # --------------------------------------------------------
    # Convert back to physical velocity space.
    # --------------------------------------------------------

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

    # ========================================================
    # Global metrics
    # ========================================================

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

    print(
        f"                     "
        f"ux relL2="
        f"{100.0 * ux_rel_error:6.2f}% "
        f"| uy relL2="
        f"{100.0 * uy_rel_error:6.2f}%"
    )

    row = {
        "version": MODEL_VERSION,
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

    # ========================================================
    # Per-time metrics
    # ========================================================

    print(
        "\nPer-time metrics:"
    )

    print(
        "-" * 78
    )

    for (
        time_index,
        time_value,
    ) in enumerate(
        SENSOR_TIMES
    ):

        time_prediction = (
            prediction[
                :,
                time_index,
                :,
            ]
        )

        time_truth = (
            truth[
                :,
                time_index,
                :,
            ]
        )

        time_rmse = rmse(
            time_prediction,
            time_truth,
        )

        time_rel = relative_l2(
            time_prediction,
            time_truth,
        )

        ux_time_rmse = rmse(
            prediction[
                :,
                time_index,
                0,
            ],
            truth[
                :,
                time_index,
                0,
            ],
        )

        uy_time_rmse = rmse(
            prediction[
                :,
                time_index,
                1,
            ],
            truth[
                :,
                time_index,
                1,
            ],
        )

        print(
            f"t={time_value:4.1f} "
            f"| RMSE={time_rmse:.6f} "
            f"| relL2="
            f"{100.0 * time_rel:6.2f}% "
            f"| ux={ux_time_rmse:.6f} "
            f"| uy={uy_time_rmse:.6f}"
        )

        row[
            f"rmse_t_{time_value:.1f}"
        ] = time_rmse

        row[
            (
                "relative_l2_t_"
                f"{time_value:.1f}"
                "_percent"
            )
        ] = (
            100.0
            * time_rel
        )

    # ========================================================
    # Save metrics
    # ========================================================

    output_csv = (
        RESULTS_ROOT
        / "test_metrics.csv"
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                row.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(
            row
        )

    # ========================================================
    # Figures
    # ========================================================

    plot_case_all_times(
        alpha_deg,
        sensor_x,
        sensor_y,
        truth,
        prediction,
        time_indices=PLOT_TIME_INDICES,
    )

    # ========================================================
    # Final summary
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        f"Saved test metrics: "
        f"{output_csv}"
    )

    print(
        f"Saved test figures: "
        f"{RESULTS_ROOT}"
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FINAL TEST COMPLETE"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()