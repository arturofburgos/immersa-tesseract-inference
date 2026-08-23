"""Sanity checks for the complete WakeSurrogate CFD dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from immersa_tesseract_inference.wake_surrogate_config import (
    SENSOR_TIMES,
    TEST_ALPHAS,
    TRAIN_ALPHAS_BY_VERSION,
    VALIDATION_ALPHAS,
)

# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("data/wake_surrogate")

RESULTS_ROOT = Path("results/wake_surrogate/dataset_validation")

RESULTS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Dataset version
#
# The complete generated CFD dataset corresponds to V3.
#
# V1 and V2 use subsets of this training pool.
# ============================================================

DATASET_VERSION = "v3"


EXPECTED_ALPHAS = {
    "train": tuple(TRAIN_ALPHAS_BY_VERSION[DATASET_VERSION]),
    "validation": tuple(VALIDATION_ALPHAS),
    "test": tuple(TEST_ALPHAS),
}


EXPECTED_COUNTS = {split: len(alphas) for split, alphas in EXPECTED_ALPHAS.items()}


# ============================================================
# Expected data structure
# ============================================================

EXPECTED_SENSOR_COUNT = 1681
EXPECTED_TIME_COUNT = len(SENSOR_TIMES)

EXPECTED_MEASUREMENT_SHAPE = (
    EXPECTED_SENSOR_COUNT,
    EXPECTED_TIME_COUNT,
    2,
)

EXPECTED_INPUT_SHAPE = (
    EXPECTED_SENSOR_COUNT,
    3,
)

EXPECTED_TARGET_SHAPE = (
    EXPECTED_SENSOR_COUNT,
    2 * EXPECTED_TIME_COUNT,
)

EXPECTED_TIMES = np.asarray(
    SENSOR_TIMES,
    dtype=np.float64,
)


# ============================================================
# Expected CFD metadata
# ============================================================

EXPECTED_RE = 200.0
EXPECTED_H = 0.05
EXPECTED_DT = 0.0025
EXPECTED_TF = 20.0
EXPECTED_SNAPSHOT_FREQ = 40


# ============================================================
# File checks
# ============================================================


def check_file(
    path: Path,
) -> dict[str, float]:
    """Validate one WakeSurrogate dataset file."""
    with np.load(path) as data:
        measurements = np.asarray(data["measurements"])

        inputs = np.asarray(data["inputs"])

        targets = np.asarray(data["targets"])

        alpha_deg = float(np.asarray(data["alpha_deg"]).reshape(-1)[0])

        sensor_x = np.asarray(data["sensor_x"])

        sensor_y = np.asarray(data["sensor_y"])

        sensor_times = np.asarray(data["sensor_times"])

        # ----------------------------------------------------
        # Array shapes
        # ----------------------------------------------------

        assert measurements.shape == EXPECTED_MEASUREMENT_SHAPE, (
            path,
            measurements.shape,
            EXPECTED_MEASUREMENT_SHAPE,
        )

        assert inputs.shape == EXPECTED_INPUT_SHAPE, (
            path,
            inputs.shape,
            EXPECTED_INPUT_SHAPE,
        )

        assert targets.shape == EXPECTED_TARGET_SHAPE, (
            path,
            targets.shape,
            EXPECTED_TARGET_SHAPE,
        )

        assert sensor_x.shape == (EXPECTED_SENSOR_COUNT,)

        assert sensor_y.shape == (EXPECTED_SENSOR_COUNT,)

        assert sensor_times.shape == (EXPECTED_TIME_COUNT,)

        # ----------------------------------------------------
        # Observation times
        # ----------------------------------------------------

        assert np.allclose(
            sensor_times,
            EXPECTED_TIMES,
            rtol=0.0,
            atol=1.0e-12,
        ), (
            path,
            sensor_times,
            EXPECTED_TIMES,
        )

        # ----------------------------------------------------
        # Finite values
        # ----------------------------------------------------

        assert np.all(np.isfinite(measurements)), path

        assert np.all(np.isfinite(inputs)), path

        assert np.all(np.isfinite(targets)), path

        assert np.all(np.isfinite(sensor_x)), path

        assert np.all(np.isfinite(sensor_y)), path

        # ----------------------------------------------------
        # ML representation
        #
        # targets must be exactly:
        #
        # [ux(t1), uy(t1), ..., ux(t5), uy(t5)]
        # ----------------------------------------------------

        expected_targets = measurements.reshape(
            EXPECTED_SENSOR_COUNT,
            2 * EXPECTED_TIME_COUNT,
        )

        assert np.allclose(
            targets,
            expected_targets,
            rtol=0.0,
            atol=0.0,
        ), path

        # ----------------------------------------------------
        # Input representation
        #
        # inputs[:, 0] = alpha
        # inputs[:, 1] = x
        # inputs[:, 2] = y
        # ----------------------------------------------------

        assert np.allclose(
            inputs[:, 0],
            alpha_deg,
            rtol=0.0,
            atol=1.0e-12,
        ), path

        assert np.allclose(
            inputs[:, 1],
            sensor_x,
            rtol=0.0,
            atol=1.0e-12,
        ), path

        assert np.allclose(
            inputs[:, 2],
            sensor_y,
            rtol=0.0,
            atol=1.0e-12,
        ), path

        # ----------------------------------------------------
        # CFD metadata
        # ----------------------------------------------------

        assert np.isclose(
            float(np.asarray(data["Re"])),
            EXPECTED_RE,
        ), path

        assert np.isclose(
            float(np.asarray(data["h"])),
            EXPECTED_H,
        ), path

        assert np.isclose(
            float(np.asarray(data["dt"])),
            EXPECTED_DT,
        ), path

        assert np.isclose(
            float(np.asarray(data["tf"])),
            EXPECTED_TF,
        ), path

        assert int(np.asarray(data["snapshot_freq"])) == EXPECTED_SNAPSHOT_FREQ, path

        # ----------------------------------------------------
        # Summary statistics
        # ----------------------------------------------------

        ux = measurements[
            :,
            :,
            0,
        ]

        uy = measurements[
            :,
            :,
            1,
        ]

        return {
            "alpha_deg": alpha_deg,
            "ux_min": float(np.min(ux)),
            "ux_max": float(np.max(ux)),
            "uy_min": float(np.min(uy)),
            "uy_max": float(np.max(uy)),
            "target_rms": float(np.sqrt(np.mean(targets**2))),
        }


# ============================================================
# Split checks
# ============================================================


def check_split(
    split: str,
) -> list[dict[str, float]]:
    """Validate one complete dataset split."""
    split_dir = DATA_ROOT / split

    files = sorted(split_dir.glob("*.npz"))

    expected_alphas = np.asarray(
        sorted(EXPECTED_ALPHAS[split]),
        dtype=np.float64,
    )

    expected_count = EXPECTED_COUNTS[split]

    print(f"{split:10s}: {len(files)} files (expected {expected_count})")

    assert len(files) == expected_count, (
        split,
        len(files),
        expected_count,
    )

    summaries = []

    loaded_alphas = []

    for path in files:
        summary = check_file(path)

        summaries.append(summary)

        loaded_alphas.append(summary["alpha_deg"])

    loaded_alphas = np.asarray(
        sorted(loaded_alphas),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Verify exact split membership.
    #
    # This catches cases where the number of files is right
    # but the wrong AoA was generated.
    # --------------------------------------------------------

    assert loaded_alphas.shape == expected_alphas.shape

    assert np.allclose(
        loaded_alphas,
        expected_alphas,
        rtol=0.0,
        atol=1.0e-8,
    ), (
        split,
        loaded_alphas.tolist(),
        expected_alphas.tolist(),
    )

    return summaries


# ============================================================
# Visualization
# ============================================================


def plot_test_case() -> None:
    """Plot the sealed alpha=63 deg CFD measurement field."""
    test_alpha = float(TEST_ALPHAS[0])

    path = DATA_ROOT / "test" / "alpha_063p0.npz"

    if not path.exists():
        raise FileNotFoundError(f"Missing test dataset: {path}")

    with np.load(path) as data:
        alpha_deg = float(np.asarray(data["alpha_deg"]).reshape(-1)[0])

        sensor_x = np.asarray(data["sensor_x"])

        sensor_y = np.asarray(data["sensor_y"])

        measurements = np.asarray(data["measurements"])

    assert np.isclose(
        alpha_deg,
        test_alpha,
    )

    x_values = np.unique(sensor_x)

    y_values = np.unique(sensor_y)

    nx = len(x_values)

    ny = len(y_values)

    assert nx * ny == EXPECTED_SENSOR_COUNT

    # --------------------------------------------------------
    # Plot middle observation time:
    #
    # t = 15.1
    # --------------------------------------------------------

    time_index = 2

    time_value = EXPECTED_TIMES[time_index]

    ux = measurements[
        :,
        time_index,
        0,
    ].reshape(
        ny,
        nx,
    )

    uy = measurements[
        :,
        time_index,
        1,
    ].reshape(
        ny,
        nx,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            11,
            4,
        ),
        constrained_layout=True,
    )

    im0 = axes[0].pcolormesh(
        x_values,
        y_values,
        ux,
        shading="auto",
    )

    axes[0].set_title(
        rf"$u_x$, "
        rf"$\alpha={alpha_deg:g}^\circ$, "
        rf"$t={time_value:g}$"
    )

    axes[0].set_xlabel("x")

    axes[0].set_ylabel("y")

    fig.colorbar(
        im0,
        ax=axes[0],
    )

    im1 = axes[1].pcolormesh(
        x_values,
        y_values,
        uy,
        shading="auto",
    )

    axes[1].set_title(
        rf"$u_y$, "
        rf"$\alpha={alpha_deg:g}^\circ$, "
        rf"$t={time_value:g}$"
    )

    axes[1].set_xlabel("x")

    axes[1].set_ylabel("y")

    fig.colorbar(
        im1,
        ax=axes[1],
    )

    output = RESULTS_ROOT / "dataset_check_alpha63_t15p1.png"

    fig.savefig(
        output,
        dpi=200,
    )

    plt.close(fig)

    print(f"\nSaved sanity-check figure: {output}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Validate the complete WakeSurrogate CFD dataset."""
    print("=" * 78)

    print("WAKE SURROGATE DATASET CHECK")

    print("=" * 78)

    print(f"Dataset version : {DATASET_VERSION}")

    print(f"Expected splits : {EXPECTED_COUNTS}")

    print(f"Observation times: {EXPECTED_TIMES.tolist()}")

    print()

    total_files = 0

    summaries = []

    # --------------------------------------------------------
    # Validate all three splits.
    # --------------------------------------------------------

    for split in (
        "train",
        "validation",
        "test",
    ):
        split_summaries = check_split(split)

        for summary in split_summaries:
            summary["split"] = split

        summaries.extend(split_summaries)

        total_files += len(split_summaries)

    print(f"\nValidated {total_files} files.")

    # --------------------------------------------------------
    # Print physical ranges.
    # --------------------------------------------------------

    print("\nAoA / velocity ranges:")

    print("-" * 78)

    for summary in sorted(
        summaries,
        key=lambda row: float(row["alpha_deg"]),
    ):
        print(
            f"{summary['split']:10s} "
            f"alpha="
            f"{summary['alpha_deg']:7.3f} "
            f"| ux=["
            f"{summary['ux_min']: .4f}, "
            f"{summary['ux_max']: .4f}] "
            f"| uy=["
            f"{summary['uy_min']: .4f}, "
            f"{summary['uy_max']: .4f}] "
            f"| RMS="
            f"{summary['target_rms']:.4f}"
        )

    # --------------------------------------------------------
    # Produce one visual sanity check.
    # --------------------------------------------------------

    plot_test_case()

    print("\n" + "=" * 78)

    print("ALL DATASET CHECKS PASSED")

    print("=" * 78)


if __name__ == "__main__":
    main()
