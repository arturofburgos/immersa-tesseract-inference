"""Generate CFD data for the WakeSurrogate Tesseract."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path

import numpy as np

from immersa_tesseract_inference.pipeline import (
    ForwardObservationPipeline,
)
from immersa_tesseract_inference.wake_surrogate_config import (
    SENSOR_TIMES as CONFIG_SENSOR_TIMES,
    TEST_ALPHAS,
    TRAIN_ALPHAS_BY_VERSION,
    VALIDATION_ALPHAS,
)


# ============================================================
# Command-line interface
# ============================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate CFD data for one WakeSurrogate "
            "training-data version."
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
            "WakeSurrogate dataset version to generate."
        ),
    )

    return parser.parse_args()


# ============================================================
# Sensor-query domain
# ============================================================

X_VALUES = np.arange(
    1.0,
    3.0 + 0.025,
    0.05,
    dtype=np.float64,
)

Y_VALUES = np.arange(
    -1.0,
    1.0 + 0.025,
    0.05,
    dtype=np.float64,
)

X_GRID, Y_GRID = np.meshgrid(
    X_VALUES,
    Y_VALUES,
    indexing="xy",
)

SENSOR_X = X_GRID.ravel()

SENSOR_Y = Y_GRID.ravel()

SENSOR_TIMES = np.asarray(
    CONFIG_SENSOR_TIMES,
    dtype=np.float64,
)


# ============================================================
# CFD configuration
# ============================================================

H = 0.05

DT = 0.0025

TF = 20.0

RE = 200.0

SNAPSHOT_FREQ = 40

MAX_WORKERS = 15

OUTPUT_ROOT = Path(
    "data/wake_surrogate"
)


# ============================================================
# Utility
# ============================================================


def alpha_tag(
    alpha_deg: float,
) -> str:
    """
    Convert an AoA into a filesystem-safe tag.

    The filename keeps the existing one-decimal convention.
    The exact AoA is stored inside the NPZ metadata.
    """

    return (
        f"{alpha_deg:05.1f}"
        .replace(
            ".",
            "p",
        )
    )


# ============================================================
# Generate one CFD case
# ============================================================


def generate_case(
    split: str,
    alpha_deg: float,
) -> dict[str, object]:
    """Generate one WakeSurrogate CFD dataset trajectory."""

    output_dir = (
        OUTPUT_ROOT
        / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"alpha_"
            f"{alpha_tag(alpha_deg)}"
            f".npz"
        )
    )

    # --------------------------------------------------------
    # Allow interrupted dataset generation to resume without
    # recomputing successful trajectories.
    # --------------------------------------------------------

    if output_path.exists():

        return {
            "split": split,
            "alpha_deg": alpha_deg,
            "status": "skipped",
            "path": str(
                output_path
            ),
            "wall_time_min": 0.0,
        }

    start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Each process owns its own Tesseract pipeline/container.
    # --------------------------------------------------------

    with ForwardObservationPipeline(
        max_cached_flows=1
    ) as pipeline:

        flow = pipeline.run_forward(
            angle_of_attack_deg=alpha_deg,
            h=H,
            dt=DT,
            tf=TF,
            Re=RE,
            snapshot_freq=SNAPSHOT_FREQ,
        )

        measurements = pipeline.observe(
            flow,
            SENSOR_X,
            SENSOR_Y,
            SENSOR_TIMES,
        )

    measurements = np.asarray(
        measurements,
        dtype=np.float64,
    )

    expected_shape = (
        SENSOR_X.size,
        SENSOR_TIMES.size,
        2,
    )

    if (
        measurements.shape
        != expected_shape
    ):
        raise RuntimeError(
            "Unexpected measurement shape: "
            f"{measurements.shape}; "
            f"expected {expected_shape}."
        )

    if not np.all(
        np.isfinite(
            measurements
        )
    ):
        raise RuntimeError(
            "Non-finite measurements for "
            f"alpha={alpha_deg}."
        )

    # ========================================================
    # ML representation
    #
    # inputs:
    #
    #   [alpha, x, y]
    #
    # targets:
    #
    #   [ux(t1), uy(t1), ..., ux(t5), uy(t5)]
    # ========================================================

    inputs = np.column_stack(
        (
            np.full(
                SENSOR_X.size,
                alpha_deg,
                dtype=np.float64,
            ),
            SENSOR_X,
            SENSOR_Y,
        )
    )

    targets = measurements.reshape(
        SENSOR_X.size,
        -1,
    )

    # --------------------------------------------------------
    # Save exact physical and ML representations together.
    # --------------------------------------------------------

    np.savez_compressed(
        output_path,

        alpha_deg=np.array(
            alpha_deg,
            dtype=np.float64,
        ),

        sensor_x=SENSOR_X,

        sensor_y=SENSOR_Y,

        sensor_times=SENSOR_TIMES,

        measurements=measurements,

        inputs=inputs,

        targets=targets,

        Re=np.array(
            RE,
            dtype=np.float64,
        ),

        h=np.array(
            H,
            dtype=np.float64,
        ),

        dt=np.array(
            DT,
            dtype=np.float64,
        ),

        tf=np.array(
            TF,
            dtype=np.float64,
        ),

        snapshot_freq=np.array(
            SNAPSHOT_FREQ,
            dtype=np.int64,
        ),
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "split": split,
        "alpha_deg": alpha_deg,
        "status": "generated",
        "path": str(
            output_path
        ),
        "wall_time_min": (
            elapsed
            / 60.0
        ),
    }


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Generate one WakeSurrogate CFD dataset version."""

    args = parse_arguments()

    dataset_version = (
        args.version
    )

    # --------------------------------------------------------
    # Dataset membership comes entirely from the shared config.
    # --------------------------------------------------------

    train_alphas = list(
        TRAIN_ALPHAS_BY_VERSION[
            dataset_version
        ]
    )

    validation_alphas = list(
        VALIDATION_ALPHAS
    )

    test_alphas = list(
        TEST_ALPHAS
    )

    splits = {
        "train": train_alphas,
        "validation": validation_alphas,
        "test": test_alphas,
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    jobs = [
        (
            split,
            alpha,
        )
        for (
            split,
            alphas,
        )
        in splits.items()
        for alpha
        in alphas
    ]

    # ========================================================
    # Dataset sizes
    # ========================================================

    n_train_samples = (
        len(
            train_alphas
        )
        * SENSOR_X.size
    )

    n_validation_samples = (
        len(
            validation_alphas
        )
        * SENSOR_X.size
    )

    n_test_samples = (
        len(
            test_alphas
        )
        * SENSOR_X.size
    )

    n_total_samples = (
        n_train_samples
        + n_validation_samples
        + n_test_samples
    )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "=" * 78
    )

    print(
        "WAKE SURROGATE DATASET GENERATION"
    )

    print(
        "=" * 78
    )

    print(
        f"Dataset version  : "
        f"{dataset_version}"
    )

    print(
        f"Training AoAs    : "
        f"{train_alphas}"
    )

    print(
        f"Validation AoAs  : "
        f"{validation_alphas}"
    )

    print(
        f"Test AoAs        : "
        f"{test_alphas}"
    )

    print(
        f"\nTraining AoA count   : "
        f"{len(train_alphas)}"
    )

    print(
        f"Validation AoA count : "
        f"{len(validation_alphas)}"
    )

    print(
        f"Test AoA count       : "
        f"{len(test_alphas)}"
    )

    print(
        f"\nSpatial grid     : "
        f"{len(X_VALUES)} "
        f"x "
        f"{len(Y_VALUES)}"
    )

    print(
        f"Sensor locations : "
        f"{SENSOR_X.size}"
    )

    print(
        f"Times            : "
        f"{SENSOR_TIMES.tolist()}"
    )

    print(
        f"\nTraining samples   : "
        f"{n_train_samples:,}"
    )

    print(
        f"Validation samples : "
        f"{n_validation_samples:,}"
    )

    print(
        f"Test samples       : "
        f"{n_test_samples:,}"
    )

    print(
        f"Total ML samples   : "
        f"{n_total_samples:,}"
    )

    print(
        f"\nTotal requested CFD runs : "
        f"{len(jobs)}"
    )

    print(
        f"Workers                  : "
        f"{MAX_WORKERS}"
    )

    print(
        f"CFD resolution           : "
        f"h={H}, "
        f"dt={DT}, "
        f"tf={TF}, "
        f"Re={RE}"
    )

    print(
        f"Snapshot frequency       : "
        f"{SNAPSHOT_FREQ}"
    )

    print()

    print(
        "Existing dataset files are skipped."
    )

    # ========================================================
    # Generate trajectories
    # ========================================================

    total_start = (
        time.perf_counter()
    )

    context = mp.get_context(
        "spawn"
    )

    generated_count = 0

    skipped_count = 0

    failed_count = 0

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=context,
    ) as executor:

        futures = {
            executor.submit(
                generate_case,
                split,
                alpha,
            ): (
                split,
                alpha,
            )
            for (
                split,
                alpha,
            )
            in jobs
        }

        for future in as_completed(
            futures
        ):

            (
                split,
                alpha,
            ) = futures[
                future
            ]

            try:

                result = (
                    future.result()
                )

            except Exception as exc:

                failed_count += 1

                print(
                    f"[FAILED   ] "
                    f"{split:10s} "
                    f"alpha={alpha:7.3f}: "
                    f"{exc!r}",
                    flush=True,
                )

                raise

            if (
                result[
                    "status"
                ]
                == "generated"
            ):

                generated_count += 1

            elif (
                result[
                    "status"
                ]
                == "skipped"
            ):

                skipped_count += 1

            elapsed = (
                time.perf_counter()
                - total_start
            )

            print(
                f"[{result['status'].upper():9s}] "
                f"{split:10s} "
                f"alpha={alpha:7.3f} "
                f"| case "
                f"{result['wall_time_min']:.2f} min "
                f"| total "
                f"{elapsed / 60.0:.2f} min",
                flush=True,
            )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    # ========================================================
    # Final summary
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "DATASET COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Dataset version : "
        f"{dataset_version}"
    )

    print(
        f"Requested runs  : "
        f"{len(jobs)}"
    )

    print(
        f"Generated       : "
        f"{generated_count}"
    )

    print(
        f"Skipped         : "
        f"{skipped_count}"
    )

    print(
        f"Failed          : "
        f"{failed_count}"
    )

    print(
        f"Total wall time : "
        f"{total_elapsed / 60.0:.2f} min"
    )

    print(
        f"Output          : "
        f"{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()