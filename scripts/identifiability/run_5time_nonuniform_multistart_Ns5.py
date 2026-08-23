"""Noise-free multi-start AoA inference with sparse nonuniform observations."""

import csv
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.inverse import infer_angle_of_attack
from immersa_tesseract_inference.pipeline import ForwardObservationPipeline

# ============================================================
# Experiment configuration
# ============================================================

ALPHA_TRUE = 63.0

INITIAL_ANGLES = [
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
]

MAX_WORKERS = 10

# Sparse, nonuniform temporal observations
SENSOR_TIMES = np.array(
    [12.0, 13.3, 15.1, 17.4, 20.0],
    dtype=np.float64,
)

# Center sensor: locally high-information location
SENSOR_X = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float64,
)

SENSOR_Y = np.array(
    [-0.8, -0.4, 0.0, 0.4, 0.8],
    dtype=np.float64,
)

# Developed-wake CFD settings
H = 0.05
DT = 0.0025
TF = 20.0
RE = 200.0
SNAPSHOT_FREQ = 40

OUTPUT_DIR = Path("results/identifiability/multistart")
OUTPUT_CSV = OUTPUT_DIR / "Ns5.csv"


# ============================================================
# Basin classification helper
# ============================================================

KNOWN_MINIMA = {
    "low_basin_27": 27.0,
    "mid_basin_42": 42.0,
    "mid_basin_51": 51.0,
    "true_basin": 63.0,
    "high_basin_83": 83.0,
}


def classify_basin(alpha: float) -> str:
    """Assign recovered AoA to nearest observed landscape minimum."""
    return min(
        KNOWN_MINIMA,
        key=lambda name: abs(alpha - KNOWN_MINIMA[name]),
    )


# ============================================================
# One independent worker per initial condition
# ============================================================


def run_initial_condition(
    initial_angle: float,
    observations: np.ndarray,
) -> dict[str, object]:
    """Run one noise-free Gauss-Newton inversion."""
    tag = str(initial_angle).replace(".", "p")

    log_path = OUTPUT_DIR / f"Ns5_initial_{tag}_deg.log"

    start = time.perf_counter()

    with log_path.open(
        "w",
        buffering=1,
    ) as log_file:
        with (
            redirect_stdout(log_file),
            redirect_stderr(log_file),
        ):
            print("=" * 76, flush=True)
            print(
                "NOISE-FREE Ns=5 MULTI-START AoA INFERENCE",
                flush=True,
            )
            print("=" * 76, flush=True)

            print(
                f"true AoA       = {ALPHA_TRUE:.2f} deg",
                flush=True,
            )

            print(
                f"initial AoA    = {initial_angle:.2f} deg",
                flush=True,
            )

            print(
                f"sensor x       = {SENSOR_X.tolist()}, sensor y = {SENSOR_Y.tolist()}",
                flush=True,
            )

            print(
                f"sensor times   = {SENSOR_TIMES.tolist()}",
                flush=True,
            )

            print(
                "noise           = NONE",
                flush=True,
            )

            print(flush=True)

            # Independent Tesseract pipeline for this worker.
            with ForwardObservationPipeline(max_cached_flows=32) as pipeline:
                result = infer_angle_of_attack(
                    pipeline,
                    observations,
                    SENSOR_X,
                    SENSOR_Y,
                    SENSOR_TIMES,
                    initial_angle_deg=initial_angle,
                    epsilon_deg=0.5,
                    max_step_deg=10.0,
                    max_iterations=20,
                    h=H,
                    dt=DT,
                    tf=TF,
                    Re=RE,
                    snapshot_freq=SNAPSHOT_FREQ,
                    verbose=True,
                )

                cache_info = pipeline.forward_cache_info()

            elapsed = time.perf_counter() - start

            recovered = float(result.angle_of_attack_deg)

            signed_error = recovered - ALPHA_TRUE

            basin = classify_basin(recovered)

            print("\n" + "=" * 76, flush=True)
            print(
                "MULTI-START RESULT",
                flush=True,
            )
            print("=" * 76, flush=True)

            print(
                f"initial AoA    = {initial_angle:.8f} deg",
                flush=True,
            )

            print(
                f"recovered AoA  = {recovered:.8f} deg",
                flush=True,
            )

            print(
                f"signed error   = {signed_error:+.8e} deg",
                flush=True,
            )

            print(
                f"basin          = {basin}",
                flush=True,
            )

            print(
                f"iterations     = {result.iterations}",
                flush=True,
            )

            print(
                f"final J        = {result.objective:.8e}",
                flush=True,
            )

            print(
                f"converged      = {result.converged}",
                flush=True,
            )

            print(
                f"wall time      = {elapsed / 60.0:.2f} min",
                flush=True,
            )

            print(
                f"cache          = {cache_info}",
                flush=True,
            )

    return {
        "initial_angle_deg": initial_angle,
        "recovered_angle_deg": recovered,
        "signed_error_deg": signed_error,
        "absolute_error_deg": abs(signed_error),
        "basin": basin,
        "iterations": result.iterations,
        "final_objective": float(result.objective),
        "converged": bool(result.converged),
        "wall_time_min": (elapsed / 60.0),
    }


# ============================================================
# Main process
# ============================================================


def main() -> None:
    """Compute truth data and launch all initial guesses."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_start = time.perf_counter()

    print("=" * 84, flush=True)
    print(
        "PARALLEL NOISE-FREE Ns=5 MULTI-START AoA STUDY",
        flush=True,
    )
    print("=" * 84, flush=True)

    print(f"truth AoA       = {ALPHA_TRUE:.1f} deg")

    print(f"initial guesses = {INITIAL_ANGLES}")

    print(f"workers         = {MAX_WORKERS}")

    print(
        f"sensor x        = {SENSOR_X.tolist()}\nsensor y        = {SENSOR_Y.tolist()}"
    )

    print(f"sensor times    = {SENSOR_TIMES.tolist()}")

    print(
        "\nComputing exact noise-free truth observations...",
        flush=True,
    )

    # ========================================================
    # Truth observations computed once
    # ========================================================

    with ForwardObservationPipeline(max_cached_flows=1) as pipeline:
        truth_flow = pipeline.run_forward(
            angle_of_attack_deg=ALPHA_TRUE,
            h=H,
            dt=DT,
            tf=TF,
            Re=RE,
            snapshot_freq=SNAPSHOT_FREQ,
        )

        observations = pipeline.observe(
            truth_flow,
            SENSOR_X,
            SENSOR_Y,
            SENSOR_TIMES,
        )

    print(
        "Truth observations complete.",
        flush=True,
    )

    print(
        "\nLaunching 10 independent initial conditions...\n",
        flush=True,
    )

    # ========================================================
    # Parallel multi-start
    # ========================================================

    rows = []

    context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                run_initial_condition,
                initial_angle,
                observations,
            ): initial_angle
            for initial_angle in INITIAL_ANGLES
        }

        for future in as_completed(futures):
            initial_angle = futures[future]

            try:
                row = future.result()

            except Exception as exc:
                print(
                    f"[FAILED] initial={initial_angle:.1f}: {exc!r}",
                    flush=True,
                )

                raise

            rows.append(row)

            elapsed = time.perf_counter() - total_start

            print(
                f"[DONE] "
                f"alpha0={initial_angle:5.1f} "
                f"-> "
                f"{row['recovered_angle_deg']:9.4f} deg "
                f"| {row['basin']:<17} "
                f"| elapsed "
                f"{elapsed / 60.0:.2f} min",
                flush=True,
            )

    # ========================================================
    # Sort by initial condition
    # ========================================================

    rows.sort(key=lambda row: float(row["initial_angle_deg"]))

    # ========================================================
    # Save CSV
    # ========================================================

    csv_path = OUTPUT_CSV

    fieldnames = list(rows[0].keys())

    with csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # Final table
    # ========================================================

    print("\n\n" + "=" * 116)
    print("MULTI-START BASIN-OF-ATTRACTION SUMMARY")
    print("=" * 116)

    print(
        f"{'Initial':>10} "
        f"{'Recovered':>14} "
        f"{'Error':>14} "
        f"{'Basin':>20} "
        f"{'Iterations':>12} "
        f"{'Final J':>16} "
        f"{'Conv.':>8}"
    )

    for row in rows:
        print(
            f"{float(row['initial_angle_deg']):10.2f} "
            f"{float(row['recovered_angle_deg']):14.8f} "
            f"{float(row['signed_error_deg']):14.6f} "
            f"{row['basin']!s:>20} "
            f"{int(row['iterations']):12d} "
            f"{float(row['final_objective']):16.8e} "
            f"{row['converged']!s:>8}"
        )

    # ========================================================
    # Basin counts
    # ========================================================

    print("\nBasin counts:")

    for basin_name in KNOWN_MINIMA:
        count = sum(row["basin"] == basin_name for row in rows)

        print(f"  {basin_name:<20}: {count}/{len(rows)}")

    total_elapsed = time.perf_counter() - total_start

    print(
        f"\nCSV saved to: {csv_path}",
        flush=True,
    )

    print(
        f"total wall time = {total_elapsed / 60.0:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
