"""Parallel noisy best-vs-worst sensor robustness study."""

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
# Global experiment configuration
# ============================================================

ALPHA_TRUE = 63.0
ALPHA_INITIAL = 40.0

SIGMA = 0.025

SEEDS = [
    2026,
    2027,
    2028,
    2029,
    2030,
]

MAX_WORKERS = 5

# SENSOR_TIMES = np.arange(
#     12.0,
#     20.0 + 0.5,
#     0.5,
#     dtype=np.float64,
# )

SENSOR_TIMES = np.array(
    [12.0, 14.0, 16.0, 18.0, 20.0],
    dtype=np.float64,
)

# Best and worst admissible downstream sensors
SENSOR_CASES = {
    "best": {
        "x": np.array([1.0], dtype=np.float64),
        "y": np.array([0.0], dtype=np.float64),
    },
    "worst": {
        "x": np.array([1.0], dtype=np.float64),
        "y": np.array([-1.0], dtype=np.float64),
    },
}

# Developed-wake CFD configuration
H = 0.05
DT = 0.0025
TF = 20.0
RE = 200.0
SNAPSHOT_FREQ = 40

OUTPUT_DIR = Path("results")


# ============================================================
# One independent worker per random seed
# ============================================================


def run_seed(seed: int) -> list[dict[str, object]]:
    """Run best and worst sensor inversions for one noise realization."""
    log_path = OUTPUT_DIR / f"noise_seed_{seed}.log"

    results: list[dict[str, object]] = []

    seed_start = time.perf_counter()

    # Each process writes its detailed optimization trace
    # to its own log file.
    with log_path.open("w", buffering=1) as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            print("=" * 76, flush=True)
            print(f"NOISE ROBUSTNESS STUDY — SEED {seed}", flush=True)
            print("=" * 76, flush=True)
            print(f"true AoA   = {ALPHA_TRUE:.2f} deg", flush=True)
            print(f"initial AoA= {ALPHA_INITIAL:.2f} deg", flush=True)
            print(f"sigma      = {SIGMA:.4f}", flush=True)
            print(f"seed       = {seed}", flush=True)
            print(flush=True)

            rng = np.random.default_rng(seed)

            # Same standardized noise realization is applied to
            # both sensor locations for this seed.
            standard_noise = rng.standard_normal((1, SENSOR_TIMES.size, 2))

            noise = SIGMA * standard_noise

            actual_noise_rms = float(np.sqrt(np.mean(noise**2)))

            noise_objective = 0.5 * float(np.sum(noise**2))

            print(
                f"actual noise RMS = {actual_noise_rms:.8e}",
                flush=True,
            )
            print(
                f"J(true) from noise only = {noise_objective:.8e}",
                flush=True,
            )

            # One Tesseract pipeline per worker.
            #
            # Keep best/worst serial inside this worker so they
            # can reuse common forward-flow evaluations.
            with ForwardObservationPipeline(max_cached_flows=32) as pipeline:
                # ------------------------------------------------
                # Hidden truth CFD once per seed worker
                # ------------------------------------------------

                print(
                    f"\nComputing truth flow F({ALPHA_TRUE:.1f} deg)...",
                    flush=True,
                )

                truth_flow = pipeline.run_forward(
                    angle_of_attack_deg=ALPHA_TRUE,
                    h=H,
                    dt=DT,
                    tf=TF,
                    Re=RE,
                    snapshot_freq=SNAPSHOT_FREQ,
                )

                print("Truth flow complete.", flush=True)
                print(
                    "Cache:",
                    pipeline.forward_cache_info(),
                    flush=True,
                )

                # ------------------------------------------------
                # Best and worst sensors
                # ------------------------------------------------

                for case_name in ["best", "worst"]:
                    case = SENSOR_CASES[case_name]

                    clean_observations = pipeline.observe(
                        truth_flow,
                        case["x"],
                        case["y"],
                        SENSOR_TIMES,
                    )

                    noisy_observations = clean_observations + noise

                    clean_rms = float(np.sqrt(np.mean(clean_observations**2)))

                    relative_noise = actual_noise_rms / clean_rms

                    print("\n" + "#" * 76, flush=True)
                    print(
                        f"# SEED {seed} — {case_name.upper()} SENSOR",
                        flush=True,
                    )
                    print(
                        f"# location = ({case['x'][0]:.2f}, {case['y'][0]:.2f})",
                        flush=True,
                    )
                    print(
                        f"# clean signal RMS = {clean_rms:.8e}",
                        flush=True,
                    )
                    print(
                        f"# noise RMS / signal RMS = {relative_noise:.4%}",
                        flush=True,
                    )
                    print("#" * 76, flush=True)

                    case_start = time.perf_counter()

                    result = infer_angle_of_attack(
                        pipeline,
                        noisy_observations,
                        case["x"],
                        case["y"],
                        SENSOR_TIMES,
                        initial_angle_deg=ALPHA_INITIAL,
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

                    case_elapsed = time.perf_counter() - case_start

                    signed_error = float(result.angle_of_attack_deg - ALPHA_TRUE)

                    absolute_error = abs(signed_error)

                    row = {
                        "seed": seed,
                        "case": case_name,
                        "x": float(case["x"][0]),
                        "y": float(case["y"][0]),
                        "sigma": SIGMA,
                        "noise_rms": actual_noise_rms,
                        "relative_noise": relative_noise,
                        "recovered_angle_deg": float(result.angle_of_attack_deg),
                        "signed_error_deg": signed_error,
                        "absolute_error_deg": absolute_error,
                        "iterations": result.iterations,
                        "final_objective": float(result.objective),
                        "converged": bool(result.converged),
                        "wall_time_min": (case_elapsed / 60.0),
                    }

                    results.append(row)

                    print("\nCASE RESULT", flush=True)
                    print("-" * 60, flush=True)
                    print(
                        f"seed           = {seed}",
                        flush=True,
                    )
                    print(
                        f"case           = {case_name}",
                        flush=True,
                    )
                    print(
                        f"recovered AoA  = {result.angle_of_attack_deg:.8f} deg",
                        flush=True,
                    )
                    print(
                        f"signed error   = {signed_error:+.8e} deg",
                        flush=True,
                    )
                    print(
                        f"absolute error = {absolute_error:.8e} deg",
                        flush=True,
                    )
                    print(
                        f"iterations     = {result.iterations}",
                        flush=True,
                    )
                    print(
                        f"converged      = {result.converged}",
                        flush=True,
                    )
                    print(
                        f"wall time      = {case_elapsed / 60.0:.2f} min",
                        flush=True,
                    )
                    print(
                        "cache          =",
                        pipeline.forward_cache_info(),
                        flush=True,
                    )

            seed_elapsed = time.perf_counter() - seed_start

            print("\n" + "=" * 76, flush=True)
            print(
                f"SEED {seed} COMPLETE",
                flush=True,
            )
            print(
                f"seed wall time = {seed_elapsed / 60.0:.2f} min",
                flush=True,
            )
            print("=" * 76, flush=True)

    return results


# ============================================================
# Main process
# ============================================================


def main() -> None:
    """Launch all five noise realizations in parallel."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    total_start = time.perf_counter()

    print("=" * 76, flush=True)
    print("PARALLEL 5-SEED NOISY SENSOR ROBUSTNESS STUDY", flush=True)
    print("=" * 76, flush=True)
    print(f"workers = {MAX_WORKERS}", flush=True)
    print(f"seeds   = {SEEDS}", flush=True)
    print(f"sigma   = {SIGMA}", flush=True)
    print(flush=True)

    print("Detailed logs:", flush=True)

    for seed in SEEDS:
        print(
            f"  results/noise_seed_{seed}.log",
            flush=True,
        )

    print(
        "\nLaunching workers...\n",
        flush=True,
    )

    all_results: list[dict[str, object]] = []

    # Use spawn rather than fork because each worker launches
    # its own Tesseract/Docker runtime.
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=context,
    ) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in SEEDS}

        for future in as_completed(futures):
            seed = futures[future]

            try:
                seed_results = future.result()
            except Exception as exc:
                print(
                    f"[FAILED] seed {seed}: {exc!r}",
                    flush=True,
                )
                raise

            all_results.extend(seed_results)

            elapsed = time.perf_counter() - total_start

            print(
                f"[DONE] seed {seed} "
                f"({len(seed_results)} sensor cases) "
                f"| total elapsed "
                f"{elapsed / 60.0:.2f} min",
                flush=True,
            )

    # ========================================================
    # Sort results
    # ========================================================

    case_order = {
        "best": 0,
        "worst": 1,
    }

    all_results.sort(
        key=lambda row: (
            int(row["seed"]),
            case_order[str(row["case"])],
        )
    )

    # ========================================================
    # Save combined CSV
    # ========================================================

    csv_path = OUTPUT_DIR / "noisy_best_worst_sigma_0p025_5seeds_parallel.csv"

    fieldnames = list(all_results[0].keys())

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(all_results)

    # ========================================================
    # Per-seed summary
    # ========================================================

    print("\n\n" + "=" * 116)
    print("PER-SEED RESULTS")
    print("=" * 116)

    print(
        f"{'Seed':>6} "
        f"{'Case':>8} "
        f"{'Recovered [deg]':>18} "
        f"{'Signed error':>16} "
        f"{'Absolute error':>16} "
        f"{'Iterations':>12} "
        f"{'Converged':>10}"
    )

    for row in all_results:
        print(
            f"{int(row['seed']):6d} "
            f"{row['case']!s:>8} "
            f"{float(row['recovered_angle_deg']):18.8f} "
            f"{float(row['signed_error_deg']):16.8e} "
            f"{float(row['absolute_error_deg']):16.8e} "
            f"{int(row['iterations']):12d} "
            f"{row['converged']!s:>10}"
        )

    # ========================================================
    # Statistical summary
    # ========================================================

    summary = {}

    for case_name in ["best", "worst"]:
        case_rows = [row for row in all_results if row["case"] == case_name]

        estimates = np.array(
            [float(row["recovered_angle_deg"]) for row in case_rows],
            dtype=np.float64,
        )

        errors = estimates - ALPHA_TRUE
        absolute_errors = np.abs(errors)

        summary[case_name] = {
            "mean": float(np.mean(estimates)),
            "bias": float(np.mean(errors)),
            "mae": float(np.mean(absolute_errors)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "std": float(
                np.std(
                    estimates,
                    ddof=1,
                )
            ),
            "converged": sum(bool(row["converged"]) for row in case_rows),
        }

    print("\n\n" + "=" * 110)
    print("5-SEED NOISY SENSOR ROBUSTNESS SUMMARY")
    print("=" * 110)

    print(
        f"{'Case':>8} "
        f"{'Mean AoA':>14} "
        f"{'Bias':>14} "
        f"{'MAE':>14} "
        f"{'RMSE':>14} "
        f"{'Std(AoA)':>14} "
        f"{'Conv.':>10}"
    )

    for case_name in ["best", "worst"]:
        row = summary[case_name]

        print(
            f"{case_name:>8} "
            f"{row['mean']:14.8f} "
            f"{row['bias']:14.6e} "
            f"{row['mae']:14.6e} "
            f"{row['rmse']:14.6e} "
            f"{row['std']:14.6e} "
            f"{row['converged']:>4}/5"
        )

    best_rmse = summary["best"]["rmse"]
    worst_rmse = summary["worst"]["rmse"]

    best_std = summary["best"]["std"]
    worst_std = summary["worst"]["std"]

    if best_rmse > 0.0:
        rmse_ratio = worst_rmse / best_rmse
    else:
        rmse_ratio = np.inf

    if best_std > 0.0:
        std_ratio = worst_std / best_std
    else:
        std_ratio = np.inf

    print("-" * 110)

    print(f"worst / best RMSE ratio = {rmse_ratio:.3f}")

    print(f"worst / best std ratio  = {std_ratio:.3f}")

    total_elapsed = time.perf_counter() - total_start

    print(
        f"\ncombined CSV = {csv_path}",
        flush=True,
    )

    print(
        f"total wall time = {total_elapsed / 60.0:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
