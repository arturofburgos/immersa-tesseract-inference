"""Random-layout baseline and matched-budget comparison for the T4 design.

Ten thousand uniformly sampled feasible two-sensor layouts are scored with the
same frozen T4 criterion used by the gradient campaign, so the optimized layout
can be placed in that distribution.

WakeSurrogate evaluates each sensor independently, so all K layouts are scored
with one call per design angle by packing 2K sensor coordinates into a single
request -- the measured cost of a 20000-sensor call is a few milliseconds more
than a 2-sensor one.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.sensor_design import (
    ALPHA_GRID_DEG,
    BASELINE_LAYOUT,
    DELTA_ALPHA_MIN_DEG,
    DESIGN_BOUNDS,
    MIN_SENSOR_DISTANCE,
    SensorDesignPipeline,
)

K_LAYOUTS = 10_000
RANDOM_SEED = 0
BOOTSTRAP_REPLICATES = 2000

OPTIMIZATION_DIR = Path("results/sensor_design/optimization")
OUTPUT_DIR = Path("results/sensor_design/random_baseline")

SCORES_NPZ = OUTPUT_DIR / "random_layout_scores.npz"
SUMMARY_JSON = OUTPUT_DIR / "random_baseline_summary.json"
MATCHED_CSV = OUTPUT_DIR / "matched_budget_comparison.csv"


def sample_layouts(count: int, seed: int) -> np.ndarray:
    """Uniform feasible layouts as an (count, 4) array [x1, y1, x2, y2]."""
    rng = np.random.default_rng(seed)

    accepted = []

    while len(accepted) < count:
        draw = max(count - len(accepted), 1) * 2

        x = rng.uniform(DESIGN_BOUNDS[0][0], DESIGN_BOUNDS[0][1], (draw, 2))
        y = rng.uniform(DESIGN_BOUNDS[1][0], DESIGN_BOUNDS[1][1], (draw, 2))

        feasible = np.hypot(x[:, 0] - x[:, 1], y[:, 0] - y[:, 1]) >= MIN_SENSOR_DISTANCE

        block = np.stack(
            [x[feasible, 0], y[feasible, 0], x[feasible, 1], y[feasible, 1]],
            axis=1,
        )

        accepted.append(block)

        if sum(len(b) for b in accepted) >= count:
            break

    return np.concatenate(accepted, axis=0)[:count]


def main() -> None:
    """Score the random population and run the matched-budget comparison."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection = json.loads((OPTIMIZATION_DIR / "s_star_surrogate.json").read_text())
    calibration = json.loads((OPTIMIZATION_DIR / "calibration.json").read_text())

    tau = calibration["tau"]

    d_star = float(selection["D_tau"])

    layouts = sample_layouts(K_LAYOUTS, RANDOM_SEED)

    print("=" * 78, flush=True)
    print(
        f"Random-layout baseline: {len(layouts)} feasible layouts, seed {RANDOM_SEED}"
    )
    print("=" * 78, flush=True)

    started = time.perf_counter()

    with SensorDesignPipeline() as pipeline:
        baseline_measurements = pipeline.measurements(BASELINE_LAYOUT)
        d_baseline = pipeline.discrimination(baseline_measurements, tau)[
            "discrimination"
        ]

        # ----------------------------------------------------
        # One packed WakeSurrogate call per design angle.
        # ----------------------------------------------------

        sensor_x = layouts[:, [0, 2]].reshape(-1).astype(np.float32)
        sensor_y = layouts[:, [1, 3]].reshape(-1).astype(np.float32)

        measure_start = time.perf_counter()

        batch = []

        for alpha in ALPHA_GRID_DEG:
            outputs = pipeline._surrogate.apply(
                {
                    "angle_of_attack_deg": float(alpha),
                    "sensor_x": sensor_x,
                    "sensor_y": sensor_y,
                }
            )
            batch.append(np.asarray(outputs["measurements"], dtype=np.float32))

        measure_time = time.perf_counter() - measure_start

        # (N_alpha, 2K, 5, 2) -> (K, N_alpha, 2, 5, 2)
        stacked = np.stack(batch, axis=0)
        stacked = stacked.reshape(len(ALPHA_GRID_DEG), len(layouts), 2, 5, 2)
        stacked = np.transpose(stacked, (1, 0, 2, 3, 4))

        print(
            f"measurements: {len(ALPHA_GRID_DEG)} packed T3 calls "
            f"({sensor_x.size} sensors each) in {measure_time:.2f} s",
            flush=True,
        )

        # ----------------------------------------------------
        # Score each layout with the frozen T4 criterion.
        # ----------------------------------------------------

        score_start = time.perf_counter()

        scores = np.empty(len(layouts), dtype=np.float64)

        for index in range(len(layouts)):
            scores[index] = pipeline.discrimination(stacked[index], tau)[
                "discrimination"
            ]

            if (index + 1) % 2000 == 0:
                print(f"  scored {index + 1}/{len(layouts)}", flush=True)

        score_time = time.perf_counter() - score_start

    total_time = time.perf_counter() - started

    seconds_per_candidate = score_time / len(layouts)

    np.savez_compressed(SCORES_NPZ, layouts=layouts, scores=scores)

    # --------------------------------------------------------
    # Distribution statistics
    # --------------------------------------------------------

    better_than_star = int(np.sum(scores > d_star))
    percentile = 100.0 * float(np.mean(scores < d_star))

    summary = {
        "n_layouts": len(layouts),
        "seed": RANDOM_SEED,
        "tau": tau,
        "delta_alpha_min_deg": DELTA_ALPHA_MIN_DEG,
        "D_baseline": d_baseline,
        "D_star_surrogate": d_star,
        "random_best": float(scores.max()),
        "random_median": float(np.median(scores)),
        "random_p95": float(np.percentile(scores, 95)),
        "random_p99": float(np.percentile(scores, 99)),
        "random_p999": float(np.percentile(scores, 99.9)),
        "s_star_percentile": percentile,
        "s_star_rank": better_than_star + 1,
        "n_random_beating_s_star": better_than_star,
        "fraction_beating_baseline": float(np.mean(scores > d_baseline)),
        "fraction_within_half_percent_of_star": float(
            np.mean(scores >= d_star * (1.0 - 0.005))
        ),
        "measurement_time_s": measure_time,
        "scoring_time_s": score_time,
        "seconds_per_candidate": seconds_per_candidate,
        "total_time_s": total_time,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    for key, value in summary.items():
        print(f"  {key:38s} {value}")

    # --------------------------------------------------------
    # Matched-budget comparison
    # --------------------------------------------------------
    #
    # Subsampling the scored population without replacement is equivalent to
    # running a fresh random search of that size, and costs no further calls.

    rows = list(csv.DictReader(open(OPTIMIZATION_DIR / "multistart_summary.csv")))

    evaluation_counts = np.array(
        [int(r["objective_gradient_evaluations"]) for r in rows]
    )
    wall_times = np.array([float(r["wall_time_s"]) for r in rows])
    final_scores = np.array([float(r["D_tau_final"]) for r in rows])

    best_row = rows[int(np.argmax(final_scores))]

    budgets = {
        "A_equal_candidates_best_run": int(best_row["objective_gradient_evaluations"]),
        "A_equal_candidates_median_run": int(np.median(evaluation_counts)),
        "B_equal_wallclock_best_run": int(
            float(best_row["wall_time_s"]) / seconds_per_candidate
        ),
        "B_equal_wallclock_median_run": int(
            float(np.median(wall_times)) / seconds_per_candidate
        ),
    }

    rng = np.random.default_rng(RANDOM_SEED)

    matched_rows = []

    for label, budget in budgets.items():
        budget = max(1, min(budget, len(scores)))

        best_of = np.array(
            [
                scores[rng.choice(len(scores), size=budget, replace=False)].max()
                for _ in range(BOOTSTRAP_REPLICATES)
            ]
        )

        matched_rows.append(
            {
                "comparison": label,
                "budget_candidates": budget,
                "random_best_mean": float(best_of.mean()),
                "random_best_median": float(np.median(best_of)),
                "random_best_p05": float(np.percentile(best_of, 5)),
                "random_best_p95": float(np.percentile(best_of, 95)),
                "random_best_max": float(best_of.max()),
                "gradient_D_tau": d_star,
                "fraction_of_replicates_beating_gradient": float(
                    np.mean(best_of > d_star)
                ),
                "replicates": BOOTSTRAP_REPLICATES,
            }
        )

    with MATCHED_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matched_rows[0]))
        writer.writeheader()
        writer.writerows(matched_rows)

    print()
    print("Matched-budget random search (bootstrap over the scored population):")
    for row in matched_rows:
        print(
            f"  {row['comparison']:32s} budget {row['budget_candidates']:5d}  "
            f"median best {row['random_best_median']:.6f}  "
            f"p95 {row['random_best_p95']:.6f}  "
            f"beats gradient in {row['fraction_of_replicates_beating_gradient']:.1%}"
        )

    print()
    print(f"Wrote {SCORES_NPZ}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {MATCHED_CSV}")


if __name__ == "__main__":
    main()
