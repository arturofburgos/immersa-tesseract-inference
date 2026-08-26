"""Random-layout reference for the refined 66-angle surrogate objective.

The Phase-I random distribution cannot be reused: changing the AoA grid changes
the objective, so the population must be rescored. Sampling and seed are
identical to Phase I so the comparison is like-for-like.
"""

import csv
import json
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.sensor_design import (
    BASELINE_LAYOUT,
    DESIGN_BOUNDS,
    MIN_SENSOR_DISTANCE,
    SensorDesignPipeline,
)

K_LAYOUTS = 10_000
RANDOM_SEED = 0
BOOTSTRAP_REPLICATES = 2000

REFINED_GRID_DEG = np.arange(20.0, 85.0 + 1.0e-9, 1.0, dtype=np.float64)
DELTA_ALPHA_MIN_DEG = 7.5

SURROGATE_DIR = Path("results/sensor_design/refined_design/surrogate_v2")
OUTPUT_DIR = Path("results/sensor_design/refined_design/surrogate_v2")


def sample_layouts(count: int, seed: int) -> np.ndarray:
    """Uniform feasible layouts as an (count, 4) array."""
    rng = np.random.default_rng(seed)
    accepted = []
    while sum(len(b) for b in accepted) < count:
        draw = max(count, 1) * 2
        x = rng.uniform(DESIGN_BOUNDS[0][0], DESIGN_BOUNDS[0][1], (draw, 2))
        y = rng.uniform(DESIGN_BOUNDS[1][0], DESIGN_BOUNDS[1][1], (draw, 2))
        ok = np.hypot(x[:, 0] - x[:, 1], y[:, 0] - y[:, 1]) >= MIN_SENSOR_DISTANCE
        accepted.append(np.stack([x[ok, 0], y[ok, 0], x[ok, 1], y[ok, 1]], axis=1))
    return np.concatenate(accepted, axis=0)[:count]


def main() -> None:
    """Score the random population against the refined objective."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection = json.loads((SURROGATE_DIR / "s_star_surrogate_v2.json").read_text())
    tau = selection["tau"]
    d_star = selection["D_tau"]

    layouts = sample_layouts(K_LAYOUTS, RANDOM_SEED)

    print(
        f"Refined random baseline: {len(layouts)} layouts on "
        f"{REFINED_GRID_DEG.size} angles",
        flush=True,
    )

    with SensorDesignPipeline(
        alpha_grid_deg=REFINED_GRID_DEG,
        delta_alpha_min_deg=DELTA_ALPHA_MIN_DEG,
    ) as pipeline:
        d_baseline = pipeline.discrimination(
            pipeline.measurements(BASELINE_LAYOUT), tau
        )["discrimination"]

        # One packed WakeSurrogate call per angle covers every layout at once.
        sensor_x = layouts[:, [0, 2]].reshape(-1).astype(np.float32)
        sensor_y = layouts[:, [1, 3]].reshape(-1).astype(np.float32)

        batch = []
        for alpha in REFINED_GRID_DEG:
            outputs = pipeline._surrogate.apply(
                {
                    "angle_of_attack_deg": float(alpha),
                    "sensor_x": sensor_x,
                    "sensor_y": sensor_y,
                }
            )
            batch.append(np.asarray(outputs["measurements"], dtype=np.float32))

        stacked = np.stack(batch, axis=0).reshape(
            REFINED_GRID_DEG.size, len(layouts), 2, 5, 2
        )
        stacked = np.transpose(stacked, (1, 0, 2, 3, 4))

        scores = np.empty(len(layouts))
        for index in range(len(layouts)):
            scores[index] = pipeline.discrimination(stacked[index], tau)[
                "discrimination"
            ]
            if (index + 1) % 2500 == 0:
                print(f"  scored {index + 1}/{len(layouts)}", flush=True)

    np.savez_compressed(
        OUTPUT_DIR / "random_layout_scores_v2.npz", layouts=layouts, scores=scores
    )

    beating = int(np.sum(scores > d_star))

    summary = {
        "n_layouts": len(layouts),
        "seed": RANDOM_SEED,
        "grid_size": int(REFINED_GRID_DEG.size),
        "tau": tau,
        "D_baseline": d_baseline,
        "D_star_v2": d_star,
        "random_best": float(scores.max()),
        "random_median": float(np.median(scores)),
        "random_p95": float(np.percentile(scores, 95)),
        "random_p99": float(np.percentile(scores, 99)),
        "random_p999": float(np.percentile(scores, 99.9)),
        "s_star_v2_rank": beating + 1,
        "n_random_beating_s_star_v2": beating,
        "s_star_v2_percentile": 100.0 * float(np.mean(scores < d_star)),
        "fraction_beating_baseline": float(np.mean(scores > d_baseline)),
    }

    # Matched budget, bootstrapped from the scored population.
    rows = list(csv.DictReader((SURROGATE_DIR / "multistart_summary_v2.csv").open()))
    evaluations = np.array([int(r["objective_gradient_evaluations"]) for r in rows])
    best_row = rows[int(np.argmax([float(r["D_tau_final"]) for r in rows]))]

    rng = np.random.default_rng(RANDOM_SEED)
    matched = []
    for label, budget in (
        ("equal_candidates_best_run", int(best_row["objective_gradient_evaluations"])),
        ("equal_candidates_median_run", int(np.median(evaluations))),
    ):
        budget = max(1, min(budget, len(scores)))
        best_of = np.array(
            [
                scores[rng.choice(len(scores), size=budget, replace=False)].max()
                for _ in range(BOOTSTRAP_REPLICATES)
            ]
        )
        matched.append(
            {
                "comparison": label,
                "budget_candidates": budget,
                "random_best_median": float(np.median(best_of)),
                "random_best_p95": float(np.percentile(best_of, 95)),
                "gradient_D": d_star,
                "fraction_beating_gradient": float(np.mean(best_of > d_star)),
            }
        )

    summary["matched_budget"] = matched

    (OUTPUT_DIR / "random_baseline_v2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    for key, value in summary.items():
        if key != "matched_budget":
            print(f"  {key:32s} {value}")
    for row in matched:
        print(
            f"  {row['comparison']:32s} budget {row['budget_candidates']:4d}  "
            f"median {row['random_best_median']:.6f}  "
            f"beats gradient {row['fraction_beating_gradient']:.1%}"
        )

    print(f"\nWrote {OUTPUT_DIR / 'random_baseline_v2_summary.json'}")


if __name__ == "__main__":
    main()
