"""Robustness of the frozen design to the pair-separation threshold.

s_star_surrogate was selected once, under delta_alpha_min = 7.5 deg. Here it is
only *evaluated* under other thresholds -- no reselection, no reoptimization.

Scores are not comparable across thresholds: each delta retains a different
pair set, so the criterion is a different functional. Only the baseline-versus-
optimized comparison within a single delta is meaningful, which is what the
relative improvement column reports.
"""

import csv
import json
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.sensor_design import (
    BASELINE_LAYOUT,
    SensorDesignPipeline,
)

DELTAS = (5.0, 7.5, 10.0, 15.0)

OPTIMIZATION_DIR = Path("results/sensor_design/optimization")
OUTPUT_CSV = OPTIMIZATION_DIR / "delta_robustness.csv"


def main() -> None:
    """Evaluate baseline and the frozen design across pair thresholds."""
    selection = json.loads((OPTIMIZATION_DIR / "s_star_surrogate.json").read_text())

    s_star = np.array(selection["layout_vector"], dtype=np.float64)

    print(f"frozen s_star_surrogate: {s_star.tolist()}")
    print(f"selected under delta   : {selection['delta_alpha_min_deg']} deg")
    print()

    rows = []

    for delta in DELTAS:
        with SensorDesignPipeline(delta_alpha_min_deg=delta) as pipeline:
            # tau is recalibrated at the ORIGINAL baseline for each threshold,
            # so N_eff = 10 holds for every pair definition.
            calibration = pipeline.calibrate()

            tau = calibration["tau"]

            baseline = pipeline.discrimination(
                pipeline.measurements(BASELINE_LAYOUT), tau
            )
            optimized = pipeline.discrimination(pipeline.measurements(s_star), tau)

        improvement = (optimized["discrimination"] - baseline["discrimination"]) / abs(
            baseline["discrimination"]
        )

        rows.append(
            {
                "delta_alpha_min_deg": delta,
                "n_pairs": baseline["n_pairs"],
                "tau": tau,
                "effective_pairs": calibration["effective_pairs"],
                "D_baseline": baseline["discrimination"],
                "D_s_star": optimized["discrimination"],
                "absolute_improvement": optimized["discrimination"]
                - baseline["discrimination"],
                "relative_improvement": improvement,
                "hard_min_baseline": baseline["min_pair_distance"],
                "hard_min_s_star": optimized["min_pair_distance"],
                "hard_min_relative_improvement": (
                    optimized["min_pair_distance"] - baseline["min_pair_distance"]
                )
                / baseline["min_pair_distance"],
            }
        )

        print(
            f"  delta={delta:4.1f}  pairs={baseline['n_pairs']:3d}  "
            f"tau={tau:.6f}  D: {baseline['discrimination']:.6f} -> "
            f"{optimized['discrimination']:.6f}  ({improvement:+.1%})   "
            f"hard_min: {baseline['min_pair_distance']:.6f} -> "
            f"{optimized['min_pair_distance']:.6f}  "
            f"({rows[-1]['hard_min_relative_improvement']:+.1%})",
            flush=True,
        )

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
