"""Production multistart campaign for the differentiable sensor-array design.

Twenty starts -- ten fixed layouts spanning the design box plus ten seeded
random ones -- each optimized with L-BFGS-B against the frozen T4 score.

Everything the optimizer consumes comes from Tesseract derivative endpoints:

    SensorArrayDesign.vector_jacobian_product
      -> WakeSurrogate.vector_jacobian_product, once per design AoA
      -> dD/d[x1, y1, x2, y2]

The 63 degree sealed truth is not on the design grid and is never evaluated.
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
    canonicalize_layout,
)
from scipy.optimize import minimize

# ============================================================
# Configuration
# ============================================================

FIXED_STARTS = {
    "F01_baseline": [1.0, -0.4, 1.0, 0.4],
    "F02_near_wide": [1.0, -0.9, 1.0, 0.9],
    "F03_streamwise": [1.0, 0.0, 2.0, 0.0],
    "F04_mid_vertical": [2.0, -0.5, 2.0, 0.5],
    "F05_far_vertical": [3.0, -0.6, 3.0, 0.6],
    "F06_diagonal": [1.2, -0.8, 2.8, 0.8],
    "F07_antidiagonal": [1.2, 0.8, 2.8, -0.8],
    "F08_near_far_asym": [1.1, 0.2, 2.6, -0.7],
    "F09_far_close": [2.6, 0.15, 3.0, -0.25],
    "F10_centre_offaxis": [1.5, 0.0, 2.2, 0.85],
}

N_RANDOM_STARTS = 10
RANDOM_SEED = 0

MAX_ITERATIONS = 200
F_TOLERANCE = 1.0e-9
G_TOLERANCE = 1.0e-6

OUTPUT_DIR = Path("results/sensor_design/optimization")
TRAJECTORY_DIR = OUTPUT_DIR / "trajectories"

SUMMARY_CSV = OUTPUT_DIR / "multistart_summary.csv"
STARTS_CSV = OUTPUT_DIR / "multistart_starts.csv"
CALIBRATION_JSON = OUTPUT_DIR / "calibration.json"

# Layouts within this relative distance of the best score count as
# near-optimal when characterizing the landscape.
NEAR_OPTIMAL_TOLERANCE = 0.005

# Two final layouts closer than this in the 4-D design space (after
# canonicalization) are treated as the same solution cluster.
CLUSTER_RADIUS = 0.05


def sample_random_starts(
    count: int,
    seed: int,
    min_distance: float = MIN_SENSOR_DISTANCE,
) -> dict[str, list[float]]:
    """Uniform feasible starting layouts, rejecting colocated sensors."""
    rng = np.random.default_rng(seed)

    starts: dict[str, list[float]] = {}

    while len(starts) < count:
        x = rng.uniform(DESIGN_BOUNDS[0][0], DESIGN_BOUNDS[0][1], 2)
        y = rng.uniform(DESIGN_BOUNDS[1][0], DESIGN_BOUNDS[1][1], 2)

        if np.hypot(x[0] - x[1], y[0] - y[1]) < min_distance:
            continue

        starts[f"R{len(starts) + 1:02d}_random"] = [
            float(x[0]),
            float(y[0]),
            float(x[1]),
            float(y[1]),
        ]

    return starts


def optimize_one(
    pipeline: SensorDesignPipeline,
    name: str,
    start: np.ndarray,
    calibration: dict[str, float],
) -> tuple[dict[str, object], list[dict[str, float]]]:
    """Run one L-BFGS-B optimization, recording every evaluation."""
    tau = calibration["tau"]
    lambda_separation = calibration["lambda_separation"]

    trajectory: list[dict[str, float]] = []

    def objective(design: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, diagnostics = pipeline.objective_gradient_and_diagnostics(
            design,
            tau=tau,
            lambda_separation=lambda_separation,
        )

        trajectory.append(
            {
                "evaluation": len(trajectory),
                "x1": float(design[0]),
                "y1": float(design[1]),
                "x2": float(design[2]),
                "y2": float(design[3]),
                "D_tau": diagnostics["discrimination"],
                "hard_min_distance": diagnostics["hard_min_distance"],
                "N_eff": diagnostics["effective_pairs"],
                "separation": diagnostics["separation"],
                "objective": value,
            }
        )

        return value, gradient

    iteration_marks: list[int] = []

    def on_iteration(design: np.ndarray) -> None:
        iteration_marks.append(len(trajectory) - 1)

    wall_start = time.perf_counter()

    result = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=DESIGN_BOUNDS,
        callback=on_iteration,
        options={
            "maxiter": MAX_ITERATIONS,
            "ftol": F_TOLERANCE,
            "gtol": G_TOLERANCE,
        },
    )

    wall_time = time.perf_counter() - wall_start

    # Tag each recorded evaluation with the iteration it belongs to.
    for row in trajectory:
        row["iteration"] = int(
            np.searchsorted(iteration_marks, row["evaluation"], side="left")
        )

    final = np.asarray(result.x, dtype=np.float64)

    _, _, final_diagnostics = pipeline.objective_gradient_and_diagnostics(
        final, tau=tau, lambda_separation=lambda_separation
    )

    initial = trajectory[0]

    canonical = canonicalize_layout(final)

    summary = {
        "start": name,
        "x1_initial": float(start[0]),
        "y1_initial": float(start[1]),
        "x2_initial": float(start[2]),
        "y2_initial": float(start[3]),
        "x1_final": canonical[0],
        "y1_final": canonical[1],
        "x2_final": canonical[2],
        "y2_final": canonical[3],
        "D_tau_initial": initial["D_tau"],
        "D_tau_final": final_diagnostics["discrimination"],
        "hard_min_initial": initial["hard_min_distance"],
        "hard_min_final": final_diagnostics["hard_min_distance"],
        "N_eff_final": final_diagnostics["effective_pairs"],
        "top1_weight_final": final_diagnostics["top1_weight"],
        "top10_weight_final": final_diagnostics["top10_weight"],
        "separation_final": final_diagnostics["separation"],
        "penalty_active": final_diagnostics["penalty"] > 0.0,
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective_gradient_evaluations": len(trajectory),
        "converged": bool(result.success),
        "status_message": str(result.message),
        "wall_time_s": wall_time,
    }

    print(
        f"  {name:20s} D_tau {initial['D_tau']:.6f} -> "
        f"{final_diagnostics['discrimination']:.6f}   "
        f"hard_min {final_diagnostics['hard_min_distance']:.6f}   "
        f"evals {len(trajectory):3d}   {wall_time:5.1f} s   "
        f"{'OK' if result.success else 'STOP'}",
        flush=True,
    )

    return summary, trajectory


def cluster_solutions(
    layouts: np.ndarray,
    radius: float = CLUSTER_RADIUS,
) -> np.ndarray:
    """Greedy single-pass clustering of canonicalized final layouts."""
    labels = -np.ones(len(layouts), dtype=int)

    next_label = 0

    for index, layout in enumerate(layouts):
        if labels[index] >= 0:
            continue

        labels[index] = next_label

        for other in range(index + 1, len(layouts)):
            if labels[other] < 0 and np.linalg.norm(layouts[other] - layout) <= radius:
                labels[other] = next_label

        next_label += 1

    return labels


def main() -> None:
    """Run the campaign and write all artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    starts = dict(FIXED_STARTS)
    starts.update(sample_random_starts(N_RANDOM_STARTS, RANDOM_SEED))

    campaign_start = time.perf_counter()

    print("=" * 78, flush=True)
    print("Production multistart sensor-array design campaign", flush=True)
    print("=" * 78, flush=True)
    print(
        f"AoA grid        : {ALPHA_GRID_DEG[0]:.1f}..{ALPHA_GRID_DEG[-1]:.1f} deg, "
        f"{ALPHA_GRID_DEG.size} points",
        flush=True,
    )
    print(f"63 deg in grid  : {bool(np.any(ALPHA_GRID_DEG == 63.0))}", flush=True)
    print(f"delta_alpha_min : {DELTA_ALPHA_MIN_DEG} deg", flush=True)
    print(f"starts          : {len(starts)}", flush=True)
    print(flush=True)

    with STARTS_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start", "x1", "y1", "x2", "y2", "separation"])
        for name, layout in starts.items():
            writer.writerow(
                [
                    name,
                    *layout,
                    float(np.hypot(layout[0] - layout[2], layout[1] - layout[3])),
                ]
            )

    summaries: list[dict[str, object]] = []

    with SensorDesignPipeline() as pipeline:
        calibration = pipeline.calibrate()

        print("Calibration at the baseline layout (frozen for the campaign):")
        for key in (
            "n_pairs",
            "min_distance",
            "median_distance",
            "tau",
            "effective_pairs",
            "top1_weight",
            "top10_weight",
            "lambda_separation",
        ):
            print(f"  {key:20s} {calibration[key]}", flush=True)
        print(flush=True)

        CALIBRATION_JSON.write_text(json.dumps(calibration, indent=2) + "\n")

        # Sensor labels are physical fictions: swapping them must not move
        # the score. Checked on the baseline before anything is optimized.
        baseline_measurements = pipeline.measurements(BASELINE_LAYOUT)
        swapped_layout = BASELINE_LAYOUT[[2, 3, 0, 1]]
        swapped_measurements = pipeline.measurements(swapped_layout)

        score_original = pipeline.discrimination(
            baseline_measurements, calibration["tau"]
        )["discrimination"]
        score_swapped = pipeline.discrimination(
            swapped_measurements, calibration["tau"]
        )["discrimination"]

        print(
            f"Sensor-label swap invariance: {score_original:.10f} vs "
            f"{score_swapped:.10f}  |diff| = "
            f"{abs(score_original - score_swapped):.3e}",
            flush=True,
        )
        print(flush=True)

        for name, layout in starts.items():
            summary, trajectory = optimize_one(
                pipeline,
                name,
                np.asarray(layout, dtype=np.float64),
                calibration,
            )

            summaries.append(summary)

            with (TRAJECTORY_DIR / f"{name}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "iteration",
                        "evaluation",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "D_tau",
                        "hard_min_distance",
                        "N_eff",
                        "separation",
                        "objective",
                    ],
                )
                writer.writeheader()
                writer.writerows(trajectory)

    campaign_time = time.perf_counter() - campaign_start

    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    # Selection is deliberately NOT done here. It lives in
    # scripts/sensor_design/select_frozen_design.py so the tie-breaking rule
    # can be audited and re-run without repeating the campaign.

    scores = np.array([s["D_tau_final"] for s in summaries], dtype=np.float64)

    layouts = np.array(
        [
            [s["x1_final"], s["y1_final"], s["x2_final"], s["y2_final"]]
            for s in summaries
        ]
    )

    labels = cluster_solutions(layouts)

    near_optimal = scores >= scores.max() * (1.0 - NEAR_OPTIMAL_TOLERANCE)

    print()
    print("=" * 78)
    print("Campaign result")
    print("=" * 78)
    print(
        f"best / median / worst D_tau : "
        f"{scores.max():.6f} / {np.median(scores):.6f} / {scores.min():.6f}"
    )
    print(f"solution clusters : {labels.max() + 1}")
    print(
        f"near-optimal (<= {NEAR_OPTIMAL_TOLERANCE:.1%}): "
        f"{int(near_optimal.sum())} / {len(scores)}"
    )
    print(f"campaign wall time: {campaign_time / 60.0:.2f} min")
    print()
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {STARTS_CSV}")
    print(f"Wrote {TRAJECTORY_DIR}/*.csv")
    print()
    print("Next: python scripts/sensor_design/select_frozen_design.py")


if __name__ == "__main__":
    main()
