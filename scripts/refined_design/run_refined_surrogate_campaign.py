"""Phase-II surrogate sensor design on a 1 degree AoA grid.

Phase I designed against a 2.5 degree AoA grid. Real CFD then showed that the
alias dominating the inverse problem, 63 versus 83 degrees, is roughly three
times sharper than its nearest design-grid proxy, so the criterion was blind to
it. Phase II changes exactly one thing: the design grid becomes the integers
20..85, which represents that pair directly.

Nothing else moves. The CFD settings, T3 weights, T4 mathematics, sensor bounds,
separation floor, optimizer and tie rule are all unchanged, and delta_alpha_min
stays at 7.5 degrees so only resolution differs.

Phase-I artifacts are never written to.
"""

import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.sensor_design import (
    BASELINE_LAYOUT,
    DESIGN_BOUNDS,
    MIN_SENSOR_DISTANCE,
    SensorDesignPipeline,
    canonicalize_layout,
    retained_pair_mask,
)
from scipy.optimize import minimize

# ============================================================
# Phase-II configuration
# ============================================================

REFINED_GRID_DEG = np.arange(20.0, 85.0 + 1.0e-9, 1.0, dtype=np.float64)

DELTA_ALPHA_MIN_DEG = 7.5

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

TIE_TOLERANCE = 1.0e-6
CLUSTER_RADIUS = 0.05
NEAR_OPTIMAL_TOLERANCE = 0.005

OUTPUT_DIR = Path("results/sensor_design/refined_design/surrogate_v2")

CRITICAL_PAIR = (63.0, 83.0)


def git_commit() -> str:
    """Current HEAD, marked dirty when the tree has uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{head}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sample_random_starts(count: int, seed: int) -> dict[str, list[float]]:
    """Uniform feasible starting layouts, rejecting colocated sensors."""
    rng = np.random.default_rng(seed)

    starts: dict[str, list[float]] = {}

    while len(starts) < count:
        x = rng.uniform(DESIGN_BOUNDS[0][0], DESIGN_BOUNDS[0][1], 2)
        y = rng.uniform(DESIGN_BOUNDS[1][0], DESIGN_BOUNDS[1][1], 2)

        if np.hypot(x[0] - x[1], y[0] - y[1]) < MIN_SENSOR_DISTANCE:
            continue

        starts[f"R{len(starts) + 1:02d}_random"] = [
            float(x[0]),
            float(y[0]),
            float(x[1]),
            float(y[1]),
        ]

    return starts


def cluster_solutions(
    layouts: np.ndarray, radius: float = CLUSTER_RADIUS
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


def critical_pair_distance(
    pipeline: SensorDesignPipeline,
    design: np.ndarray,
    tau: float,
) -> float:
    """Physical-alias proxy: the surrogate distance between 63 and 83 degrees."""
    scored = pipeline.discrimination(pipeline.measurements(design), tau)

    i = int(np.where(REFINED_GRID_DEG == CRITICAL_PAIR[0])[0][0])
    j = int(np.where(REFINED_GRID_DEG == CRITICAL_PAIR[1])[0][0])

    return float(scored["pair_distances"][i, j])


def optimize_one(
    pipeline: SensorDesignPipeline,
    name: str,
    start: np.ndarray,
    calibration: dict[str, float],
) -> dict[str, object]:
    """One L-BFGS-B run against the refined objective."""
    tau = calibration["tau"]
    lambda_separation = calibration["lambda_separation"]

    evaluations = {"count": 0}

    def objective(design: np.ndarray) -> tuple[float, np.ndarray]:
        evaluations["count"] += 1
        return pipeline.objective_and_gradient(
            design, tau=tau, lambda_separation=lambda_separation
        )

    wall_start = time.perf_counter()

    result = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=DESIGN_BOUNDS,
        options={
            "maxiter": MAX_ITERATIONS,
            "ftol": F_TOLERANCE,
            "gtol": G_TOLERANCE,
        },
    )

    wall_time = time.perf_counter() - wall_start

    final = np.asarray(result.x, dtype=np.float64)

    _, _, diagnostics = pipeline.objective_gradient_and_diagnostics(
        final, tau=tau, lambda_separation=lambda_separation
    )

    _, _, initial = pipeline.objective_gradient_and_diagnostics(
        np.asarray(start, dtype=np.float64),
        tau=tau,
        lambda_separation=lambda_separation,
    )

    canonical = canonicalize_layout(final)

    print(
        f"  {name:20s} D {initial['discrimination']:.6f} -> "
        f"{diagnostics['discrimination']:.6f}   "
        f"hard_min {diagnostics['hard_min_distance']:.6f}   "
        f"evals {evaluations['count']:3d}  {wall_time:5.1f} s  "
        f"{'OK' if result.success else 'STOP'}",
        flush=True,
    )

    return {
        "start": name,
        "x1_initial": float(start[0]),
        "y1_initial": float(start[1]),
        "x2_initial": float(start[2]),
        "y2_initial": float(start[3]),
        "x1_final": canonical[0],
        "y1_final": canonical[1],
        "x2_final": canonical[2],
        "y2_final": canonical[3],
        "D_tau_initial": initial["discrimination"],
        "D_tau_final": diagnostics["discrimination"],
        "hard_min_initial": initial["hard_min_distance"],
        "hard_min_final": diagnostics["hard_min_distance"],
        "N_eff_final": diagnostics["effective_pairs"],
        "top1_weight_final": diagnostics["top1_weight"],
        "top10_weight_final": diagnostics["top10_weight"],
        "separation_final": diagnostics["separation"],
        "penalty_active": diagnostics["penalty"] > 0.0,
        "iterations": int(result.nit),
        "objective_gradient_evaluations": evaluations["count"],
        "converged": bool(result.success),
        "status_message": str(result.message),
        "wall_time_s": wall_time,
    }


def main() -> None:
    """Calibrate, optimize, then freeze s_star_surrogate_v2."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    starts = dict(FIXED_STARTS)
    starts.update(sample_random_starts(N_RANDOM_STARTS, RANDOM_SEED))

    mask = retained_pair_mask(REFINED_GRID_DEG, DELTA_ALPHA_MIN_DEG)

    print("=" * 78)
    print("Phase-II refined surrogate sensor design")
    print("=" * 78)
    print(
        f"design grid    : {REFINED_GRID_DEG.size} angles, "
        f"{REFINED_GRID_DEG[0]:.0f}..{REFINED_GRID_DEG[-1]:.0f} deg, step 1 deg"
    )
    print(f"delta_alpha_min: {DELTA_ALPHA_MIN_DEG} deg -> retains gaps >= 8 deg")
    print(f"retained pairs : {int(mask.sum())}")
    print(
        f"63 and 83 both on grid: "
        f"{bool(np.any(REFINED_GRID_DEG == 63.0) and np.any(REFINED_GRID_DEG == 83.0))}"
    )
    print()

    campaign_start = time.perf_counter()

    summaries = []

    with SensorDesignPipeline(
        alpha_grid_deg=REFINED_GRID_DEG,
        delta_alpha_min_deg=DELTA_ALPHA_MIN_DEG,
    ) as pipeline:
        calibration = pipeline.calibrate()

        tau = calibration["tau"]

        baseline_critical = critical_pair_distance(pipeline, BASELINE_LAYOUT, tau)

        distances = np.sort(
            pipeline.discrimination(pipeline.measurements(BASELINE_LAYOUT), tau)[
                "pair_distances"
            ][mask]
        )

        indices_i, indices_j = np.where(mask)
        raw = pipeline.discrimination(pipeline.measurements(BASELINE_LAYOUT), tau)[
            "pair_distances"
        ][mask]
        order = np.argsort(raw)[:5]
        hardest = [
            [
                float(REFINED_GRID_DEG[indices_i[k]]),
                float(REFINED_GRID_DEG[indices_j[k]]),
                float(raw[k]),
            ]
            for k in order
        ]

        print("Phase-II calibration at the original baseline layout:")
        print(f"  retained pairs      : {calibration['n_pairs']}")
        print(f"  tau (frozen)        : {tau:.10f}")
        print(f"  N_eff               : {calibration['effective_pairs']:.6f}")
        print(
            f"  top-1 / top-10      : {calibration['top1_weight']:.6f} / "
            f"{calibration['top10_weight']:.6f}"
        )
        print(
            f"  baseline D_tau      : "
            f"{pipeline.discrimination(pipeline.measurements(BASELINE_LAYOUT), tau)['discrimination']:.8f}"
        )
        print(f"  baseline hard min   : {distances[0]:.8f}")
        print(f"  baseline d(63,83)   : {baseline_critical:.8f}")
        print("  hardest pairs       :")
        for a, b, d in hardest:
            print(f"    ({a:.0f}, {b:.0f}) = {d:.6f}")
        print(flush=True)

        for name, layout in starts.items():
            summaries.append(
                optimize_one(
                    pipeline, name, np.asarray(layout, dtype=np.float64), calibration
                )
            )

        # ----------------------------------------------------
        # Selection: same deterministic rule as Phase I.
        # ----------------------------------------------------

        scores = np.array([s["D_tau_final"] for s in summaries])
        layouts = np.array(
            [[s[f"{c}_final"] for c in ("x1", "y1", "x2", "y2")] for s in summaries]
        )

        labels = cluster_solutions(layouts)

        argmax = int(np.argmax(scores))
        tied = np.where(scores >= scores.max() - TIE_TOLERANCE)[0]
        same_cluster = [
            i
            for i in tied
            if np.linalg.norm(layouts[i] - layouts[argmax]) <= CLUSTER_RADIUS
        ]
        converged = [i for i in same_cluster if summaries[i]["converged"]]
        pool = converged if converged else same_cluster

        chosen = min(
            pool,
            key=lambda i: (
                summaries[i]["objective_gradient_evaluations"],
                summaries[i]["start"],
            ),
        )

        s_star_v2 = canonicalize_layout(layouts[chosen])

        optimized_critical = critical_pair_distance(pipeline, s_star_v2, tau)

        final_scored = pipeline.discrimination(pipeline.measurements(s_star_v2), tau)
        baseline_scored = pipeline.discrimination(
            pipeline.measurements(BASELINE_LAYOUT), tau
        )

    campaign_time = time.perf_counter() - campaign_start

    with (OUTPUT_DIR / "multistart_summary_v2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    near_optimal = scores >= scores.max() * (1.0 - NEAR_OPTIMAL_TOLERANCE)

    frozen = {
        "s_star_surrogate_v2": {
            "x1": s_star_v2[0],
            "y1": s_star_v2[1],
            "x2": s_star_v2[2],
            "y2": s_star_v2[3],
        },
        "layout_vector": s_star_v2.tolist(),
        "phase": "II",
        "D_tau": final_scored["discrimination"],
        "D_tau_baseline": baseline_scored["discrimination"],
        "relative_improvement": (
            final_scored["discrimination"] - baseline_scored["discrimination"]
        )
        / baseline_scored["discrimination"],
        "hard_min_distance": final_scored["min_pair_distance"],
        "hard_min_baseline": baseline_scored["min_pair_distance"],
        "hard_min_relative_improvement": (
            final_scored["min_pair_distance"] - baseline_scored["min_pair_distance"]
        )
        / baseline_scored["min_pair_distance"],
        "critical_pair_deg": list(CRITICAL_PAIR),
        "critical_pair_distance_baseline": baseline_critical,
        "critical_pair_distance_s_star_v2": optimized_critical,
        "critical_pair_relative_improvement": (optimized_critical - baseline_critical)
        / baseline_critical,
        "source_run": summaries[chosen]["start"],
        "source_converged": summaries[chosen]["converged"],
        "source_convergence_status": summaries[chosen]["status_message"],
        "selection_rule": (
            "Maximum refined-grid T4 discrimination; layouts within the tie "
            "tolerance are numerically tied, and among tied layouts in the same "
            "cluster a normally converged termination is preferred, then the "
            "fewest evaluations. No physical or CFD result influenced selection."
        ),
        "tie_tolerance": TIE_TOLERANCE,
        "tau": tau,
        "tau_calibration": (
            "solved at the original baseline layout on the 66-angle grid so "
            "that N_eff = 10; frozen for the campaign"
        ),
        "n_eff_baseline": calibration["effective_pairs"],
        "delta_alpha_min_deg": DELTA_ALPHA_MIN_DEG,
        "delta_alpha_min_note": (
            "On an integer grid the >= 7.5 deg condition retains exactly the "
            "pairs separated by >= 8 deg."
        ),
        "design_grid_deg": REFINED_GRID_DEG.tolist(),
        "n_retained_pairs": int(mask.sum()),
        "n_clusters": int(labels.max() + 1),
        "n_near_optimal": int(near_optimal.sum()),
        "baseline_hardest_pairs": hardest,
        "phase1_design_untouched": True,
        "immutability": (
            "Frozen Phase-II surrogate proposal. Separate from the Phase-I "
            "s_star_surrogate, which remains unmodified."
        ),
        "campaign_wall_time_s": campaign_time,
        "git_commit": git_commit(),
    }

    (OUTPUT_DIR / "s_star_surrogate_v2.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )

    print()
    print("=" * 78)
    print("Phase-II surrogate result")
    print("=" * 78)
    print(
        f"selected from     : {frozen['source_run']} "
        f"(converged={frozen['source_converged']})"
    )
    print(f"s_star_surrogate_v2: {s_star_v2.tolist()}")
    print(
        f"D_tau              : {frozen['D_tau_baseline']:.8f} -> "
        f"{frozen['D_tau']:.8f}  ({frozen['relative_improvement']:+.2%})"
    )
    print(
        f"hard minimum       : {frozen['hard_min_baseline']:.8f} -> "
        f"{frozen['hard_min_distance']:.8f}  "
        f"({frozen['hard_min_relative_improvement']:+.2%})"
    )
    print(
        f"d(63,83) surrogate : {baseline_critical:.8f} -> {optimized_critical:.8f}  "
        f"({frozen['critical_pair_relative_improvement']:+.2%})"
    )
    print(
        f"clusters           : {frozen['n_clusters']}   "
        f"near-optimal {frozen['n_near_optimal']}/{len(scores)}"
    )
    print(
        f"best/median/worst D: {scores.max():.6f} / {np.median(scores):.6f} / "
        f"{scores.min():.6f}"
    )
    print(f"wall time          : {campaign_time / 60.0:.2f} min")
    print()
    print(f"Wrote {OUTPUT_DIR / 'multistart_summary_v2.csv'}")
    print(f"Wrote {OUTPUT_DIR / 's_star_surrogate_v2.json'}")


if __name__ == "__main__":
    main()
