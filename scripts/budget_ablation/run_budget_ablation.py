"""Sensor budget versus sensor placement, evaluated on real CFD.

For each budget Ns in {1, 2, 3, 5, 8} this compares the conventional vertically
aligned probe array against a differentiably optimized array of the same size.
Designs come from T3 -> T4 with no CFD in the loop; both families are then
scored on the existing 66-angle real-CFD bank. No new CFD is run.

Cross-budget claims use the hard minimum retained pair distance, which is
normalised per scalar measurement and free of tau. D_tau is additionally
reported at a single common tau so the softmin view is comparable too.
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from ablation_core import (
    DELTA_ALPHA_MIN_DEG,
    DESIGN_GRID_DEG,
    MIN_SEPARATION,
    X_BOUNDS,
    Y_BOUNDS,
    AblationPipeline,
    bounds_for,
    canonicalize,
    min_pairwise_separation,
    naive_layout,
    pack,
    separation_penalty,
)

SENSOR_COUNTS = (1, 2, 3, 5, 8)

NAIVE_Y = {
    1: [0.0],
    2: [-0.4, 0.4],
    3: [-0.4, 0.0, 0.4],
    5: [-0.8, -0.4, 0.0, 0.4, 0.8],
    8: np.linspace(-0.9, 0.9, 8).round(6).tolist(),
}

N_STARTS = 10
RANDOM_SEED = 0

MAX_ITERATIONS = 300
F_TOLERANCE = 1.0e-9
G_TOLERANCE = 1.0e-6
TIE_TOLERANCE = 1.0e-6

REFERENCE_COUNT = 2

OUTPUT_DIR = Path("results/sensor_budget_ablation")


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


def random_layout(n_sensors: int, rng: np.random.Generator) -> np.ndarray:
    """A feasible random layout respecting the pairwise separation floor."""
    while True:
        x = rng.uniform(*X_BOUNDS, n_sensors)
        y = rng.uniform(*Y_BOUNDS, n_sensors)
        layout = pack(x, y)
        if n_sensors == 1 or min_pairwise_separation(layout) >= MIN_SEPARATION:
            return layout


def optimize_one(
    pipeline: AblationPipeline,
    layout: np.ndarray,
    tau: float,
    lambda_separation: float,
    n_sensors: int,
) -> tuple[np.ndarray, int, bool, str, float]:
    """One L-BFGS-B run; returns the final layout and run diagnostics."""
    evaluations = {"count": 0}

    def objective(design: np.ndarray) -> tuple[float, np.ndarray]:
        evaluations["count"] += 1
        return pipeline.surrogate_objective(
            design, tau=tau, lambda_separation=lambda_separation
        )

    started = time.perf_counter()

    result = minimize(
        objective,
        layout,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds_for(n_sensors),
        options={
            "maxiter": MAX_ITERATIONS,
            "ftol": F_TOLERANCE,
            "gtol": G_TOLERANCE,
        },
    )

    return (
        np.asarray(result.x, dtype=np.float64),
        evaluations["count"],
        bool(result.success),
        str(result.message),
        time.perf_counter() - started,
    )


def gradient_cost_comparison(
    pipeline: AblationPipeline,
    layout: np.ndarray,
    tau: float,
    lambda_separation: float,
) -> dict:
    """Reverse-mode gradient cost against a full central finite difference.

    Reverse mode needs one forward sweep plus one backward sweep regardless of
    how many design variables there are. A central difference needs two extra
    objective evaluations per variable, so its cost grows with the budget.
    """
    dimension = layout.size

    started = time.perf_counter()
    pipeline.surrogate_objective(layout, tau=tau, lambda_separation=lambda_separation)
    reverse_seconds = time.perf_counter() - started

    def value(design: np.ndarray) -> float:
        scored = pipeline.score(pipeline.surrogate_measurements(design), tau)
        penalty, _ = separation_penalty(design, lambda_separation=lambda_separation)
        return -scored["D_tau"] + penalty

    step = 1.0e-3
    started = time.perf_counter()
    for index in range(dimension):
        plus, minus = layout.copy(), layout.copy()
        plus[index] += step
        minus[index] -= step
        value(plus)
        value(minus)
    finite_seconds = time.perf_counter() - started

    return {
        "design_dimension": int(dimension),
        "reverse_mode_seconds": reverse_seconds,
        "central_difference_seconds": finite_seconds,
        "central_difference_objective_evaluations": 2 * dimension,
        "speedup": finite_seconds / reverse_seconds,
    }


def main() -> None:
    """Run the ablation across every sensor budget."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    started = time.perf_counter()

    rows: list[dict] = []
    designs: dict[str, list[float]] = {}
    scaling: list[dict] = []

    print("=" * 78)
    print("Sensor budget versus sensor placement")
    print("=" * 78)
    print(
        f"design grid : {DESIGN_GRID_DEG.size} angles, delta_alpha_min "
        f"{DELTA_ALPHA_MIN_DEG} deg"
    )
    print(f"budgets     : {list(SENSOR_COUNTS)}")
    print(f"starts      : {N_STARTS} per budget (naive + {N_STARTS - 1} random)")
    print(flush=True)

    with AblationPipeline() as pipeline:
        # A single common physical tau, calibrated at the reference budget's
        # naive array, so D_tau is comparable across budgets.
        reference_naive = naive_layout(REFERENCE_COUNT, NAIVE_Y[REFERENCE_COUNT])
        common_tau = pipeline.calibrate_tau(
            pipeline.physical_measurements(reference_naive)
        )

        print(
            f"common physical tau (from Ns={REFERENCE_COUNT} naive): "
            f"{common_tau:.10f}\n",
            flush=True,
        )

        for n_sensors in SENSOR_COUNTS:
            print("-" * 78)
            print(f"Ns = {n_sensors}")
            print("-" * 78, flush=True)

            naive = naive_layout(n_sensors, NAIVE_Y[n_sensors])

            # Per-budget surrogate tau, calibrated at that budget's naive array.
            naive_surrogate = pipeline.surrogate_measurements(naive)
            tau = pipeline.calibrate_tau(naive_surrogate)

            naive_scored = pipeline.score(naive_surrogate, tau)
            lambda_separation = (
                10.0
                * float(np.median(naive_scored["pair_distances"][pipeline.mask]))
                / MIN_SEPARATION**2
            )

            print(
                f"  surrogate tau {tau:.8f}  N_eff "
                f"{naive_scored['n_eff']:.3f}  pairs {naive_scored['n_pairs']}",
                flush=True,
            )

            # -------- multistart surrogate optimization --------
            starts = [naive.copy()] + [
                random_layout(n_sensors, rng) for _ in range(N_STARTS - 1)
            ]

            best = None

            for index, start in enumerate(starts):
                final, evaluations, converged, message, wall = optimize_one(
                    pipeline, start, tau, lambda_separation, n_sensors
                )
                scored = pipeline.score(pipeline.surrogate_measurements(final), tau)

                candidate = {
                    "layout": final,
                    "D": scored["D_tau"],
                    "evaluations": evaluations,
                    "converged": converged,
                    "message": message,
                    "wall": wall,
                    "start": "naive" if index == 0 else f"random{index:02d}",
                }

                if best is None or (
                    candidate["D"] > best["D"] + TIE_TOLERANCE
                    or (
                        abs(candidate["D"] - best["D"]) <= TIE_TOLERANCE
                        and candidate["converged"]
                        and not best["converged"]
                    )
                ):
                    best = candidate

            optimized = canonicalize(best["layout"])

            print(
                f"  best start {best['start']}  surrogate D "
                f"{naive_scored['D_tau']:.6f} -> {best['D']:.6f}  "
                f"({best['evaluations']} evals, "
                f"{'OK' if best['converged'] else 'STOP'})",
                flush=True,
            )

            designs[f"Ns{n_sensors}_naive"] = canonicalize(naive).tolist()
            designs[f"Ns{n_sensors}_optimized"] = optimized.tolist()

            # -------- real-CFD evaluation of BOTH families --------
            for family, layout in (("naive", naive), ("optimized", optimized)):
                physical = pipeline.physical_measurements(layout)

                common = pipeline.score(physical, common_tau)
                own = pipeline.score(physical, tau)

                rows.append(
                    {
                        "n_sensors": n_sensors,
                        "family": family,
                        "layout": json.dumps(
                            [round(v, 8) for v in canonicalize(layout).tolist()]
                        ),
                        "physical_hard_min": common["hard_min"],
                        "physical_D_common_tau": common["D_tau"],
                        "physical_D_own_tau": own["D_tau"],
                        "physical_n_eff_common_tau": common["n_eff"],
                        "hardest_pair_deg": str(common["hardest_pair_deg"]),
                        "min_separation": min_pairwise_separation(layout),
                        "n_scalar": int(n_sensors * 5 * 2),
                        "surrogate_D_own_tau": (
                            best["D"]
                            if family == "optimized"
                            else naive_scored["D_tau"]
                        ),
                        "own_tau": tau,
                        "common_tau": common_tau,
                        "optimizer_evaluations": (
                            best["evaluations"] if family == "optimized" else 0
                        ),
                    }
                )

                print(
                    f"    {family:9s} real-CFD hard_min "
                    f"{common['hard_min']:.6f}   D(common tau) "
                    f"{common['D_tau']:.6f}   hardest "
                    f"{common['hardest_pair_deg']}",
                    flush=True,
                )

            # -------- gradient scaling --------
            scaling.append(
                {
                    "n_sensors": n_sensors,
                    **gradient_cost_comparison(
                        pipeline, optimized, tau, lambda_separation
                    ),
                }
            )
            print(
                f"    gradient cost: reverse "
                f"{scaling[-1]['reverse_mode_seconds']:.2f} s vs central FD "
                f"{scaling[-1]['central_difference_seconds']:.2f} s "
                f"({scaling[-1]['speedup']:.1f}x, dim "
                f"{scaling[-1]['design_dimension']})",
                flush=True,
            )

    elapsed = time.perf_counter() - started

    # ------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------

    with (OUTPUT_DIR / "budget_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (OUTPUT_DIR / "gradient_scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scaling[0]))
        writer.writeheader()
        writer.writerows(scaling)

    lookup = {(r["n_sensors"], r["family"]): r for r in rows}

    headline = {
        "ns2_optimized_hard_min": lookup[(2, "optimized")]["physical_hard_min"],
        "ns5_naive_hard_min": lookup[(5, "naive")]["physical_hard_min"],
        "ns8_naive_hard_min": lookup[(8, "naive")]["physical_hard_min"],
        "ns2_optimized_beats_ns5_naive": (
            lookup[(2, "optimized")]["physical_hard_min"]
            > lookup[(5, "naive")]["physical_hard_min"]
        ),
        "ns2_optimized_beats_ns8_naive": (
            lookup[(2, "optimized")]["physical_hard_min"]
            > lookup[(8, "naive")]["physical_hard_min"]
        ),
        "ns1_optimized_hard_min": lookup[(1, "optimized")]["physical_hard_min"],
        "ns1_optimized_beats_ns5_naive": (
            lookup[(1, "optimized")]["physical_hard_min"]
            > lookup[(5, "naive")]["physical_hard_min"]
        ),
    }

    summary = {
        "experiment": "sensor budget versus sensor placement",
        "no_new_cfd": True,
        "evaluation_source": "existing 66-angle real-CFD bank",
        "common_tau": common_tau,
        "reference_count_for_common_tau": REFERENCE_COUNT,
        "designs": designs,
        "headline": headline,
        "gradient_scaling": scaling,
        "wall_time_s": elapsed,
        "git_commit": git_commit(),
    }

    (OUTPUT_DIR / "budget_ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print()
    print("=" * 78)
    print("Real-CFD hard minimum (tau-free, per-scalar normalised)")
    print("=" * 78)
    print(f"{'Ns':>3} {'naive':>12} {'optimized':>12} {'gain':>8}")
    for n_sensors in SENSOR_COUNTS:
        naive_value = lookup[(n_sensors, "naive")]["physical_hard_min"]
        optimized_value = lookup[(n_sensors, "optimized")]["physical_hard_min"]
        print(
            f"{n_sensors:3d} {naive_value:12.6f} {optimized_value:12.6f} "
            f"{(optimized_value - naive_value) / naive_value:+7.1%}"
        )

    print()
    print(
        f"  Ns=2 optimized ({headline['ns2_optimized_hard_min']:.6f}) vs "
        f"Ns=5 naive ({headline['ns5_naive_hard_min']:.6f}): "
        f"{'YES' if headline['ns2_optimized_beats_ns5_naive'] else 'no'}"
    )
    print(
        f"  Ns=2 optimized vs Ns=8 naive "
        f"({headline['ns8_naive_hard_min']:.6f}): "
        f"{'YES' if headline['ns2_optimized_beats_ns8_naive'] else 'no'}"
    )
    print(
        f"  Ns=1 optimized vs Ns=5 naive: "
        f"{'YES' if headline['ns1_optimized_beats_ns5_naive'] else 'no'}"
    )
    print(f"\n  wall time {elapsed / 60.0:.1f} min")
    print(f"\nWrote {OUTPUT_DIR / 'budget_ablation.csv'}")


if __name__ == "__main__":
    main()
