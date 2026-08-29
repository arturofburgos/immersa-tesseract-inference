"""Recover the L-BFGS-B iterates of the frozen Ns=5 budget-ablation design.

The committed campaign (run_budget_ablation.py) stored only each start's final
layout, so the optimization path that produced the frozen Ns=5 design was never
persisted. This script replays that optimization deterministically and records
the iterates, purely so the hero animation can show the real path.

Nothing about the optimization changes. The RNG stream, starts, objective,
bounds, separation penalty, tau calibration, optimizer, tolerances and model
weights are all the committed ones; the only addition is a scipy callback that
appends x_k. The replay is rejected unless the reconstructed winner reproduces
the frozen Ns5_optimized coordinates.

The original campaign draws its random starts from a single default_rng(0)
consumed in budget order (1, 2, 3, 5, 8), nine draws each, and uses the
generator nowhere else. The earlier budgets are therefore replayed by consuming
their draws only -- no optimization is repeated for them.

    python scripts/budget_ablation/replay_ns5_optimization.py
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from ablation_core import (
    MIN_SEPARATION,
    AblationPipeline,
    bounds_for,
    canonicalize,
    naive_layout,
    pack,
)
from run_budget_ablation import (
    F_TOLERANCE,
    G_TOLERANCE,
    MAX_ITERATIONS,
    N_STARTS,
    NAIVE_Y,
    RANDOM_SEED,
    SENSOR_COUNTS,
    TIE_TOLERANCE,
    random_layout,
)

TARGET_COUNT = 5

RESULTS_DIR = Path("results/sensor_budget_ablation")
SUMMARY_FILE = RESULTS_DIR / "budget_ablation_summary.json"

OUTPUT_CSV = RESULTS_DIR / "ns5_optimization_replay.csv"
OUTPUT_JSON = RESULTS_DIR / "ns5_optimization_replay.json"

# The replay must land on the frozen design, not merely near it. float32 JAX
# evaluation noise is of order 1e-7, so anything at 1e-9 is an exact rerun.
COORDINATE_TOLERANCE = 1.0e-9


def replay_starts(n_sensors: int) -> list[np.ndarray]:
    """The committed starts for one budget, from the original RNG stream."""
    rng = np.random.default_rng(RANDOM_SEED)

    for count in SENSOR_COUNTS:
        starts = [naive_layout(count, NAIVE_Y[count])] + [
            random_layout(count, rng) for _ in range(N_STARTS - 1)
        ]
        if count == n_sensors:
            return starts

    raise ValueError(f"{n_sensors} is not one of the committed budgets.")


def optimize_recording(
    pipeline: AblationPipeline,
    layout: np.ndarray,
    tau: float,
    lambda_separation: float,
    n_sensors: int,
) -> dict:
    """One committed L-BFGS-B run, with the iterates recorded."""
    evaluations = {"count": 0}

    iterates = [np.asarray(layout, dtype=np.float64).copy()]

    def objective(design: np.ndarray) -> tuple[float, np.ndarray]:
        evaluations["count"] += 1
        return pipeline.surrogate_objective(
            design, tau=tau, lambda_separation=lambda_separation
        )

    def record(xk: np.ndarray) -> None:
        """Informational only; returning None never alters L-BFGS-B."""
        iterates.append(np.asarray(xk, dtype=np.float64).copy())

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
        callback=record,
    )

    final = np.asarray(result.x, dtype=np.float64)

    if not np.array_equal(final, iterates[-1]):
        iterates.append(final.copy())

    return {
        "layout": final,
        "iterates": iterates,
        "evaluations": evaluations["count"],
        "converged": bool(result.success),
        "message": str(result.message),
        "objective": float(result.fun),
        "wall": time.perf_counter() - started,
    }


def main() -> None:
    """Replay every Ns=5 start and validate the winner against the artifact."""
    summary = json.loads(SUMMARY_FILE.read_text())
    frozen = np.asarray(
        summary["designs"][f"Ns{TARGET_COUNT}_optimized"], dtype=np.float64
    )

    starts = replay_starts(TARGET_COUNT)

    print("=" * 78)
    print(f"Deterministic replay of the frozen Ns={TARGET_COUNT} design optimization")
    print("=" * 78)
    print(f"starts        : {len(starts)} (naive + {len(starts) - 1} random)")
    print(f"frozen target : {np.round(frozen, 6).tolist()}")
    print(flush=True)

    total_started = time.perf_counter()

    with AblationPipeline() as pipeline:
        naive = naive_layout(TARGET_COUNT, NAIVE_Y[TARGET_COUNT])

        naive_surrogate = pipeline.surrogate_measurements(naive)
        tau = pipeline.calibrate_tau(naive_surrogate)

        naive_scored = pipeline.score(naive_surrogate, tau)
        lambda_separation = (
            10.0
            * float(np.median(naive_scored["pair_distances"][pipeline.mask]))
            / MIN_SEPARATION**2
        )

        print(f"surrogate tau      : {tau:.10f}")
        print(f"lambda_separation  : {lambda_separation:.10f}")
        print(f"N_eff at naive     : {naive_scored['n_eff']:.3f}\n", flush=True)

        runs = []
        best = None

        for index, start in enumerate(starts):
            name = "naive" if index == 0 else f"random{index:02d}"

            run = optimize_recording(
                pipeline, start, tau, lambda_separation, TARGET_COUNT
            )
            run["start"] = name

            scored = pipeline.score(pipeline.surrogate_measurements(run["layout"]), tau)
            run["D"] = scored["D_tau"]

            runs.append(run)

            print(
                f"  {name:9s} D={run['D']:.10f}  "
                f"{len(run['iterates']):3d} iterates  "
                f"{run['evaluations']:3d} evaluations  "
                f"{'converged' if run['converged'] else 'ABNORMAL '}  "
                f"{run['wall']:.1f}s",
                flush=True,
            )

            # The committed selection rule, unchanged.
            if best is None or (
                run["D"] > best["D"] + TIE_TOLERANCE
                or (
                    abs(run["D"] - best["D"]) <= TIE_TOLERANCE
                    and run["converged"]
                    and not best["converged"]
                )
            ):
                best = run

    elapsed = time.perf_counter() - total_started

    # ------------------------------------------------------------
    # Validation against the frozen artifact
    # ------------------------------------------------------------

    reconstructed = canonicalize(best["layout"])

    coordinate_error = float(np.max(np.abs(reconstructed - frozen)))

    # The campaign stored the surrogate discrimination, not the penalised
    # objective value, so the endpoint comparison is made on D.
    frozen_D = None
    with (RESULTS_DIR / "budget_ablation.csv").open() as handle:
        for row in csv.DictReader(handle):
            if int(row["n_sensors"]) == TARGET_COUNT and row["family"] == "optimized":
                frozen_D = float(row["surrogate_D_own_tau"])

    if frozen_D is None:
        raise RuntimeError("No frozen optimized row for this budget.")

    objective_error = abs(best["D"] - frozen_D)

    print()
    print("=" * 78)
    print("Validation")
    print("=" * 78)
    print(f"  winning start          : {best['start']}")
    print(f"  L-BFGS-B iterates      : {len(best['iterates'])}")
    print(f"  objective evaluations  : {best['evaluations']}")
    print(f"  termination            : {best['message']}")
    print(f"  reconstructed          : {np.round(reconstructed, 9).tolist()}")
    print(f"  frozen                 : {np.round(frozen, 9).tolist()}")
    print(f"  max |coordinate diff|  : {coordinate_error:.3e}")
    print(f"  surrogate D replay     : {best['D']:.12f}")
    print(f"  surrogate D frozen     : {frozen_D:.12f}")
    print(f"  |D difference|         : {objective_error:.3e}")

    if coordinate_error > COORDINATE_TOLERANCE:
        raise SystemExit(
            f"\nREPLAY REJECTED: reconstructed winner differs from the frozen "
            f"design by {coordinate_error:.3e} > {COORDINATE_TOLERANCE:.0e}. "
            f"No trajectory will be written."
        )

    print("\n  REPLAY ACCEPTED: the frozen design is reproduced exactly.")

    # ------------------------------------------------------------
    # Persist the full trajectory
    # ------------------------------------------------------------

    # Report the winner in the canonical (frozen) probe ordering so the
    # animation ends on exactly the stored coordinates. The permutation is a
    # relabelling of unlabelled probes and is applied to every iterate.
    order = np.lexsort((best["layout"][1::2], best["layout"][0::2]))

    trajectory = [
        pack(iterate[0::2][order], iterate[1::2][order]) for iterate in best["iterates"]
    ]

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["iterate"]
            + [f"{axis}{i + 1}" for i in range(TARGET_COUNT) for axis in ("x", "y")]
        )
        for index, iterate in enumerate(trajectory):
            writer.writerow([index, *iterate.tolist()])

    report = {
        "purpose": (
            "Deterministic replay of the frozen Ns=5 budget-ablation optimization, "
            "recorded only to recover the L-BFGS-B iterates for presentation. "
            "No scientific quantity is redefined or recomputed here."
        ),
        "source_campaign": "scripts/budget_ablation/run_budget_ablation.py",
        "frozen_artifact": str(SUMMARY_FILE),
        "n_sensors": TARGET_COUNT,
        "winning_start": best["start"],
        "n_iterates": len(trajectory),
        "objective_evaluations": best["evaluations"],
        "converged": best["converged"],
        "termination_message": best["message"],
        "tau": tau,
        "lambda_separation": lambda_separation,
        "reconstructed_layout": reconstructed.tolist(),
        "frozen_layout": frozen.tolist(),
        "max_absolute_coordinate_difference": coordinate_error,
        "surrogate_D_replay": best["D"],
        "surrogate_D_frozen": frozen_D,
        "absolute_D_difference": objective_error,
        "coordinate_tolerance": COORDINATE_TOLERANCE,
        "validated": True,
        "trajectory": [iterate.tolist() for iterate in trajectory],
        "all_starts": [
            {
                "start": run["start"],
                "D": run["D"],
                "n_iterates": len(run["iterates"]),
                "evaluations": run["evaluations"],
                "converged": run["converged"],
            }
            for run in runs
        ],
        "wall_time_s": elapsed,
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n  wall time: {elapsed / 60.0:.1f} min")
    print(f"\nWrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
