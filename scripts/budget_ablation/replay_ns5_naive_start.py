"""Replay only the Ns=5 start that begins from the conventional naive rake.

The ten-start replay recorded every start's discrimination but persisted the
coordinates of the winner alone. This script re-runs the single naive-start
trajectory -- the same deterministic optimization, same objective, same
settings -- so its endpoint can be compared against the frozen best-of-ten
design.

This answers one question only: does starting from the conventional rake reach
the same optimum, or a weaker one? Nothing here is a new scientific result and
no frozen artifact is written to.

    python scripts/budget_ablation/replay_ns5_naive_start.py
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ablation_core import MIN_SEPARATION, AblationPipeline, canonicalize
from replay_ns5_optimization import (
    RESULTS_DIR,
    SUMMARY_FILE,
    TARGET_COUNT,
    optimize_recording,
    replay_starts,
)

OUTPUT_CSV = RESULTS_DIR / "ns5_naive_start_replay.csv"
OUTPUT_JSON = RESULTS_DIR / "ns5_naive_start_replay.json"


def frozen_reference() -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Frozen layouts and the campaign's own scores for this budget."""
    summary = json.loads(SUMMARY_FILE.read_text())

    naive = np.asarray(summary["designs"][f"Ns{TARGET_COUNT}_naive"], dtype=np.float64)
    best = np.asarray(
        summary["designs"][f"Ns{TARGET_COUNT}_optimized"], dtype=np.float64
    )

    surrogate: dict[str, float] = {}
    physical: dict[str, float] = {}

    with (RESULTS_DIR / "budget_ablation.csv").open() as handle:
        for row in csv.DictReader(handle):
            if int(row["n_sensors"]) == TARGET_COUNT:
                surrogate[row["family"]] = float(row["surrogate_D_own_tau"])
                physical[row["family"]] = float(row["physical_hard_min"])

    return (
        naive,
        best,
        surrogate["naive"],
        surrogate["optimized"],
        physical["naive"],
    )


def main() -> None:
    """Run the naive-start trajectory and compare it with the frozen winner."""
    naive, frozen_best, d_naive, d_best, physical_naive = frozen_reference()

    starts = replay_starts(TARGET_COUNT)
    start = starts[0]

    if not np.array_equal(start, naive):
        raise SystemExit("Start 0 is not the frozen naive rake.")

    print("=" * 78)
    print(f"Naive-start replay, Ns={TARGET_COUNT}")
    print("=" * 78)
    print(f"  start (naive rake) : {np.round(start, 6).tolist()}")
    print(f"  frozen best-of-10  : {np.round(frozen_best, 6).tolist()}")
    print(flush=True)

    started = time.perf_counter()

    with AblationPipeline() as pipeline:
        naive_surrogate = pipeline.surrogate_measurements(naive)
        tau = pipeline.calibrate_tau(naive_surrogate)

        naive_scored = pipeline.score(naive_surrogate, tau)
        lambda_separation = (
            10.0
            * float(np.median(naive_scored["pair_distances"][pipeline.mask]))
            / MIN_SEPARATION**2
        )

        # Sanity: the campaign's own naive score must come back.
        if abs(naive_scored["D_tau"] - d_naive) > 1.0e-9:
            raise SystemExit(
                f"Naive baseline D mismatch: {naive_scored['D_tau']} vs {d_naive}"
            )

        print(f"  tau                : {tau:.10f}")
        print(f"  naive baseline D   : {naive_scored['D_tau']:.12f}  (reproduced)")
        print("\n  optimizing from the naive rake...", flush=True)

        run = optimize_recording(pipeline, start, tau, lambda_separation, TARGET_COUNT)

        final_scored = pipeline.score(
            pipeline.surrogate_measurements(run["layout"]), tau
        )

        # Same real-CFD metric the ablation headline uses, on the existing bank.
        physical_scored = pipeline.score(
            pipeline.physical_measurements(run["layout"]),
            float(json.loads(SUMMARY_FILE.read_text())["common_tau"]),
        )

    elapsed = time.perf_counter() - started

    final = canonicalize(run["layout"])

    d_final = final_scored["D_tau"]

    gain_over_naive = (d_final - d_naive) / d_naive
    best_gain_over_naive = (d_best - d_naive) / d_naive
    shortfall = (d_final - d_best) / d_best

    distance = float(np.linalg.norm(final - frozen_best))
    max_coordinate = float(np.max(np.abs(final - frozen_best)))

    physical_hard_min = physical_scored["hard_min"]
    physical_gain = (physical_hard_min - physical_naive) / physical_naive

    print()
    print("=" * 78)
    print("Naive-start optimum")
    print("=" * 78)
    print(f"  iterates              : {len(run['iterates'])}")
    print(f"  objective evaluations : {run['evaluations']}")
    print(f"  converged             : {run['converged']}")
    print(f"  termination           : {run['message']}")
    print(f"  final x               : {np.round(final[0::2], 6).tolist()}")
    print(f"  final y               : {np.round(final[1::2], 6).tolist()}")
    print()
    print(f"  surrogate D_tau final : {d_final:.12f}")
    print(f"  surrogate D_tau naive : {d_naive:.12f}")
    print(f"  surrogate D_tau best  : {d_best:.12f}  (frozen best-of-10)")
    print(f"  gain over naive       : {gain_over_naive:+.2%}")
    print(f"  best-of-10 gain       : {best_gain_over_naive:+.2%}")
    print(f"  shortfall vs best     : {shortfall:+.2%}")
    print()
    print(f"  ||final - frozen||_2  : {distance:.6f}")
    print(f"  max |coordinate diff| : {max_coordinate:.6f}")
    print()
    print("  Supplementary, real-CFD bank, same metric as the ablation headline:")
    print(f"    physical hard min naive : {physical_naive:.6f}")
    print(f"    physical hard min here  : {physical_hard_min:.6f}")
    print(f"    physical gain           : {physical_gain:+.2%}")

    trajectory = [iterate.tolist() for iterate in run["iterates"]]

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["iterate"]
            + [f"{axis}{i + 1}" for i in range(TARGET_COUNT) for axis in ("x", "y")]
        )
        for index, iterate in enumerate(trajectory):
            writer.writerow([index, *iterate])

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "purpose": (
                    "Replay of the single Ns=5 start that begins at the "
                    "conventional naive rake, recorded to compare its optimum "
                    "with the frozen best-of-ten design. Diagnostic only; no "
                    "frozen artifact is modified and no design is reselected."
                ),
                "source_campaign": "scripts/budget_ablation/run_budget_ablation.py",
                "n_sensors": TARGET_COUNT,
                "start": "naive",
                "start_layout": start.tolist(),
                "final_layout": final.tolist(),
                "frozen_best_layout": frozen_best.tolist(),
                "n_iterates": len(trajectory),
                "objective_evaluations": run["evaluations"],
                "converged": run["converged"],
                "termination_message": run["message"],
                "tau": tau,
                "lambda_separation": lambda_separation,
                "surrogate_D_final": d_final,
                "surrogate_D_naive_baseline": d_naive,
                "surrogate_D_frozen_best": d_best,
                "gain_over_naive_baseline": gain_over_naive,
                "frozen_best_gain_over_naive_baseline": best_gain_over_naive,
                "shortfall_versus_frozen_best": shortfall,
                "euclidean_distance_to_frozen_best": distance,
                "max_absolute_coordinate_difference": max_coordinate,
                "physical_hard_min_final": physical_hard_min,
                "physical_hard_min_naive": physical_naive,
                "physical_gain_over_naive": physical_gain,
                "physical_note": (
                    "Scored on the existing 66-angle real-CFD bank at the "
                    "campaign's common tau. No new CFD was run."
                ),
                "trajectory": trajectory,
                "wall_time_s": elapsed,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\n  wall time: {elapsed / 60.0:.1f} min")
    print(f"\nWrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
