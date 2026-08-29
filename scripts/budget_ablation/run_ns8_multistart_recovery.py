"""Matched Ns=8 multistart inverse recovery: conventional rake vs optimized array.

The budget ablation showed that the frozen optimized eight-probe layout has far
higher real-CFD discrimination than the conventional aligned rake of the same
size (per-scalar hard minimum 0.056602 -> 0.125480). This script asks whether
that discrimination advantage actually shows up in the inverse problem.

The only independent variable is sensor placement. Everything else reproduces
the committed identifiability study
(scripts/identifiability/run_5time_nonuniform_multistart_Ns{1,2,3,5}.py)
exactly: same truth, same ten initial angles, same damped Gauss-Newton solver,
same epsilon, step cap, iteration limit, tolerances, bounds, backtracking,
production CFD settings, and the same nearest-known-minimum success rule. The
sensitivity backend is the committed default, the application-level central
difference on the composed T1 -> T2 map, so the numbers stay directly
comparable to the committed 8/10, 7/10, 7/10, 7/10 family.

Candidate angles are solved live by ImmersaForward, exactly as before; no bank
is substituted for the forward model. The 63 degree truth field is solved once,
as in the committed study, and is additionally cross-checked against the frozen
bank entry to confirm the two are the same field.

No layout is optimized, refined or modified here.
"""

import csv
import io
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.cfd_bank import bank_path
from immersa_tesseract_inference.inverse import infer_angle_of_attack
from immersa_tesseract_inference.pipeline import ForwardObservationPipeline

# ============================================================
# Exactly the committed identifiability configuration
# ============================================================

ALPHA_TRUE = 63.0

INITIAL_ANGLES = [35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]

SENSOR_TIMES = np.array([12.0, 13.3, 15.1, 17.4, 20.0], dtype=np.float64)

H = 0.05
DT = 0.0025
TF = 20.0
RE = 200.0
SNAPSHOT_FREQ = 40

EPSILON_DEG = 0.5
MAX_STEP_DEG = 10.0
MAX_ITERATIONS = 20

MAX_CACHED_FLOWS = 32
MAX_WORKERS = 10

# Committed success rule: the recovered angle is assigned to the nearest known
# landscape minimum, and only the true basin counts as a success.
KNOWN_MINIMA = {
    "low_basin_27": 27.0,
    "mid_basin_42": 42.0,
    "mid_basin_51": 51.0,
    "true_basin": 63.0,
    "high_basin_83": 83.0,
}

RESULTS_DIR = Path("results/sensor_budget_ablation")

SUMMARY_FILE = RESULTS_DIR / "budget_ablation_summary.json"

OUTPUT_CSV = RESULTS_DIR / "ns8_multistart_recovery.csv"
OUTPUT_JSON = RESULTS_DIR / "ns8_multistart_summary.json"

# Frozen Ns=8 layouts, read from the budget-ablation artifact.
LAYOUT_KEYS = {
    "conventional": "Ns8_naive",
    "optimized": "Ns8_optimized",
}


def classify_basin(alpha: float) -> str:
    """Assign a recovered angle to the nearest known landscape minimum."""
    return min(KNOWN_MINIMA, key=lambda name: abs(alpha - KNOWN_MINIMA[name]))


def unpack(layout: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Split a flat [x1, y1, x2, y2, ...] layout into sensor_x and sensor_y."""
    flat = np.asarray(layout, dtype=np.float64)
    return flat[0::2].copy(), flat[1::2].copy()


def termination_reason(log_text: str) -> str:
    """Last convergence or stopping line the solver printed."""
    for line in reversed(log_text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(("Converged:", "Stopping:")):
            return stripped.rstrip(".")
    return "iteration limit reached"


def run_one(task: tuple) -> dict:
    """One damped Gauss-Newton inversion in its own process."""
    layout_name, sensor_x, sensor_y, observations, initial_angle = task

    sensor_x = np.asarray(sensor_x, dtype=np.float64)
    sensor_y = np.asarray(sensor_y, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)

    log = io.StringIO()

    started = time.perf_counter()

    with redirect_stdout(log), redirect_stderr(log):
        with ForwardObservationPipeline(max_cached_flows=MAX_CACHED_FLOWS) as pipeline:
            result = infer_angle_of_attack(
                pipeline,
                observations,
                sensor_x,
                sensor_y,
                SENSOR_TIMES,
                initial_angle_deg=initial_angle,
                epsilon_deg=EPSILON_DEG,
                max_step_deg=MAX_STEP_DEG,
                max_iterations=MAX_ITERATIONS,
                h=H,
                dt=DT,
                tf=TF,
                Re=RE,
                snapshot_freq=SNAPSHOT_FREQ,
                verbose=True,
            )

            cache = pipeline.forward_cache_info()

    elapsed = time.perf_counter() - started

    recovered = float(result.angle_of_attack_deg)
    basin = classify_basin(recovered)

    # alpha at the start of every iteration, then the final accepted angle.
    trajectory = [float(entry["angle_of_attack_deg"]) for entry in result.history]
    if not trajectory or trajectory[-1] != recovered:
        trajectory.append(recovered)

    # Cache misses are genuine ImmersaForward container solves; hits are
    # forward evaluations served from the pipeline's flow cache. Their sum is
    # the number of forward-model evaluations the solver asked for.
    forward_evaluations = cache["hits"] + cache["misses"]

    return {
        "layout": layout_name,
        "initial_angle_deg": initial_angle,
        "recovered_angle_deg": recovered,
        "signed_error_deg": recovered - ALPHA_TRUE,
        "absolute_error_deg": abs(recovered - ALPHA_TRUE),
        "basin": basin,
        "success": basin == "true_basin",
        "iterations": result.iterations,
        "forward_evaluations": forward_evaluations,
        "t1_cfd_solves": cache["misses"],
        "cache_hits": cache["hits"],
        "final_objective": float(result.objective),
        "converged": bool(result.converged),
        "termination_reason": termination_reason(log.getvalue()),
        "wall_time_min": elapsed / 60.0,
        "trajectory_deg": trajectory,
    }


def main() -> None:
    """Run the twenty matched trajectories and summarize them."""
    summary_source = json.loads(SUMMARY_FILE.read_text())

    layouts = {
        name: summary_source["designs"][key] for name, key in LAYOUT_KEYS.items()
    }

    for name, layout in layouts.items():
        if len(layout) != 16:
            raise ValueError(f"{name} layout is not an Ns=8 layout: {len(layout) // 2}")

    print("=" * 78)
    print(f"Matched Ns=8 multistart recovery (truth {ALPHA_TRUE} deg)")
    print("=" * 78)
    print(f"initial angles  : {INITIAL_ANGLES}")
    print("gradient route  : finite_difference (committed default)")
    print(f"layout source   : {SUMMARY_FILE}")
    for name, layout in layouts.items():
        sensor_x, sensor_y = unpack(layout)
        print(f"  {name:13s} x = {np.round(sensor_x, 6).tolist()}")
        print(f"  {'':13s} y = {np.round(sensor_y, 6).tolist()}")
    print(flush=True)

    total_started = time.perf_counter()

    # ------------------------------------------------------------
    # Truth observations, computed once exactly as in the committed study,
    # then cross-checked against the frozen bank entry for the same angle.
    # ------------------------------------------------------------

    print("Solving the 63 deg truth field (one ImmersaForward solve)...", flush=True)

    truth_started = time.perf_counter()

    observations: dict[str, np.ndarray] = {}

    with ForwardObservationPipeline(max_cached_flows=1) as pipeline:
        truth_flow = pipeline.run_forward(
            angle_of_attack_deg=ALPHA_TRUE,
            h=H,
            dt=DT,
            tf=TF,
            Re=RE,
            snapshot_freq=SNAPSHOT_FREQ,
        )

        for name, layout in layouts.items():
            sensor_x, sensor_y = unpack(layout)
            observations[name] = pipeline.observe(
                truth_flow, sensor_x, sensor_y, SENSOR_TIMES
            )

        # Same field, from the frozen bank: a consistency check only. The live
        # solve above is what the experiment uses.
        bank_observations = {}
        with np.load(bank_path(ALPHA_TRUE)) as data:
            bank_flow = {
                key: data[key] for key in ("ux", "uy", "ux_x", "ux_y", "uy_x", "uy_y")
            }
            bank_flow["times"] = data["times"]

        for name, layout in layouts.items():
            sensor_x, sensor_y = unpack(layout)
            bank_observations[name] = pipeline.observe(
                bank_flow, sensor_x, sensor_y, SENSOR_TIMES
            )

    truth_elapsed = time.perf_counter() - truth_started

    bank_agreement = {
        name: float(np.max(np.abs(observations[name] - bank_observations[name])))
        for name in layouts
    }

    print(f"Truth observations complete ({truth_elapsed / 60.0:.2f} min).")
    for name, difference in bank_agreement.items():
        print(f"  live vs frozen bank, {name:13s}: max |difference| = {difference:.3e}")
    print(flush=True)

    # ------------------------------------------------------------
    # Twenty matched trajectories
    # ------------------------------------------------------------

    tasks = []
    for name, layout in layouts.items():
        sensor_x, sensor_y = unpack(layout)
        for angle in INITIAL_ANGLES:
            tasks.append((name, sensor_x, sensor_y, observations[name], angle))

    print(f"Launching {len(tasks)} trajectories on {MAX_WORKERS} workers...\n")

    records = []

    context = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=context) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]

        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"  {record['layout']:13s} start {record['initial_angle_deg']:5.1f} "
                f"-> {record['recovered_angle_deg']:10.6f}  "
                f"{'SUCCESS' if record['success'] else 'fail   '}  "
                f"({record['basin']}, {record['iterations']} it, "
                f"{record['t1_cfd_solves']} solves, "
                f"{record['wall_time_min']:.1f} min)",
                flush=True,
            )

    elapsed = time.perf_counter() - total_started

    records.sort(key=lambda r: (r["layout"], r["initial_angle_deg"]))

    # ------------------------------------------------------------
    # Numerical outputs
    # ------------------------------------------------------------

    csv_fields = [key for key in records[0] if key != "trajectory_deg"]

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in csv_fields})

    by_layout = {
        name: {r["initial_angle_deg"]: r for r in records if r["layout"] == name}
        for name in layouts
    }

    common_success = sorted(
        angle
        for angle in INITIAL_ANGLES
        if all(by_layout[name][angle]["success"] for name in layouts)
    )

    def statistics(name: str) -> dict:
        """Aggregate one layout over all starts and over the shared subset."""
        rows = [by_layout[name][angle] for angle in INITIAL_ANGLES]
        shared = [by_layout[name][angle] for angle in common_success]

        basins: dict[str, int] = {}
        for row in rows:
            basins[row["basin"]] = basins.get(row["basin"], 0) + 1

        return {
            "layout": layouts[name],
            "successes": sum(1 for r in rows if r["success"]),
            "total": len(rows),
            "basins": basins,
            "successful_starts_deg": [
                r["initial_angle_deg"] for r in rows if r["success"]
            ],
            "failed_starts_deg": [
                r["initial_angle_deg"] for r in rows if not r["success"]
            ],
            "max_absolute_error_deg": (
                max(r["absolute_error_deg"] for r in rows if r["success"])
                if any(r["success"] for r in rows)
                else None
            ),
            "common_success_subset": {
                "iterations": [r["iterations"] for r in shared],
                "median_iterations": float(np.median([r["iterations"] for r in shared]))
                if shared
                else None,
                "mean_iterations": float(np.mean([r["iterations"] for r in shared]))
                if shared
                else None,
                "forward_evaluations": [r["forward_evaluations"] for r in shared],
                "median_forward_evaluations": float(
                    np.median([r["forward_evaluations"] for r in shared])
                )
                if shared
                else None,
                "t1_cfd_solves": [r["t1_cfd_solves"] for r in shared],
                "median_t1_cfd_solves": float(
                    np.median([r["t1_cfd_solves"] for r in shared])
                )
                if shared
                else None,
            },
        }

    aggregate = {name: statistics(name) for name in layouts}

    conventional_success = {
        angle for angle in INITIAL_ANGLES if by_layout["conventional"][angle]["success"]
    }
    optimized_success = {
        angle for angle in INITIAL_ANGLES if by_layout["optimized"][angle]["success"]
    }

    # ------------------------------------------------------------
    # Preregistered same-start trajectory rule
    # ------------------------------------------------------------

    if common_success:
        selected_start = min(common_success, key=lambda a: (abs(a - 55.0), a))
        selection_rule = (
            "alpha_0 = 55 deg if it succeeds for both layouts; otherwise the "
            "common-success start with minimum absolute distance to 55 deg."
        )
    else:
        selected_start = None
        selection_rule = "No start succeeded for both layouts; no trajectory selected."

    report = {
        "experiment": "matched Ns=8 multistart inverse recovery",
        "question": (
            "Does the frozen optimized Ns=8 placement translate its higher "
            "real-CFD discrimination into a larger recovery basin and/or fewer "
            "inverse iterations, at an identical sensing budget?"
        ),
        "independent_variable": "sensor placement only",
        "alpha_true_deg": ALPHA_TRUE,
        "initial_angles_deg": INITIAL_ANGLES,
        "methodology_source": (
            "damped Gauss-Newton multistart methodology following the "
            "committed 5-time nonuniform identifiability implementation in "
            "scripts/identifiability/run_5time_nonuniform_multistart_Ns2.py"
        ),
        "layout_source": str(SUMMARY_FILE),
        "solver": {
            "name": "damped Gauss-Newton with backtracking line search",
            "entrypoint": "immersa_tesseract_inference.inverse.infer_angle_of_attack",
            "sensitivity_backend": "finite_difference",
            "epsilon_deg": EPSILON_DEG,
            "max_step_deg": MAX_STEP_DEG,
            "max_iterations": MAX_ITERATIONS,
            "damping": 1.0e-12,
            "max_backtracks": 6,
            "objective_tolerance": 1.0e-10,
            "gradient_tolerance": 1.0e-8,
            "step_tolerance_deg": 1.0e-3,
            "angle_bounds_deg": [0.0, 90.0],
            "iteration_definition": "len(history); one entry per Gauss-Newton step",
        },
        "cfd": {
            "h": H,
            "dt": DT,
            "tf": TF,
            "Re": RE,
            "snapshot_freq": SNAPSHOT_FREQ,
            "observation_times": SENSOR_TIMES.tolist(),
            "candidate_angles": "solved live by ImmersaForward, as in the committed study",
        },
        "success_criterion": {
            "rule": "nearest known landscape minimum must be the true basin",
            "known_minima_deg": KNOWN_MINIMA,
        },
        "truth_field": {
            "solved_live": True,
            "solve_minutes": truth_elapsed / 60.0,
            "frozen_bank_entry": str(bank_path(ALPHA_TRUE)),
            "live_vs_bank_max_absolute_difference": bank_agreement,
        },
        "aggregate": aggregate,
        "common_success_starts_deg": common_success,
        "rescued_by_optimization_deg": sorted(optimized_success - conventional_success),
        "lost_to_optimization_deg": sorted(conventional_success - optimized_success),
        "failed_for_both_deg": sorted(
            set(INITIAL_ANGLES) - conventional_success - optimized_success
        ),
        "same_start_trajectory": {
            "selection_rule": selection_rule,
            "selected_initial_angle_deg": selected_start,
            "trajectories": (
                {
                    name: by_layout[name][selected_start]["trajectory_deg"]
                    for name in layouts
                }
                if selected_start is not None
                else None
            ),
            "iterations": (
                {
                    name: by_layout[name][selected_start]["iterations"]
                    for name in layouts
                }
                if selected_start is not None
                else None
            ),
        },
        "trajectories_deg": {
            name: {
                str(angle): by_layout[name][angle]["trajectory_deg"]
                for angle in INITIAL_ANGLES
            }
            for name in layouts
        },
        "wall_time_s": elapsed,
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("Recovery summary")
    print("=" * 78)
    for name in layouts:
        info = aggregate[name]
        print(
            f"  {name:13s} : {info['successes']}/{info['total']}   "
            f"basins {info['basins']}"
        )

    print(f"\n  common-success starts : {common_success}")
    print(f"  rescued by optimization: {report['rescued_by_optimization_deg']}")
    print(f"  lost to optimization   : {report['lost_to_optimization_deg']}")
    print(f"  failed for both        : {report['failed_for_both_deg']}")

    print()
    print("=" * 78)
    print("Convergence effort on the common-success subset")
    print("=" * 78)
    for name in layouts:
        subset = aggregate[name]["common_success_subset"]
        print(
            f"  {name:13s} : iterations {subset['iterations']}  "
            f"median {subset['median_iterations']}  "
            f"mean {subset['mean_iterations']:.2f}"
            if subset["iterations"]
            else f"  {name:13s} : no common successes"
        )
        print(
            f"  {'':13s}   forward evals {subset['forward_evaluations']}  "
            f"median {subset['median_forward_evaluations']}"
        )
        print(
            f"  {'':13s}   T1 solves     {subset['t1_cfd_solves']}  "
            f"median {subset['median_t1_cfd_solves']}"
        )

    if selected_start is not None:
        print(f"\n  same-start trajectory at alpha_0 = {selected_start} deg")
        for name in layouts:
            print(
                f"    {name:13s} : {by_layout[name][selected_start]['iterations']} "
                f"iterations"
            )

    print(f"\n  wall time: {elapsed / 60.0:.1f} min")
    print(f"\nWrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
