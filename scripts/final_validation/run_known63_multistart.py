"""The practical headline: 10-start physical inverse recovery at truth 63 deg.

Mirrors the committed Ns=2 baseline experiment exactly -- same ten initial
angles, same objective, same tolerances, damping, step cap, iteration limit,
production CFD settings and success criterion -- changing only the sensor
layout. The committed baseline result is 7/10 and is not rerun.

The angle-of-attack sensitivity comes from the Tesseract-native route,
ImmersaForward.jacobian composed with the WakeObservation JVP. That path was
shown to agree with the original app-level finite difference to 3.55e-07
relative, recovering 63.00001808 against 63.00001806 in the same five
iterations, so the comparison stays apples-to-apples.
"""

import argparse
import csv
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.inverse import infer_angle_of_attack
from immersa_tesseract_inference.pipeline import ForwardObservationPipeline

# ============================================================
# Exactly the committed experiment configuration
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

SENSITIVITY_BACKEND = "tesseract"

MAX_WORKERS = 10

KNOWN_MINIMA = {
    "low_basin_27": 27.0,
    "mid_basin_42": 42.0,
    "mid_basin_51": 51.0,
    "true_basin": 63.0,
    "high_basin_83": 83.0,
}

LAYOUT_FILES = {
    "s_star_surrogate_v2": Path(
        "results/sensor_design/refined_design/surrogate_v2/s_star_surrogate_v2.json"
    ),
    "s_star_cfd_refined": Path(
        "results/sensor_design/refined_design/physical_refinement/"
        "s_star_cfd_refined.json"
    ),
}

OUTPUT_DIR = Path("results/sensor_design/final_physical_validation")
LOG_DIR = OUTPUT_DIR / "multistart_logs"

COMMITTED_BASELINE = Path("results/identifiability/multistart/Ns2.csv")


def classify_basin(alpha: float) -> str:
    """Assign a recovered angle to the nearest known landscape minimum."""
    return min(KNOWN_MINIMA, key=lambda name: abs(alpha - KNOWN_MINIMA[name]))


def run_one(task: tuple) -> dict:
    """One independent Gauss-Newton inversion in its own process."""
    layout_name, layout, initial_angle = task

    sensor_x = np.array([layout[0], layout[2]], dtype=np.float64)
    sensor_y = np.array([layout[1], layout[3]], dtype=np.float64)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tag = str(initial_angle).replace(".", "p")
    log_path = LOG_DIR / f"{layout_name}_initial_{tag}_deg.log"

    started = time.perf_counter()

    with log_path.open("w") as handle, redirect_stdout(handle), redirect_stderr(handle):
        with ForwardObservationPipeline() as pipeline:
            observations = pipeline.run(
                angle_of_attack_deg=ALPHA_TRUE,
                sensor_x=sensor_x,
                sensor_y=sensor_y,
                sensor_times=SENSOR_TIMES,
                h=H,
                dt=DT,
                tf=TF,
                Re=RE,
                snapshot_freq=SNAPSHOT_FREQ,
            )

            result = infer_angle_of_attack(
                pipeline,
                observations,
                sensor_x,
                sensor_y,
                SENSOR_TIMES,
                initial_angle_deg=initial_angle,
                sensitivity_backend=SENSITIVITY_BACKEND,
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

    elapsed = time.perf_counter() - started

    recovered = float(result.angle_of_attack_deg)
    basin = classify_basin(recovered)

    return {
        "layout": layout_name,
        "initial_angle_deg": initial_angle,
        "recovered_angle_deg": recovered,
        "signed_error_deg": recovered - ALPHA_TRUE,
        "absolute_error_deg": abs(recovered - ALPHA_TRUE),
        "basin": basin,
        "success": basin == "true_basin",
        "iterations": result.iterations,
        "final_objective": result.objective,
        "converged": result.converged,
        "wall_time_min": elapsed / 60.0,
    }


def main() -> None:
    """Run the ten committed starts for both optimized layouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a single representative trajectory and stop.",
    )
    arguments = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    layouts = {
        name: json.loads(path.read_text())["layout_vector"]
        for name, path in LAYOUT_FILES.items()
    }

    print("=" * 78)
    print(f"Known-63 physical multistart recovery (truth {ALPHA_TRUE} deg)")
    print("=" * 78)
    print(f"initial angles : {INITIAL_ANGLES}")
    print(f"gradient route : {SENSITIVITY_BACKEND}")
    for name, layout in layouts.items():
        print(f"  {name:22s} {[round(v, 6) for v in layout]}")
    print(flush=True)

    if arguments.benchmark:
        started = time.perf_counter()
        record = run_one(("s_star_surrogate_v2", layouts["s_star_surrogate_v2"], 55.0))
        elapsed = time.perf_counter() - started

        total_serial = elapsed * len(INITIAL_ANGLES) * len(layouts)

        print(
            f"one trajectory     : {elapsed / 60.0:.2f} min "
            f"({record['iterations']} iterations, "
            f"recovered {record['recovered_angle_deg']:.6f})"
        )
        print(
            f"projected serial   : {total_serial / 3600.0:.2f} h "
            f"for {len(INITIAL_ANGLES) * len(layouts)} trajectories"
        )
        print(
            f"projected {MAX_WORKERS}-wide : "
            f"{total_serial / MAX_WORKERS / 60.0:.0f} min"
        )
        return

    tasks = [
        (name, layout, angle)
        for name, layout in layouts.items()
        for angle in INITIAL_ANGLES
    ]

    started = time.perf_counter()

    records = []

    context = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=context) as executor:
        futures = {executor.submit(run_one, task): task for task in tasks}

        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"  {record['layout']:22s} start {record['initial_angle_deg']:5.1f} "
                f"-> {record['recovered_angle_deg']:10.6f}  "
                f"{'SUCCESS' if record['success'] else 'fail   '}  "
                f"({record['basin']}, {record['iterations']} it, "
                f"{record['wall_time_min']:.1f} min)",
                flush=True,
            )

    elapsed = time.perf_counter() - started

    records.sort(key=lambda r: (r["layout"], r["initial_angle_deg"]))

    with (OUTPUT_DIR / "known63_multistart.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    # Committed baseline, for reference only; not rerun.
    baseline_rows = list(csv.DictReader(COMMITTED_BASELINE.open()))
    baseline_success = sum(1 for r in baseline_rows if r["basin"] == "true_basin")

    summary = {
        "alpha_true_deg": ALPHA_TRUE,
        "initial_angles_deg": INITIAL_ANGLES,
        "gradient_route": SENSITIVITY_BACKEND,
        "baseline": {
            "source": str(COMMITTED_BASELINE),
            "rerun": False,
            "successes": baseline_success,
            "total": len(baseline_rows),
        },
        "layouts": {},
        "wall_time_s": elapsed,
    }

    print()
    print("=" * 78)
    print("Recovery summary")
    print("=" * 78)
    print(f"  baseline (committed) : {baseline_success}/{len(baseline_rows)}")

    for name in layouts:
        rows = [r for r in records if r["layout"] == name]
        successes = sum(1 for r in rows if r["success"])

        basins: dict[str, int] = {}
        for r in rows:
            basins[r["basin"]] = basins.get(r["basin"], 0) + 1

        summary["layouts"][name] = {
            "layout": layouts[name],
            "successes": successes,
            "total": len(rows),
            "basins": basins,
            "median_iterations": float(np.median([r["iterations"] for r in rows])),
            "max_absolute_error_deg": max(
                r["absolute_error_deg"] for r in rows if r["success"]
            )
            if successes
            else None,
        }

        print(f"  {name:22s} : {successes}/{len(rows)}   basins {basins}")

    (OUTPUT_DIR / "known63_multistart_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(f"\n  wall time: {elapsed / 60.0:.1f} min")
    print(f"\nWrote {OUTPUT_DIR / 'known63_multistart.csv'}")


if __name__ == "__main__":
    main()
