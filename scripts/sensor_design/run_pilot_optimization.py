"""Pilot run of the differentiable sensor-array design loop.

Two starts only. The point is to prove the machinery end to end -- the gradient
is consumed correctly, the sensors move, the box bounds hold, and the
separation constraint behaves -- not to produce the final scientific layout.

The gradient driving the optimizer is assembled entirely from Tesseract
derivative endpoints:

    SensorArrayDesign.vector_jacobian_product
      -> WakeSurrogate.vector_jacobian_product (once per design AoA)
      -> dD/d[x1, y1, x2, y2]
"""

import csv
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
    separation_penalty,
)
from scipy.optimize import minimize

# ============================================================
# Configuration
# ============================================================

STARTS = {
    "A_baseline": BASELINE_LAYOUT.copy(),
    "B_asymmetric": np.array([1.2, -0.8, 2.8, 0.8], dtype=np.float64),
}

# Loose relative to machine precision on purpose: the objective inherits
# WakeSurrogate's float32 noise, so chasing tighter convergence just burns
# evaluations on numerical noise.
MAX_ITERATIONS = 200
F_TOLERANCE = 1.0e-9
G_TOLERANCE = 1.0e-6

OUTPUT_DIR = Path("results/sensor_design/optimization")
OUTPUT_CSV = OUTPUT_DIR / "pilot_optimization.csv"


def sensor_distance(design: np.ndarray) -> float:
    """Euclidean distance between the two sensors."""
    return float(np.hypot(design[0] - design[2], design[1] - design[3]))


def main() -> None:
    """Calibrate once, then run both pilot starts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("Pilot differentiable sensor-array design", flush=True)
    print("=" * 78, flush=True)
    print(
        f"design AoA grid : {ALPHA_GRID_DEG[0]:.1f} .. {ALPHA_GRID_DEG[-1]:.1f} deg, "
        f"{ALPHA_GRID_DEG.size} points, step "
        f"{ALPHA_GRID_DEG[1] - ALPHA_GRID_DEG[0]:.1f} deg",
        flush=True,
    )
    print(f"63 deg in grid  : {bool(np.any(ALPHA_GRID_DEG == 63.0))}", flush=True)
    print(f"delta_alpha_min : {DELTA_ALPHA_MIN_DEG} deg", flush=True)
    print(f"bounds          : {DESIGN_BOUNDS}", flush=True)
    print(f"min separation  : {MIN_SENSOR_DISTANCE}", flush=True)
    print(flush=True)

    rows = []

    with SensorDesignPipeline() as pipeline:
        # --------------------------------------------------------
        # Calibrate tau and lambda once, then freeze.
        # --------------------------------------------------------

        started = time.perf_counter()

        calibration = pipeline.calibrate()

        tau = calibration["tau"]
        lambda_separation = calibration["lambda_separation"]

        print("Calibration at the baseline layout:", flush=True)
        print(f"  retained pairs        : {calibration['n_pairs']}", flush=True)
        print(
            f"  min pair distance     : {calibration['min_distance']:.8f}",
            flush=True,
        )
        print(
            f"  10th pair distance    : {calibration['rank_distance']:.8f}",
            flush=True,
        )
        print(
            f"  median pair distance  : {calibration['median_distance']:.8f}",
            flush=True,
        )
        print(
            f"  superseded rank-10 tau: {calibration['legacy_tau']:.8f}  "
            f"(N_eff = {calibration['legacy_effective_pairs']:.2f})",
            flush=True,
        )
        print(
            f"  tau (frozen)          : {tau:.8f}  "
            f"(N_eff = {calibration['effective_pairs']:.4f})",
            flush=True,
        )
        print(
            f"  top-1 softmin weight  : {calibration['top1_weight']:.6f}",
            flush=True,
        )
        print(
            f"  top-10 cumulative wt  : {calibration['top10_weight']:.6f}",
            flush=True,
        )
        print(
            f"  lambda_separation     : {lambda_separation:.8f}",
            flush=True,
        )
        print(f"  calibration wall time : {time.perf_counter() - started:.2f} s")
        print(flush=True)

        # --------------------------------------------------------
        # Pilot starts
        # --------------------------------------------------------

        for name, start in STARTS.items():
            print("=" * 78, flush=True)
            print(f"Start {name}: {start.tolist()}", flush=True)
            print("=" * 78, flush=True)

            calls_before = (
                pipeline.n_surrogate_calls,
                pipeline.n_design_calls,
            )

            initial_measurements = pipeline.measurements(start)
            initial_scored = pipeline.discrimination(initial_measurements, tau)
            initial_penalty, _ = separation_penalty(
                start, lambda_separation=lambda_separation
            )

            initial_objective = -initial_scored["discrimination"] + initial_penalty

            evaluations = {"count": 0}

            def objective(design: np.ndarray) -> tuple[float, np.ndarray]:
                evaluations["count"] += 1
                return pipeline.objective_and_gradient(
                    design,
                    tau=tau,
                    lambda_separation=lambda_separation,
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

            final_measurements = pipeline.measurements(final)
            final_scored = pipeline.discrimination(final_measurements, tau)
            final_penalty, _ = separation_penalty(
                final, lambda_separation=lambda_separation
            )

            calls_after = (
                pipeline.n_surrogate_calls,
                pipeline.n_design_calls,
            )

            canonical = canonicalize_layout(final)

            print(f"  success            : {result.success} ({result.message})")
            print(f"  iterations         : {result.nit}")
            print(f"  obj/grad evals     : {evaluations['count']}")
            print(f"  T3 calls           : {calls_after[0] - calls_before[0]}")
            print(f"  T4 calls           : {calls_after[1] - calls_before[1]}")
            print()
            print(f"  initial layout     : {np.round(start, 6).tolist()}")
            print(f"  final layout       : {np.round(final, 6).tolist()}")
            print(f"  final (canonical)  : {np.round(canonical, 6).tolist()}")
            print()
            print(f"  initial D_tau      : {initial_scored['discrimination']:.8f}")
            print(f"  final   D_tau      : {final_scored['discrimination']:.8f}")
            print(f"  initial objective  : {initial_objective:.8f}")
            print(f"  final   objective  : {float(result.fun):.8f}")
            print()
            print(f"  initial min pair d : {initial_scored['min_pair_distance']:.8f}")
            print(f"  final   min pair d : {final_scored['min_pair_distance']:.8f}")
            print(f"  final separation r : {sensor_distance(final):.8f}")
            print(f"  penalty active     : {final_penalty > 0.0}")
            print(f"  wall time          : {wall_time / 60.0:.2f} min")
            print(
                f"  per objective+grad : "
                f"{wall_time / max(evaluations['count'], 1):.3f} s",
                flush=True,
            )
            print(flush=True)

            rows.append(
                {
                    "start": name,
                    "x1_initial": start[0],
                    "y1_initial": start[1],
                    "x2_initial": start[2],
                    "y2_initial": start[3],
                    "x1_final": canonical[0],
                    "y1_final": canonical[1],
                    "x2_final": canonical[2],
                    "y2_final": canonical[3],
                    "discrimination_initial": initial_scored["discrimination"],
                    "discrimination_final": final_scored["discrimination"],
                    "objective_initial": initial_objective,
                    "objective_final": float(result.fun),
                    "min_pair_distance_initial": initial_scored["min_pair_distance"],
                    "min_pair_distance_final": final_scored["min_pair_distance"],
                    "sensor_distance_final": sensor_distance(final),
                    "separation_penalty_active": final_penalty > 0.0,
                    "iterations": int(result.nit),
                    "evaluations": evaluations["count"],
                    "success": bool(result.success),
                    "tau": tau,
                    "effective_pairs": calibration["effective_pairs"],
                    "lambda_separation": lambda_separation,
                    "delta_alpha_min_deg": DELTA_ALPHA_MIN_DEG,
                    "wall_time_min": wall_time / 60.0,
                    "seconds_per_evaluation": wall_time
                    / max(evaluations["count"], 1),
                }
            )

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
