"""Recover a hidden angle of attack using the Tesseract-native derivative.

This is the closing demonstration for the T1 derivative work. The angle-of-
attack sensitivity is not formed in this application; it is requested from the
ImmersaForward Tesseract and propagated through WakeObservation:

    alpha
      -> ImmersaForward.jacobian                    (Julia, central FD inside)
      -> d(ux, uy)/d(alpha)
      -> WakeObservation.jacobian_vector_product    (JAX)
      -> dm/dalpha
      -> damped Gauss-Newton
      -> alpha

The configuration matches the committed Ns2 identifiability study, so the
recovered trajectory is directly comparable to the existing finite-difference
result for the same start (results/identifiability/multistart/Ns2.csv).
"""

import csv
import time
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.inverse import (
    infer_angle_of_attack,
    measurement_sensitivity,
    measurement_sensitivity_tesseract,
)
from immersa_tesseract_inference.pipeline import ForwardObservationPipeline

# ============================================================
# Experiment configuration
# ============================================================

ALPHA_TRUE = 63.0

ALPHA_INITIAL = 55.0

# Same sparse nonuniform temporal schedule as the identifiability study.
SENSOR_TIMES = np.array(
    [12.0, 13.3, 15.1, 17.4, 20.0],
    dtype=np.float64,
)

# Baseline two-sensor layout.
SENSOR_X = np.array([1.0, 1.0], dtype=np.float64)
SENSOR_Y = np.array([-0.4, 0.4], dtype=np.float64)

# Developed-wake CFD settings.
H = 0.05
DT = 0.0025
TF = 20.0
RE = 200.0
SNAPSHOT_FREQ = 40

# Gauss-Newton settings, matching the committed multistart study.
EPSILON_DEG = 0.5
MAX_STEP_DEG = 10.0
MAX_ITERATIONS = 20
DAMPING = 1.0e-12

OUTPUT_DIR = Path("results/tesseract_gradient_inference")
OUTPUT_CSV = OUTPUT_DIR / "t1_t2_gradient_inference.csv"
SENSITIVITY_CSV = OUTPUT_DIR / "sensitivity_comparison.csv"


# ============================================================
# Sensitivity cross-check
# ============================================================


def compare_sensitivities(
    pipeline: ForwardObservationPipeline,
) -> dict[str, float]:
    """Compare the app-level FD sensitivity against the Tesseract-native one."""
    common = {
        "h": H,
        "dt": DT,
        "tf": TF,
        "Re": RE,
        "snapshot_freq": SNAPSHOT_FREQ,
    }

    old = measurement_sensitivity(
        pipeline,
        ALPHA_INITIAL,
        SENSOR_X,
        SENSOR_Y,
        SENSOR_TIMES,
        epsilon_deg=EPSILON_DEG,
        **common,
    )

    new = measurement_sensitivity_tesseract(
        pipeline,
        ALPHA_INITIAL,
        SENSOR_X,
        SENSOR_Y,
        SENSOR_TIMES,
        **common,
    )

    norm_old = float(np.linalg.norm(old))
    norm_new = float(np.linalg.norm(new))

    difference = float(np.linalg.norm(new - old))

    return {
        "angle_of_attack_deg": ALPHA_INITIAL,
        "norm_finite_difference": norm_old,
        "norm_tesseract": norm_new,
        "absolute_difference": difference,
        "relative_difference": difference / norm_old,
    }


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Generate observations, run the inference, and store the artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    print("=" * 76, flush=True)
    print("Tesseract-native T1 -> T2 gradient-based AoA inference", flush=True)
    print("=" * 76, flush=True)
    print(f"truth AoA        = {ALPHA_TRUE:.1f} deg", flush=True)
    print(f"initial AoA      = {ALPHA_INITIAL:.1f} deg", flush=True)
    print(f"sensor x         = {SENSOR_X.tolist()}", flush=True)
    print(f"sensor y         = {SENSOR_Y.tolist()}", flush=True)
    print(f"sensor times     = {SENSOR_TIMES.tolist()}", flush=True)
    print(f"h, dt, tf        = {H}, {DT}, {TF}", flush=True)
    print("sensitivity      = ImmersaForward.jacobian -> WakeObservation JVP")
    print(flush=True)

    with ForwardObservationPipeline() as pipeline:
        # --------------------------------------------------------
        # Synthetic observations from the hidden truth.
        # --------------------------------------------------------

        print("Generating observations at the hidden truth...", flush=True)

        observations = pipeline.run(
            angle_of_attack_deg=ALPHA_TRUE,
            sensor_x=SENSOR_X,
            sensor_y=SENSOR_Y,
            sensor_times=SENSOR_TIMES,
            h=H,
            dt=DT,
            tf=TF,
            Re=RE,
            snapshot_freq=SNAPSHOT_FREQ,
        )

        print(f"Observation tensor shape: {observations.shape}", flush=True)
        print(flush=True)

        # --------------------------------------------------------
        # Sanity check: the two sensitivity routes must agree.
        # --------------------------------------------------------

        print("Comparing sensitivity routes at the initial angle...", flush=True)

        comparison = compare_sensitivities(pipeline)

        print(
            f"  ||dm/dalpha|| finite difference : "
            f"{comparison['norm_finite_difference']:.8e}",
            flush=True,
        )
        print(
            f"  ||dm/dalpha|| tesseract         : {comparison['norm_tesseract']:.8e}",
            flush=True,
        )
        print(
            f"  absolute difference             : "
            f"{comparison['absolute_difference']:.8e}",
            flush=True,
        )
        print(
            f"  relative difference             : "
            f"{comparison['relative_difference']:.8e}",
            flush=True,
        )
        print(flush=True)

        # --------------------------------------------------------
        # Gradient-based inference.
        # --------------------------------------------------------

        result = infer_angle_of_attack(
            pipeline,
            observations,
            SENSOR_X,
            SENSOR_Y,
            SENSOR_TIMES,
            initial_angle_deg=ALPHA_INITIAL,
            sensitivity_backend="tesseract",
            max_step_deg=MAX_STEP_DEG,
            max_iterations=MAX_ITERATIONS,
            damping=DAMPING,
            h=H,
            dt=DT,
            tf=TF,
            Re=RE,
            snapshot_freq=SNAPSHOT_FREQ,
            verbose=True,
        )

    elapsed = (time.perf_counter() - started) / 60.0

    # ------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "iteration",
                "alpha_deg",
                "loss",
                "gradient",
                "step_deg",
                "damping",
                "accepted",
            ]
        )

        for entry in result.history:
            writer.writerow(
                [
                    int(entry["iteration"]),
                    entry["angle_of_attack_deg"],
                    entry["objective"],
                    entry["gradient"],
                    entry["step_deg"],
                    DAMPING,
                    entry["step_deg"] != 0.0,
                ]
            )

        # Append the final state only when the solver stopped by exhausting its
        # iteration budget. On an early convergence break the last history row
        # already holds the recovered angle, and repeating it would put a
        # spurious flat segment at the end of the trajectory plot.
        last = result.history[-1]

        if last["angle_of_attack_deg"] != result.angle_of_attack_deg:
            writer.writerow(
                [
                    len(result.history),
                    result.angle_of_attack_deg,
                    result.objective,
                    "",
                    0.0,
                    DAMPING,
                    "",
                ]
            )

    with SENSITIVITY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison))
        writer.writeheader()
        writer.writerow(comparison)

    print("\n" + "=" * 76, flush=True)
    print("Result", flush=True)
    print("=" * 76, flush=True)
    print(f"truth AoA        = {ALPHA_TRUE:.8f} deg", flush=True)
    print(f"initial AoA      = {ALPHA_INITIAL:.8f} deg", flush=True)
    print(f"recovered AoA    = {result.angle_of_attack_deg:.8f} deg", flush=True)
    print(
        f"absolute error   = {abs(result.angle_of_attack_deg - ALPHA_TRUE):.8e} deg",
        flush=True,
    )
    print(f"initial loss     = {result.history[0]['objective']:.8e}", flush=True)
    print(f"final loss       = {result.objective:.8e}", flush=True)
    print(f"iterations       = {result.iterations}", flush=True)
    print(f"converged        = {result.converged}", flush=True)
    print(f"wall time        = {elapsed:.2f} min", flush=True)
    print(flush=True)
    print(f"Wrote {OUTPUT_CSV}", flush=True)
    print(f"Wrote {SENSITIVITY_CSV}", flush=True)


if __name__ == "__main__":
    main()
