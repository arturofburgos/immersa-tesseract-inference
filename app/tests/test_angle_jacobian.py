"""Equivalence of the app-level FD and the T1 Tesseract derivative.

WakeObservation interpolates the velocity fields linearly for fixed sensor
coordinates. Differencing therefore commutes with observing:

    FD(T2 . T1)  ==  T2 . FD(T1)

The left side is the already-validated path used by every committed
identifiability study. The right side is the new path, in which ImmersaForward
reports d(field)/d(alpha) through its own Jacobian endpoint and WakeObservation
pushes that tangent forward. Agreement shows the Tesseract-native architecture
reproduces the existing science rather than approximating it.

The two are not bit-identical, and the reason favors the new path: T2 runs in
float32, so the existing path differences two O(1) float32 measurements and
divides by 2 eps, amplifying round-off. The new path differences in float64
inside the Julia solver and applies the float32 interpolation only afterwards,
so it avoids that cancellation entirely.
"""

import numpy as np
from immersa_tesseract_inference.inverse import measurement_sensitivity
from immersa_tesseract_inference.pipeline import ForwardObservationPipeline

# Must match _AOA_EPSILON_DEG in the ImmersaForward Tesseract, otherwise the
# two paths are different finite differences and are not expected to agree.
EPSILON_DEG = 0.5

# Inexpensive configuration, matching the existing pipeline integration test.
ALPHA_DEG = 60.0
H = 0.1
DT = 0.005
TF = 0.1
RE = 200.0
SNAPSHOT_FREQ = 20


def test_t1_jacobian_matches_app_finite_difference() -> None:
    """T2(FD(T1)) must reproduce FD(T2(T1)) to floating-point precision."""
    sensor_x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    sensor_y = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    sensor_times = np.array([0.05, 0.10], dtype=np.float64)

    with ForwardObservationPipeline() as pipeline:
        # --------------------------------------------------------------
        # Existing path: finite-difference the composed T1 -> T2 map.
        # --------------------------------------------------------------

        existing = measurement_sensitivity(
            pipeline,
            ALPHA_DEG,
            sensor_x,
            sensor_y,
            sensor_times,
            epsilon_deg=EPSILON_DEG,
            h=H,
            dt=DT,
            tf=TF,
            Re=RE,
            snapshot_freq=SNAPSHOT_FREQ,
        )

        # --------------------------------------------------------------
        # New path: T1 reports d(field)/d(alpha), T2 pushes it forward.
        # --------------------------------------------------------------

        field_tangent = pipeline.forward_angle_jacobian(
            ALPHA_DEG,
            h=H,
            dt=DT,
            tf=TF,
            Re=RE,
            snapshot_freq=SNAPSHOT_FREQ,
        )

        # T2 is linear in the field, so its JVP does not depend on the point
        # it is linearized about; the unperturbed flow supplies the grids.
        flow = pipeline.run_forward(
            ALPHA_DEG,
            h=H,
            dt=DT,
            tf=TF,
            Re=RE,
            snapshot_freq=SNAPSHOT_FREQ,
        )

        composed = pipeline.observe_jvp(
            flow,
            field_tangent,
            sensor_x,
            sensor_y,
            sensor_times,
        )

        # Measurement scale, used below to size the float32 noise floor.
        # The alpha flow is already cached, so this costs no extra CFD solve.
        measurements = pipeline.observe(
            flow,
            sensor_x,
            sensor_y,
            sensor_times,
        )

    assert composed.shape == existing.shape
    assert np.all(np.isfinite(composed))

    # The sensitivity must be a real signal, not numerical noise.
    derivative_norm = float(np.linalg.norm(existing))
    assert derivative_norm > 0.0

    difference_norm = float(np.linalg.norm(composed - existing))
    relative_difference = difference_norm / derivative_norm

    # WakeObservation evaluates in float32 (JAX x64 is not enabled there).
    # The existing path therefore differences two O(||m||) float32 quantities
    # and divides by 2 eps, so its absolute noise floor is
    #
    #     eps_float32 * ||m|| / (2 eps).
    #
    # The new path forms the difference in float64 inside the Julia solver and
    # only then applies the float32 interpolation, so it never incurs that
    # cancellation. Agreement is therefore limited by the *existing* path's
    # precision, not by any error in the derivative -- which is why this is an
    # absolute bound rather than a relative one. The derivative norm here is
    # ~1e-3 of the measurement norm, so the same absolute agreement looks like
    # a much larger relative number.
    noise_floor = float(
        np.finfo(np.float32).eps * np.linalg.norm(measurements) / (2.0 * EPSILON_DEG)
    )

    print(f"\n||FD(T2(T1))||                 : {derivative_norm:.6e}")
    print(f"||T2(FD(T1)) - FD(T2(T1))||    : {difference_norm:.6e}")
    print(f"float32 cancellation floor     : {noise_floor:.6e}")
    print(f"difference / noise floor       : {difference_norm / noise_floor:.4f}")
    print(f"relative to derivative norm    : {relative_difference:.6e}")

    assert difference_norm < 5.0 * noise_floor

    # Guard against a grossly wrong derivative, which would be O(1) relative.
    assert relative_difference < 1.0e-3
