"""Integration tests for the ImmersaForward -> WakeObservation pipeline."""

import numpy as np
from immersa_tesseract_inference.pipeline import ForwardObservationPipeline


def test_forward_observation_pipeline() -> None:
    """Run a coarse CFD case and sample its wake at sparse sensors."""
    sensor_x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    sensor_y = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    sensor_times = np.array([0.05, 0.10], dtype=np.float64)

    with ForwardObservationPipeline() as pipeline:
        measurements = pipeline.run(
            angle_of_attack_deg=60.0,
            sensor_x=sensor_x,
            sensor_y=sensor_y,
            sensor_times=sensor_times,
            h=0.1,
            dt=0.005,
            tf=0.1,
            Re=200.0,
            snapshot_freq=20,
        )

    assert measurements.shape == (3, 2, 2)
    assert np.all(np.isfinite(measurements))
    assert np.linalg.norm(measurements) > 0.0
