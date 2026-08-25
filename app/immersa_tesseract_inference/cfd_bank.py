"""Access to the persisted real-CFD validation bank.

The bank holds dense ImmersaForward wake fields at the five observation times,
one entry per angle of attack. Sensor coordinates never enter the CFD state, so
these fields can be observed at any sensor layout without new CFD.

Only the five observation-time snapshots are stored. WakeObservation brackets in
time with ``searchsorted(..., side="right") - 1`` clipped to ``n - 2``, so a
five-entry time grid reproduces every requested time exactly, the final endpoint
included.
"""

from pathlib import Path

import numpy as np

# Production configuration; must match the committed identifiability studies.
H = 0.05
DT = 0.0025
TF = 20.0
RE = 200.0
SNAPSHOT_FREQ = 40

OBSERVATION_TIMES = np.array([12.0, 13.3, 15.1, 17.4, 20.0], dtype=np.float64)

DESIGN_GRID_DEG = np.arange(20.0, 85.0 + 1.0e-9, 2.5, dtype=np.float64)
LANDSCAPE_GRID_DEG = np.arange(20.0, 85.0 + 1.0e-9, 1.0, dtype=np.float64)

BANK_DIR = Path("data/cfd_validation_bank")

FLOW_KEYS = ("ux", "uy", "ux_x", "ux_y", "uy_x", "uy_y")

BASELINE_LAYOUT = np.array([1.0, -0.4, 1.0, 0.4], dtype=np.float64)


def union_grid() -> np.ndarray:
    """Every angle of attack the bank covers."""
    return np.unique(np.concatenate([DESIGN_GRID_DEG, LANDSCAPE_GRID_DEG]).round(6))


def bank_path(alpha_deg: float, directory: Path = BANK_DIR) -> Path:
    """Deterministic filename for one angle of attack."""
    return directory / f"alpha_{alpha_deg:07.3f}.npz".replace(".", "p", 1)


def load_flow(alpha_deg: float, directory: Path = BANK_DIR) -> dict:
    """Reload one bank entry as a WakeObservation-ready flow mapping."""
    path = bank_path(alpha_deg, directory)

    if not path.exists():
        raise FileNotFoundError(f"No bank entry for alpha={alpha_deg}: {path}")

    with np.load(path) as data:
        flow = {key: data[key] for key in FLOW_KEYS}
        flow["times"] = data["times"]

    return flow


def observe_bank(
    observation: object,
    flow: dict,
    sensor_x: np.ndarray,
    sensor_y: np.ndarray,
    sensor_times: np.ndarray = OBSERVATION_TIMES,
) -> np.ndarray:
    """Apply WakeObservation to a bank flow at the given sensors."""
    outputs = observation.apply(
        {
            **{key: flow[key] for key in FLOW_KEYS},
            "times": flow["times"],
            "sensor_x": np.asarray(sensor_x, dtype=np.float64),
            "sensor_y": np.asarray(sensor_y, dtype=np.float64),
            "sensor_times": np.asarray(sensor_times, dtype=np.float64),
        }
    )

    return np.asarray(outputs["measurements"], dtype=np.float64)


def observation_sensor_jacobian(
    observation: object,
    flow: dict,
    sensor_x: np.ndarray,
    sensor_y: np.ndarray,
    sensor_times: np.ndarray = OBSERVATION_TIMES,
) -> dict[str, np.ndarray]:
    """d(measurements)/d(sensor coordinates) for a fixed CFD field.

    Sensors are passive, so this is the complete physical sensor-position
    derivative: the CFD state is held fixed and only the observation operator
    depends on where the probes sit.
    """
    outputs = observation.jacobian(
        {
            **{key: flow[key] for key in FLOW_KEYS},
            "times": flow["times"],
            "sensor_x": np.asarray(sensor_x, dtype=np.float64),
            "sensor_y": np.asarray(sensor_y, dtype=np.float64),
            "sensor_times": np.asarray(sensor_times, dtype=np.float64),
        },
        jac_inputs=["sensor_x", "sensor_y"],
        jac_outputs=["measurements"],
    )

    return {
        "sensor_x": np.asarray(outputs["measurements"]["sensor_x"], dtype=np.float64),
        "sensor_y": np.asarray(outputs["measurements"]["sensor_y"], dtype=np.float64),
    }
