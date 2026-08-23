"""Composition of ImmersaForward and WakeObservation Tesseracts."""

from collections import OrderedDict
from contextlib import ExitStack
from typing import Any

import numpy as np
from tesseract_core import Tesseract

IMMERSA_FORWARD_IMAGE = "immersa_tesseract_inference_immersa_forward"
WAKE_OBSERVATION_IMAGE = "immersa_tesseract_inference_wake_observation"

ForwardCacheKey = tuple[
    float,
    float,
    float,
    float,
    float,
    int,
]


class ForwardObservationPipeline:
    """Compose the Immersa forward solver with the wake observation operator.

    The pipeline represents

        angle_of_attack
            -> ImmersaForward
            -> dense staggered wake velocity fields
            -> WakeObservation
            -> sparse sensor measurements.

    Both Tesseracts remain alive while the context manager is active.

    ImmersaForward results are cached using the physical/numerical simulation
    parameters. Sensor locations are intentionally not part of the cache key
    because changing sensors does not change the CFD solution.

    This is useful for inverse problems and sensor-design studies, where the
    same flow solution may be sampled repeatedly using different sensor arrays.
    """

    def __init__(
        self,
        forward_image: str = IMMERSA_FORWARD_IMAGE,
        observation_image: str = WAKE_OBSERVATION_IMAGE,
        max_cached_flows: int = 8,
    ) -> None:
        """Initialize the pipeline and bounded forward-solution cache."""
        if max_cached_flows < 0:
            raise ValueError("max_cached_flows must be non-negative.")

        self.forward_image = forward_image
        self.observation_image = observation_image
        self.max_cached_flows = max_cached_flows

        self._stack: ExitStack | None = None
        self._forward: Any = None
        self._observation: Any = None

        self._flow_cache: OrderedDict[
            ForwardCacheKey,
            dict[str, Any],
        ] = OrderedDict()

        self._cache_hits = 0
        self._cache_misses = 0

    def __enter__(self) -> "ForwardObservationPipeline":
        """Start both Tesseracts and initialize a fresh cache session."""
        self._flow_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

        self._stack = ExitStack()

        self._forward = self._stack.enter_context(
            Tesseract.from_image(self.forward_image)
        )

        self._observation = self._stack.enter_context(
            Tesseract.from_image(self.observation_image)
        )

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """Stop both Tesseracts and release cached flow fields."""
        self.clear_forward_cache()

        if self._stack is not None:
            self._stack.close()

        self._stack = None
        self._forward = None
        self._observation = None

    def run_forward(
        self,
        angle_of_attack_deg: float,
        *,
        h: float = 0.1,
        dt: float = 0.005,
        tf: float = 0.1,
        Re: float = 200.0,
        snapshot_freq: int = 20,
    ) -> dict[str, Any]:
        """Run ImmersaForward or reuse a cached CFD solution."""
        if self._forward is None:
            raise RuntimeError(
                "Pipeline is not active. Use "
                "'with ForwardObservationPipeline() as pipeline:'."
            )

        cache_key: ForwardCacheKey = (
            float(angle_of_attack_deg),
            float(h),
            float(dt),
            float(tf),
            float(Re),
            int(snapshot_freq),
        )

        # ------------------------------------------------------------
        # Cache hit: the CFD solution for this exact set of physical
        # and numerical parameters has already been computed.
        # ------------------------------------------------------------

        if cache_key in self._flow_cache:
            self._cache_hits += 1

            # Move the entry to the end so it is marked as recently used.
            flow = self._flow_cache.pop(cache_key)
            self._flow_cache[cache_key] = flow

            return flow

        # ------------------------------------------------------------
        # Cache miss: run ImmersaForward.
        # ------------------------------------------------------------

        self._cache_misses += 1

        inputs = {
            "angle_of_attack_deg": float(angle_of_attack_deg),
            "h": float(h),
            "dt": float(dt),
            "tf": float(tf),
            "Re": float(Re),
            "snapshot_freq": int(snapshot_freq),
        }

        flow = self._forward.apply(inputs)

        # ------------------------------------------------------------
        # Store the CFD result in a bounded least-recently-used cache.
        # ------------------------------------------------------------

        if self.max_cached_flows > 0:
            self._flow_cache[cache_key] = flow

            while len(self._flow_cache) > self.max_cached_flows:
                self._flow_cache.popitem(last=False)

        return flow

    def observe(
        self,
        flow: dict[str, Any],
        sensor_x: np.ndarray,
        sensor_y: np.ndarray,
        sensor_times: np.ndarray,
    ) -> np.ndarray:
        """Sample a dense Immersa wake at sparse sensor locations and times."""
        if self._observation is None:
            raise RuntimeError(
                "Pipeline is not active. Use "
                "'with ForwardObservationPipeline() as pipeline:'."
            )

        sensor_x = np.asarray(sensor_x, dtype=np.float64)
        sensor_y = np.asarray(sensor_y, dtype=np.float64)
        sensor_times = np.asarray(sensor_times, dtype=np.float64)

        self._validate_sensor_inputs(
            flow,
            sensor_x,
            sensor_y,
            sensor_times,
        )

        inputs = {
            "ux": flow["ux"],
            "uy": flow["uy"],
            "ux_x": flow["ux_x"],
            "ux_y": flow["ux_y"],
            "uy_x": flow["uy_x"],
            "uy_y": flow["uy_y"],
            "times": flow["times"],
            "sensor_x": sensor_x,
            "sensor_y": sensor_y,
            "sensor_times": sensor_times,
        }

        outputs = self._observation.apply(inputs)

        return np.asarray(
            outputs["measurements"],
            dtype=np.float64,
        )

    def forward_angle_jacobian(
        self,
        angle_of_attack_deg: float,
        *,
        h: float = 0.1,
        dt: float = 0.005,
        tf: float = 0.1,
        Re: float = 200.0,
        snapshot_freq: int = 20,
    ) -> dict[str, np.ndarray]:
        """Request d(field)/d(alpha) from the ImmersaForward Tesseract itself.

        This uses T1's own Jacobian endpoint, which forms the central finite
        difference internally using the Julia solver. It therefore costs two
        CFD solves that are *not* visible to this pipeline's flow cache; the
        cached alpha +/- epsilon path in
        ``immersa_tesseract_inference.inverse.measurement_sensitivity``
        remains available and is still cheaper across a line search.
        """
        if self._forward is None:
            raise RuntimeError(
                "Pipeline is not active. Use "
                "'with ForwardObservationPipeline() as pipeline:'."
            )

        inputs = {
            "angle_of_attack_deg": float(angle_of_attack_deg),
            "h": float(h),
            "dt": float(dt),
            "tf": float(tf),
            "Re": float(Re),
            "snapshot_freq": int(snapshot_freq),
        }

        outputs = self._forward.jacobian(
            inputs,
            jac_inputs=["angle_of_attack_deg"],
            jac_outputs=["ux", "uy"],
        )

        return {
            name: np.asarray(
                outputs[name]["angle_of_attack_deg"],
                dtype=np.float64,
            )
            for name in ("ux", "uy")
        }

    def observe_jvp(
        self,
        flow: dict[str, Any],
        field_tangent: dict[str, np.ndarray],
        sensor_x: np.ndarray,
        sensor_y: np.ndarray,
        sensor_times: np.ndarray,
    ) -> np.ndarray:
        """Push a field-space tangent through WakeObservation.

        Given d(field)/d(theta) in ``field_tangent``, this returns
        d(measurements)/d(theta) by evaluating T2's JVP at ``flow``. Composing
        it with :meth:`forward_angle_jacobian` carries a derivative across the
        Julia -> JAX boundary.
        """
        if self._observation is None:
            raise RuntimeError(
                "Pipeline is not active. Use "
                "'with ForwardObservationPipeline() as pipeline:'."
            )

        sensor_x = np.asarray(sensor_x, dtype=np.float64)
        sensor_y = np.asarray(sensor_y, dtype=np.float64)
        sensor_times = np.asarray(sensor_times, dtype=np.float64)

        self._validate_sensor_inputs(
            flow,
            sensor_x,
            sensor_y,
            sensor_times,
        )

        inputs = {
            "ux": flow["ux"],
            "uy": flow["uy"],
            "ux_x": flow["ux_x"],
            "ux_y": flow["ux_y"],
            "uy_x": flow["uy_x"],
            "uy_y": flow["uy_y"],
            "times": flow["times"],
            "sensor_x": sensor_x,
            "sensor_y": sensor_y,
            "sensor_times": sensor_times,
        }

        outputs = self._observation.jacobian_vector_product(
            inputs,
            jvp_inputs=["ux", "uy"],
            jvp_outputs=["measurements"],
            tangent_vector={
                "ux": np.asarray(field_tangent["ux"], dtype=np.float64),
                "uy": np.asarray(field_tangent["uy"], dtype=np.float64),
            },
        )

        return np.asarray(
            outputs["measurements"],
            dtype=np.float64,
        )

    def run(
        self,
        angle_of_attack_deg: float,
        sensor_x: np.ndarray,
        sensor_y: np.ndarray,
        sensor_times: np.ndarray,
        *,
        h: float = 0.1,
        dt: float = 0.005,
        tf: float = 0.1,
        Re: float = 200.0,
        snapshot_freq: int = 20,
    ) -> np.ndarray:
        """Run the complete forward-observation map."""
        flow = self.run_forward(
            angle_of_attack_deg,
            h=h,
            dt=dt,
            tf=tf,
            Re=Re,
            snapshot_freq=snapshot_freq,
        )

        return self.observe(
            flow,
            sensor_x,
            sensor_y,
            sensor_times,
        )

    def clear_forward_cache(self) -> None:
        """Remove all cached ImmersaForward solutions."""
        self._flow_cache.clear()

    def forward_cache_info(self) -> dict[str, int]:
        """Return statistics for the ImmersaForward cache."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._flow_cache),
            "max_size": self.max_cached_flows,
        }

    @staticmethod
    def _validate_sensor_inputs(
        flow: dict[str, Any],
        sensor_x: np.ndarray,
        sensor_y: np.ndarray,
        sensor_times: np.ndarray,
    ) -> None:
        """Check that sensors lie inside the interpolation domain."""
        if sensor_x.ndim != 1:
            raise ValueError("sensor_x must be one-dimensional.")

        if sensor_y.ndim != 1:
            raise ValueError("sensor_y must be one-dimensional.")

        if sensor_times.ndim != 1:
            raise ValueError("sensor_times must be one-dimensional.")

        if sensor_x.size != sensor_y.size:
            raise ValueError(
                "sensor_x and sensor_y must contain the same number of sensors."
            )

        if sensor_x.size == 0:
            raise ValueError("At least one sensor is required.")

        if sensor_times.size == 0:
            raise ValueError("At least one sensor time is required.")

        ux_x = np.asarray(flow["ux_x"], dtype=np.float64)
        ux_y = np.asarray(flow["ux_y"], dtype=np.float64)
        uy_x = np.asarray(flow["uy_x"], dtype=np.float64)
        uy_y = np.asarray(flow["uy_y"], dtype=np.float64)
        times = np.asarray(flow["times"], dtype=np.float64)

        # WakeObservation interpolates ux and uy on their own staggered grids.
        # Therefore sensors must lie inside the domain common to both fields.
        x_min = max(ux_x[0], uy_x[0])
        x_max = min(ux_x[-1], uy_x[-1])

        y_min = max(ux_y[0], uy_y[0])
        y_max = min(ux_y[-1], uy_y[-1])

        if np.any(sensor_x < x_min) or np.any(sensor_x > x_max):
            raise ValueError(f"sensor_x must lie inside [{x_min}, {x_max}].")

        if np.any(sensor_y < y_min) or np.any(sensor_y > y_max):
            raise ValueError(f"sensor_y must lie inside [{y_min}, {y_max}].")

        if np.any(sensor_times < times[0]) or np.any(sensor_times > times[-1]):
            raise ValueError(f"sensor_times must lie inside [{times[0]}, {times[-1]}].")


def run_forward_observation(
    angle_of_attack_deg: float,
    sensor_x: np.ndarray,
    sensor_y: np.ndarray,
    sensor_times: np.ndarray,
    *,
    h: float = 0.1,
    dt: float = 0.005,
    tf: float = 0.1,
    Re: float = 200.0,
    snapshot_freq: int = 20,
) -> np.ndarray:
    """Convenience function for a single forward-observation evaluation."""
    with ForwardObservationPipeline() as pipeline:
        return pipeline.run(
            angle_of_attack_deg,
            sensor_x,
            sensor_y,
            sensor_times,
            h=h,
            dt=dt,
            tf=tf,
            Re=Re,
            snapshot_freq=snapshot_freq,
        )
