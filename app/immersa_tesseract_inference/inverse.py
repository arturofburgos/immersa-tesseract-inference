"""Inverse-problem utilities for recovering angle of attack."""

import time
from dataclasses import dataclass

import numpy as np

from immersa_tesseract_inference.pipeline import ForwardObservationPipeline

# How infer_angle_of_attack obtains dm/dalpha. Both routes return the same
# quantity; they differ in where the finite difference is formed.
_SENSITIVITY_BACKENDS = frozenset({"finite_difference", "tesseract"})


def _print_cache_status(
    pipeline: ForwardObservationPipeline,
    *,
    prefix: str = "  ",
) -> None:
    """Print the current ImmersaForward cache statistics."""
    info = pipeline.forward_cache_info()

    print(
        f"{prefix}cache: "
        f"{info['hits']} hits, "
        f"{info['misses']} misses, "
        f"{info['size']}/{info['max_size']} stored",
        flush=True,
    )


def objective(
    pipeline: ForwardObservationPipeline,
    angle_of_attack_deg: float,
    observed_measurements: np.ndarray,
    sensor_x: np.ndarray,
    sensor_y: np.ndarray,
    sensor_times: np.ndarray,
    *,
    h: float = 0.1,
    dt: float = 0.005,
    tf: float = 0.1,
    Re: float = 200.0,
    snapshot_freq: int = 20,
) -> float:
    """Evaluate the sparse-wake angle-of-attack objective.

    The objective is

        J(alpha) = 0.5 * ||m(alpha) - m_obs||_2^2,

    where m(alpha) is produced by the ImmersaForward -> WakeObservation
    pipeline and m_obs contains the target sparse wake measurements.
    """
    observed_measurements = np.asarray(
        observed_measurements,
        dtype=np.float64,
    )

    predicted_measurements = pipeline.run(
        angle_of_attack_deg=angle_of_attack_deg,
        sensor_x=sensor_x,
        sensor_y=sensor_y,
        sensor_times=sensor_times,
        h=h,
        dt=dt,
        tf=tf,
        Re=Re,
        snapshot_freq=snapshot_freq,
    )

    if predicted_measurements.shape != observed_measurements.shape:
        raise ValueError(
            "Predicted and observed measurements must have the same shape: "
            f"predicted={predicted_measurements.shape}, "
            f"observed={observed_measurements.shape}."
        )

    if not np.all(np.isfinite(observed_measurements)):
        raise ValueError("Observed measurements must contain only finite values.")

    residual = predicted_measurements - observed_measurements

    return 0.5 * float(np.sum(residual**2))


def measurement_sensitivity(
    pipeline: ForwardObservationPipeline,
    angle_of_attack_deg: float,
    sensor_x: np.ndarray,
    sensor_y: np.ndarray,
    sensor_times: np.ndarray,
    *,
    epsilon_deg: float = 0.5,
    h: float = 0.1,
    dt: float = 0.005,
    tf: float = 0.1,
    Re: float = 200.0,
    snapshot_freq: int = 20,
    verbose: bool = False,
) -> np.ndarray:
    """Compute dm/dalpha with a central finite difference."""
    if epsilon_deg <= 0.0:
        raise ValueError("epsilon_deg must be positive.")

    alpha_plus = angle_of_attack_deg + epsilon_deg
    alpha_minus = angle_of_attack_deg - epsilon_deg

    if verbose:
        print(
            f"  [sensitivity 1/2] Evaluating F({alpha_plus:.6f} deg)...",
            flush=True,
        )

    measurements_plus = pipeline.run(
        angle_of_attack_deg=alpha_plus,
        sensor_x=sensor_x,
        sensor_y=sensor_y,
        sensor_times=sensor_times,
        h=h,
        dt=dt,
        tf=tf,
        Re=Re,
        snapshot_freq=snapshot_freq,
    )

    if verbose:
        print("  [sensitivity 1/2] Done.", flush=True)
        _print_cache_status(pipeline, prefix="    ")

        print(
            f"  [sensitivity 2/2] Evaluating F({alpha_minus:.6f} deg)...",
            flush=True,
        )

    measurements_minus = pipeline.run(
        angle_of_attack_deg=alpha_minus,
        sensor_x=sensor_x,
        sensor_y=sensor_y,
        sensor_times=sensor_times,
        h=h,
        dt=dt,
        tf=tf,
        Re=Re,
        snapshot_freq=snapshot_freq,
    )

    if verbose:
        print("  [sensitivity 2/2] Done.", flush=True)
        _print_cache_status(pipeline, prefix="    ")

    return (measurements_plus - measurements_minus) / (2.0 * epsilon_deg)


def measurement_sensitivity_tesseract(
    pipeline: ForwardObservationPipeline,
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
    verbose: bool = False,
) -> np.ndarray:
    """Compute dm/dalpha by composing the T1 and T2 derivative endpoints.

    This is the Tesseract-native route:

        alpha
          -> ImmersaForward.jacobian      (Julia, central FD internally)
          -> d(ux, uy)/d(alpha)
          -> WakeObservation.jacobian_vector_product   (JAX)
          -> dm/dalpha

    The finite difference is never formed here; it is requested from the T1
    Tesseract, so the derivative genuinely crosses the Julia/JAX boundary
    through the Tesseract interface.

    Unlike :func:`measurement_sensitivity`, the two perturbed CFD solves happen
    inside the T1 container and are therefore invisible to this pipeline's flow
    cache. That method remains the cheaper choice for large multistart studies.
    """
    if verbose:
        print(
            "  [sensitivity 1/2] Requesting d(field)/d(alpha) from "
            f"ImmersaForward.jacobian at {angle_of_attack_deg:.6f} deg...",
            flush=True,
        )

    field_tangent = pipeline.forward_angle_jacobian(
        angle_of_attack_deg,
        h=h,
        dt=dt,
        tf=tf,
        Re=Re,
        snapshot_freq=snapshot_freq,
    )

    if verbose:
        print("  [sensitivity 1/2] Done.", flush=True)
        print(
            "  [sensitivity 2/2] Propagating the tangent through WakeObservation...",
            flush=True,
        )

    # The unperturbed flow supplies the grids and times that T2 needs. It is
    # normally already cached from the objective evaluation at this angle, so
    # this costs no additional CFD solve.
    flow = pipeline.run_forward(
        angle_of_attack_deg,
        h=h,
        dt=dt,
        tf=tf,
        Re=Re,
        snapshot_freq=snapshot_freq,
    )

    sensitivity = pipeline.observe_jvp(
        flow,
        field_tangent,
        sensor_x,
        sensor_y,
        sensor_times,
    )

    if verbose:
        print("  [sensitivity 2/2] Done.", flush=True)
        _print_cache_status(pipeline, prefix="    ")

    return sensitivity


def objective_gradient(
    pipeline: ForwardObservationPipeline,
    angle_of_attack_deg: float,
    observed_measurements: np.ndarray,
    sensor_x: np.ndarray,
    sensor_y: np.ndarray,
    sensor_times: np.ndarray,
    *,
    epsilon_deg: float = 0.5,
    h: float = 0.1,
    dt: float = 0.005,
    tf: float = 0.1,
    Re: float = 200.0,
    snapshot_freq: int = 20,
) -> float:
    """Evaluate dJ/dalpha for the least-squares AoA objective."""
    observed_measurements = np.asarray(
        observed_measurements,
        dtype=np.float64,
    )

    predicted_measurements = pipeline.run(
        angle_of_attack_deg=angle_of_attack_deg,
        sensor_x=sensor_x,
        sensor_y=sensor_y,
        sensor_times=sensor_times,
        h=h,
        dt=dt,
        tf=tf,
        Re=Re,
        snapshot_freq=snapshot_freq,
    )

    sensitivity = measurement_sensitivity(
        pipeline,
        angle_of_attack_deg,
        sensor_x,
        sensor_y,
        sensor_times,
        epsilon_deg=epsilon_deg,
        h=h,
        dt=dt,
        tf=tf,
        Re=Re,
        snapshot_freq=snapshot_freq,
    )

    if predicted_measurements.shape != observed_measurements.shape:
        raise ValueError(
            "Predicted and observed measurements must have the same shape."
        )

    residual = predicted_measurements - observed_measurements

    return float(np.sum(residual * sensitivity))


@dataclass
class AoAInferenceResult:
    """Result of angle-of-attack inference."""

    angle_of_attack_deg: float
    objective: float
    converged: bool
    iterations: int
    history: list[dict[str, float]]


def infer_angle_of_attack(
    pipeline: ForwardObservationPipeline,
    observed_measurements: np.ndarray,
    sensor_x: np.ndarray,
    sensor_y: np.ndarray,
    sensor_times: np.ndarray,
    *,
    initial_angle_deg: float,
    sensitivity_backend: str = "finite_difference",
    epsilon_deg: float = 0.5,
    damping: float = 1.0e-12,
    max_step_deg: float = 10.0,
    max_iterations: int = 10,
    max_backtracks: int = 6,
    objective_tolerance: float = 1.0e-10,
    gradient_tolerance: float = 1.0e-8,
    step_tolerance_deg: float = 1.0e-3,
    angle_bounds: tuple[float, float] = (0.0, 90.0),
    h: float = 0.1,
    dt: float = 0.005,
    tf: float = 0.1,
    Re: float = 200.0,
    snapshot_freq: int = 20,
    verbose: bool = False,
) -> AoAInferenceResult:
    """Infer angle of attack using damped Gauss-Newton iterations.

    For the least-squares objective

        J(alpha) = 0.5 * ||m(alpha) - m_obs||_2^2,

    the scalar Gauss-Newton step is

        delta_alpha = -(s^T r) / (s^T s + damping),

    where

        r = m(alpha) - m_obs
        s = dm/dalpha.

    A maximum step and backtracking are used to prevent an inaccurate local
    linearization from producing an excessively large update.

    ``sensitivity_backend`` selects how dm/dalpha is obtained without changing
    any of the numerics above:

    ``"finite_difference"``
        :func:`measurement_sensitivity` -- the app-level central difference on
        the composed T1 -> T2 map. Its perturbed solves populate the pipeline
        flow cache, so this stays the cheaper option for multistart studies.

    ``"tesseract"``
        :func:`measurement_sensitivity_tesseract` -- the derivative reported by
        the ImmersaForward Jacobian endpoint and propagated through the
        WakeObservation JVP.
    """
    if sensitivity_backend not in _SENSITIVITY_BACKENDS:
        raise ValueError(
            f"sensitivity_backend must be one of {sorted(_SENSITIVITY_BACKENDS)}; "
            f"got {sensitivity_backend!r}."
        )

    if epsilon_deg <= 0.0:
        raise ValueError("epsilon_deg must be positive.")

    if damping < 0.0:
        raise ValueError("damping must be non-negative.")

    if max_step_deg <= 0.0:
        raise ValueError("max_step_deg must be positive.")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    if max_backtracks < 0:
        raise ValueError("max_backtracks must be non-negative.")

    if step_tolerance_deg <= 0.0:
        raise ValueError("step_tolerance_deg must be positive.")

    lower_bound, upper_bound = angle_bounds

    if lower_bound >= upper_bound:
        raise ValueError("angle_bounds must satisfy lower < upper.")

    observed_measurements = np.asarray(
        observed_measurements,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(observed_measurements)):
        raise ValueError("Observed measurements must contain only finite values.")

    alpha = float(
        np.clip(
            initial_angle_deg,
            lower_bound,
            upper_bound,
        )
    )

    if verbose:
        print("\n" + "=" * 68, flush=True)
        print("Starting Gauss-Newton AoA inference", flush=True)
        print("=" * 68, flush=True)
        print(f"Initial AoA       : {alpha:.6f} deg", flush=True)
        print(f"Maximum iterations: {max_iterations}", flush=True)
        print(f"FD epsilon        : {epsilon_deg:.6f} deg", flush=True)
        print(f"Maximum step      : {max_step_deg:.6f} deg", flush=True)
        print(
            f"Sensor count      : {sensor_x.size}",
            flush=True,
        )
        print(
            f"Observation times : {sensor_times.size}",
            flush=True,
        )
        print(
            f"Scalar observations: {observed_measurements.size}",
            flush=True,
        )

        print(
            f"\nInitial forward evaluation at alpha = {alpha:.6f} deg...",
            flush=True,
        )

    predicted_measurements = pipeline.run(
        angle_of_attack_deg=alpha,
        sensor_x=sensor_x,
        sensor_y=sensor_y,
        sensor_times=sensor_times,
        h=h,
        dt=dt,
        tf=tf,
        Re=Re,
        snapshot_freq=snapshot_freq,
    )

    if verbose:
        print("Initial forward evaluation complete.", flush=True)
        _print_cache_status(pipeline)

    if predicted_measurements.shape != observed_measurements.shape:
        raise ValueError(
            "Predicted and observed measurements must have the same shape."
        )

    history: list[dict[str, float]] = []
    converged = False

    for iteration in range(max_iterations):
        iteration_start = time.perf_counter()

        residual = predicted_measurements - observed_measurements
        current_objective = 0.5 * float(np.sum(residual**2))

        if verbose:
            print("\n" + "=" * 68, flush=True)
            print(
                f"Gauss-Newton iteration {iteration + 1} / {max_iterations}",
                flush=True,
            )
            print("=" * 68, flush=True)
            print(
                f"Current AoA       : {alpha:.6f} deg",
                flush=True,
            )
            print(
                f"Current objective : {current_objective:.8e}",
                flush=True,
            )

        if current_objective <= objective_tolerance:
            history.append(
                {
                    "iteration": float(iteration),
                    "angle_of_attack_deg": alpha,
                    "objective": current_objective,
                    "gradient": 0.0,
                    "step_deg": 0.0,
                }
            )

            converged = True

            if verbose:
                print(
                    "\nConverged: objective tolerance reached.",
                    flush=True,
                )

            break

        if verbose:
            print("\nComputing measurement sensitivity...", flush=True)

        if sensitivity_backend == "tesseract":
            sensitivity = measurement_sensitivity_tesseract(
                pipeline,
                alpha,
                sensor_x,
                sensor_y,
                sensor_times,
                h=h,
                dt=dt,
                tf=tf,
                Re=Re,
                snapshot_freq=snapshot_freq,
                verbose=verbose,
            )
        else:
            sensitivity = measurement_sensitivity(
                pipeline,
                alpha,
                sensor_x,
                sensor_y,
                sensor_times,
                epsilon_deg=epsilon_deg,
                h=h,
                dt=dt,
                tf=tf,
                Re=Re,
                snapshot_freq=snapshot_freq,
                verbose=verbose,
            )

        gradient = float(np.sum(residual * sensitivity))
        gauss_newton_curvature = float(np.sum(sensitivity**2))

        if verbose:
            print("\nSensitivity complete.", flush=True)
            print(
                f"Gradient           : {gradient:.8e}",
                flush=True,
            )
            print(
                f"GN curvature       : {gauss_newton_curvature:.8e}",
                flush=True,
            )

        if abs(gradient) <= gradient_tolerance:
            history.append(
                {
                    "iteration": float(iteration),
                    "angle_of_attack_deg": alpha,
                    "objective": current_objective,
                    "gradient": gradient,
                    "step_deg": 0.0,
                }
            )

            converged = True

            if verbose:
                print(
                    "\nConverged: gradient tolerance reached.",
                    flush=True,
                )

            break

        step = -gradient / (gauss_newton_curvature + damping)

        unconstrained_step = float(step)

        step = float(
            np.clip(
                step,
                -max_step_deg,
                max_step_deg,
            )
        )

        if verbose:
            print(
                f"Raw GN step        : {unconstrained_step:+.6f} deg",
                flush=True,
            )

            if step != unconstrained_step:
                print(
                    f"Limited GN step    : {step:+.6f} deg",
                    flush=True,
                )
            else:
                print(
                    f"Proposed GN step   : {step:+.6f} deg",
                    flush=True,
                )

        if abs(step) <= step_tolerance_deg:
            history.append(
                {
                    "iteration": float(iteration),
                    "angle_of_attack_deg": alpha,
                    "objective": current_objective,
                    "gradient": gradient,
                    "step_deg": 0.0,
                }
            )

            converged = True

            if verbose:
                print(
                    "\nConverged: proposed Gauss-Newton step "
                    "is below the step tolerance.",
                    flush=True,
                )

            break

        accepted = False
        trial_step = step
        actual_step = 0.0

        if verbose:
            print("\nStarting backtracking line search...", flush=True)

        for backtrack_index in range(max_backtracks + 1):
            trial_alpha = float(
                np.clip(
                    alpha + trial_step,
                    lower_bound,
                    upper_bound,
                )
            )

            actual_step = trial_alpha - alpha

            if abs(actual_step) <= step_tolerance_deg:
                if verbose:
                    print(
                        "  Trial step became smaller than the step tolerance.",
                        flush=True,
                    )

                break

            if verbose:
                print(
                    f"  Trial {backtrack_index + 1}/"
                    f"{max_backtracks + 1}: "
                    f"alpha = {trial_alpha:.6f} deg",
                    flush=True,
                )
                print("    Evaluating forward model...", flush=True)

            trial_measurements = pipeline.run(
                angle_of_attack_deg=trial_alpha,
                sensor_x=sensor_x,
                sensor_y=sensor_y,
                sensor_times=sensor_times,
                h=h,
                dt=dt,
                tf=tf,
                Re=Re,
                snapshot_freq=snapshot_freq,
            )

            trial_residual = trial_measurements - observed_measurements

            trial_objective = 0.5 * float(np.sum(trial_residual**2))

            if verbose:
                print(
                    f"    Trial objective: {trial_objective:.8e}",
                    flush=True,
                )
                _print_cache_status(
                    pipeline,
                    prefix="    ",
                )

            if trial_objective < current_objective:
                accepted = True

                if verbose:
                    print("    Result: ACCEPTED", flush=True)

                break

            if verbose:
                print("    Result: REJECTED", flush=True)
                print("    Halving trial step.", flush=True)

            trial_step *= 0.5

        history.append(
            {
                "iteration": float(iteration),
                "angle_of_attack_deg": alpha,
                "objective": current_objective,
                "gradient": gradient,
                "step_deg": actual_step if accepted else 0.0,
            }
        )

        iteration_elapsed = time.perf_counter() - iteration_start

        if not accepted:
            if verbose:
                print(
                    "\nStopping: backtracking could not find "
                    "an objective-decreasing step.",
                    flush=True,
                )
                print(
                    f"Iteration wall time : {iteration_elapsed / 60.0:.2f} min",
                    flush=True,
                )

            break

        alpha = trial_alpha
        predicted_measurements = trial_measurements

        if verbose:
            print("\nIteration accepted.", flush=True)
            print(
                f"Accepted AoA        : {alpha:.6f} deg",
                flush=True,
            )
            print(
                f"Accepted step       : {actual_step:+.6f} deg",
                flush=True,
            )
            print(
                f"New objective       : {trial_objective:.8e}",
                flush=True,
            )
            print(
                f"Iteration wall time : {iteration_elapsed / 60.0:.2f} min",
                flush=True,
            )
            _print_cache_status(pipeline)

        if abs(actual_step) <= step_tolerance_deg:
            converged = True

            if verbose:
                print(
                    "\nConverged: accepted step is below the step tolerance.",
                    flush=True,
                )

            break

    final_residual = predicted_measurements - observed_measurements

    final_objective = 0.5 * float(np.sum(final_residual**2))

    if final_objective <= objective_tolerance:
        converged = True

    if verbose:
        print("\n" + "=" * 68, flush=True)
        print("Gauss-Newton inference finished", flush=True)
        print("=" * 68, flush=True)
        print(
            f"Recovered AoA   : {alpha:.8f} deg",
            flush=True,
        )
        print(
            f"Final objective : {final_objective:.8e}",
            flush=True,
        )
        print(
            f"Iterations      : {len(history)}",
            flush=True,
        )
        print(
            f"Converged       : {converged}",
            flush=True,
        )
        _print_cache_status(pipeline)

    return AoAInferenceResult(
        angle_of_attack_deg=alpha,
        objective=final_objective,
        converged=converged,
        iterations=len(history),
        history=history,
    )
