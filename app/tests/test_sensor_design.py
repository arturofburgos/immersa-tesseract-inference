"""Tests for the SensorArrayDesign component and the T3 -> T4 composition."""

import numpy as np
from immersa_tesseract_inference.sensor_design import (
    ALPHA_GRID_DEG,
    SensorDesignPipeline,
    canonicalize_layout,
    retained_pair_mask,
    separation_penalty,
)
from tesseract_core import Tesseract

DESIGN_IMAGE = "immersa_tesseract_inference_sensor_array_design"

TAU = 0.5


def _synthetic_batch(
    n_alpha: int = 8,
    n_sensors: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Reproducible measurement batch of shape (n_alpha, n_sensors, 5, 2)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_alpha, n_sensors, 5, 2)).astype(np.float32)


def _design_inputs(measurements: np.ndarray, delta: float = 7.5) -> dict:
    alpha = np.linspace(20.0, 85.0, measurements.shape[0]).astype(np.float32)
    return {
        "measurements": measurements,
        "alpha_deg": alpha,
        "delta_alpha_min_deg": delta,
        "tau": TAU,
    }


def test_score_is_invariant_to_sensor_relabelling() -> None:
    """Sensors are unlabelled, so swapping them must not change the score."""
    measurements = _synthetic_batch()

    swapped = measurements[:, ::-1].copy()

    with Tesseract.from_image(DESIGN_IMAGE) as design:
        original = float(
            np.asarray(design.apply(_design_inputs(measurements))["discrimination"])
        )
        permuted = float(
            np.asarray(design.apply(_design_inputs(swapped))["discrimination"])
        )

    assert original == permuted


def test_pair_distances_are_symmetric_with_zero_diagonal() -> None:
    """pair_distances must be a proper distance matrix."""
    measurements = _synthetic_batch()

    with Tesseract.from_image(DESIGN_IMAGE) as design:
        outputs = design.apply(_design_inputs(measurements))

    distances = np.asarray(outputs["pair_distances"])
    discrimination = float(np.asarray(outputs["discrimination"]))
    min_distance = float(np.asarray(outputs["min_pair_distance"]))

    assert distances.shape == (measurements.shape[0], measurements.shape[0])
    assert np.allclose(distances, distances.T, atol=1e-6)
    assert np.max(np.abs(np.diag(distances))) == 0.0

    # The normalized soft minimum is bracketed by the hard minimum below and
    # the mean above: -tau*log(mean exp(-d/tau)) >= min(d) because the mean of
    # the exponentials cannot exceed exp(-min(d)/tau), and <= mean(d) by
    # Jensen. The unnormalized log-sum-exp form satisfied the opposite bound.
    mask = retained_pair_mask(
        np.linspace(20.0, 85.0, measurements.shape[0]), 7.5
    )
    retained = distances[mask]

    assert min_distance <= discrimination <= float(retained.mean()) + 1e-6


def test_retained_pair_count_matches_analytic_value() -> None:
    """The AoA mask must retain exactly the analytically expected pairs.

    On the production 2.5 degree grid, requiring a separation of at least
    delta means an index gap of at least k = delta / 2.5, which retains
    sum_{j=1}^{n-k} j pairs.
    """
    n_alpha = ALPHA_GRID_DEG.size

    measurements = _synthetic_batch(n_alpha=n_alpha)

    with Tesseract.from_image(DESIGN_IMAGE) as design:
        for delta, expected in ((5.0, 325), (7.5, 300), (10.0, 276), (15.0, 231)):
            inputs = {
                "measurements": measurements,
                "alpha_deg": ALPHA_GRID_DEG.astype(np.float32),
                "delta_alpha_min_deg": delta,
                "tau": TAU,
            }

            reported = int(np.asarray(design.apply(inputs)["n_pairs"]))

            # Agrees with the closed form...
            assert reported == expected, (delta, reported, expected)

            # ...and with the application-side mask used to select distances.
            assert reported == int(
                retained_pair_mask(ALPHA_GRID_DEG, delta).sum()
            )


def test_design_vjp_matches_finite_differences() -> None:
    """T4's reverse-mode gradient must match central differences.

    Directional derivatives along random unit vectors are compared rather than
    single tensor entries: one entry of a 120-element batch barely moves the
    score, so a per-element difference is dominated by float32 noise, while a
    directional probe carries the full gradient magnitude.
    """
    measurements = _synthetic_batch(n_alpha=6)

    inputs = _design_inputs(measurements)

    step = 1e-2

    rng = np.random.default_rng(1)

    with Tesseract.from_image(DESIGN_IMAGE) as design:
        cotangent = np.asarray(
            design.vector_jacobian_product(
                inputs,
                vjp_inputs=["measurements"],
                vjp_outputs=["discrimination"],
                cotangent_vector={"discrimination": 1.0},
            )["measurements"],
            dtype=np.float64,
        )

        def score(batch: np.ndarray) -> float:
            return float(
                np.asarray(
                    design.apply(
                        {**inputs, "measurements": batch.astype(np.float32)}
                    )["discrimination"]
                )
            )

        for _ in range(3):
            direction = rng.normal(size=measurements.shape)
            direction /= np.linalg.norm(direction)

            finite_difference = (
                score(measurements + step * direction)
                - score(measurements - step * direction)
            ) / (2.0 * step)

            analytic = float(np.sum(cotangent * direction))

            assert np.isclose(analytic, finite_difference, rtol=1e-3, atol=1e-7), (
                analytic,
                finite_difference,
            )


def test_separation_penalty_is_inactive_when_satisfied() -> None:
    """The penalty and its gradient vanish once the sensors are far enough."""
    far = np.array([1.0, -0.4, 1.0, 0.4])

    penalty, gradient = separation_penalty(far, lambda_separation=100.0)

    assert penalty == 0.0
    assert np.all(gradient == 0.0)

    close = np.array([1.5, 0.0, 1.55, 0.0])

    penalty, gradient = separation_penalty(close, lambda_separation=100.0)

    assert penalty > 0.0

    # A descent step must push the sensors apart and reduce the penalty. The
    # per-component signs depend on which sensor happens to be to the left, so
    # assert the geometric property rather than a fixed sign pattern.
    stepped = close - 1e-4 * gradient

    def separation(design: np.ndarray) -> float:
        return float(np.hypot(design[0] - design[2], design[1] - design[3]))

    stepped_penalty, _ = separation_penalty(stepped, lambda_separation=100.0)

    assert separation(stepped) > separation(close)
    assert stepped_penalty < penalty


def test_canonicalize_orders_sensors_without_moving_them() -> None:
    """Canonicalization is a relabelling, not a change of layout."""
    layout = np.array([2.4, 0.7, 1.3, -0.6])

    canonical = canonicalize_layout(layout)

    assert canonical.tolist() == [1.3, -0.6, 2.4, 0.7]
    assert canonicalize_layout(canonical).tolist() == canonical.tolist()


def test_composed_sensor_gradient_matches_finite_differences() -> None:
    """The full T3 -> T4 reverse chain must match central differences.

    This is the acceptance test for the design gradient: it exercises the real
    WakeSurrogate and SensorArrayDesign containers and the hand-assembled
    reverse chain between them, not any in-process JAX shortcut.
    """
    layout = np.array([1.3, -0.6, 2.4, 0.7])

    step = 3e-3

    with SensorDesignPipeline() as pipeline:
        calibration = pipeline.calibrate()

        tau = calibration["tau"]
        lambda_separation = calibration["lambda_separation"]

        # Keep the separation term out of the comparison's way.
        assert (
            separation_penalty(layout, lambda_separation=lambda_separation)[0] == 0.0
        )

        _, analytic = pipeline.objective_and_gradient(
            layout, tau=tau, lambda_separation=lambda_separation
        )

        def objective(design: np.ndarray) -> float:
            measurements = pipeline.measurements(design)
            scored = pipeline.discrimination(measurements, tau)
            penalty, _ = separation_penalty(
                design, lambda_separation=lambda_separation
            )
            return -scored["discrimination"] + penalty

        finite_difference = np.zeros(4)

        for index in range(4):
            plus = layout.copy()
            minus = layout.copy()
            plus[index] += step
            minus[index] -= step

            finite_difference[index] = (objective(plus) - objective(minus)) / (
                2.0 * step
            )

    names = ["x1", "y1", "x2", "y2"]

    print(f"\n{'coord':>6} {'composed AD':>16} {'central FD':>16} "
          f"{'abs err':>12} {'rel err':>12}")

    for index, name in enumerate(names):
        absolute = abs(analytic[index] - finite_difference[index])
        relative = absolute / max(abs(finite_difference[index]), 1e-12)
        print(
            f"{name:>6} {analytic[index]:16.9f} {finite_difference[index]:16.9f} "
            f"{absolute:12.3e} {relative:12.3e}"
        )

    cosine = float(
        analytic
        @ finite_difference
        / (np.linalg.norm(analytic) * np.linalg.norm(finite_difference))
    )

    relative_l2 = float(
        np.linalg.norm(analytic - finite_difference) / np.linalg.norm(finite_difference)
    )

    print(f"\nrelative L2 error : {relative_l2:.6e}")
    print(f"cosine similarity : {cosine:.9f}")

    assert relative_l2 < 5e-3
    assert cosine > 0.9999
