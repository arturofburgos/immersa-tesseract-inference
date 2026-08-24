# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for immersa_tesseract_inference_sensor_array_design

"""Differentiable SensorArrayDesign Tesseract.

Global angle-of-attack discriminability of a sparse wake measurement batch.

This component is deliberately unaware of sensor coordinates. It consumes a
batch of predicted measurements -- one entry per design angle of attack -- and
returns a single scalar saying how well those angles can be told apart. Spatial
bounds and sensor separation are design-variable constraints and stay in the
application.

Inputs
------
measurements:
    Predicted wake measurements, shape (N_alpha, N_sensors, 5, 2).

alpha_deg:
    The design angle-of-attack grid, shape (N_alpha,).

delta_alpha_min_deg:
    Angle pairs closer than this are excluded. Angles a few degrees apart are
    physically hard to separate no matter where the probes sit, and including
    them lets that inevitability dominate the soft minimum.

tau:
    Soft-minimum temperature. Calibrated once by the application from the
    baseline pair-distance distribution and then held fixed.

Outputs
-------
discrimination:
    The scalar design score to be maximized.

pair_distances, min_pair_distance, n_pairs:
    Diagnostics. ``pair_distances`` lets the application calibrate ``tau``
    without reimplementing the metric.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, Int32
from tesseract_core.runtime.jax_recipes import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)

#
# Schemas
#


class InputSchema(BaseModel):
    """A batch of predicted measurements over the design AoA grid."""

    measurements: Differentiable[Array[(None, None, 5, 2), Float32]] = Field(
        description=(
            "Predicted wake measurements with shape "
            "(N_alpha, N_sensors, 5, 2). The final dimension is [ux, uy] and "
            "the five times are [12.0, 13.3, 15.1, 17.4, 20.0]."
        )
    )

    alpha_deg: Array[(None,), Float32] = Field(
        description="Design angle-of-attack grid in degrees, shape (N_alpha,)."
    )

    delta_alpha_min_deg: Float32 = Field(
        description=(
            "Minimum angular separation for a pair to be retained. Pairs closer "
            "than this are excluded from the soft minimum."
        )
    )

    tau: Float32 = Field(
        description=(
            "Soft-minimum temperature, in the same units as the normalized "
            "pair distances. Must be positive."
        )
    )


class OutputSchema(BaseModel):
    """Global discriminability of the supplied measurement batch."""

    discrimination: Differentiable[Float32] = Field(
        description=(
            "Normalized soft minimum over retained pairwise distances, "
            "-tau * log-mean-exp(-d/tau). Carries the same units as the "
            "distances: it equals d0 when all retained pairs sit at d0, and "
            "tends to min(d) as tau tends to zero. Larger is better -- the "
            "worst confusable angle pair is further apart."
        )
    )

    pair_distances: Array[(None, None), Float32] = Field(
        description=(
            "Symmetric (N_alpha, N_alpha) matrix of normalized squared "
            "measurement distances, with a zero diagonal. Excluded pairs are "
            "still reported here; only the score applies the mask."
        )
    )

    min_pair_distance: Float32 = Field(
        description="Hard minimum distance over retained pairs, for reporting."
    )

    n_pairs: Int32 = Field(description="Number of retained pairs.")


#
# Discrimination criterion
#


def _pair_distances(measurements: jax.Array) -> jax.Array:
    """Normalized squared distance between every pair of AoA measurements.

    Each entry is

        d_ij = ||m_i - m_j||^2 / N_scalar,   N_scalar = N_sensors * 5 * 2,

    so the value is comparable across sensor budgets.

    Flattening the trailing axes makes the permutation behaviour explicit: a
    relabelling of the sensors permutes the flattened vectors of *both* m_i and
    m_j identically, and the Euclidean norm is invariant under a shared
    permutation. The score is therefore independent of sensor ordering.
    """
    n_alpha = measurements.shape[0]

    flat = measurements.reshape(n_alpha, -1)

    n_scalar = flat.shape[1]

    difference = flat[:, None, :] - flat[None, :, :]

    return jnp.sum(difference**2, axis=-1) / n_scalar


def _retained_mask(
    alpha_deg: jax.Array,
    delta_alpha_min_deg: jax.Array,
) -> jax.Array:
    """Strict upper-triangular mask of sufficiently separated AoA pairs."""
    separation = jnp.abs(alpha_deg[:, None] - alpha_deg[None, :])

    upper = jnp.triu(
        jnp.ones(
            (alpha_deg.shape[0], alpha_deg.shape[0]),
            dtype=bool,
        ),
        k=1,
    )

    return upper & (separation >= delta_alpha_min_deg)


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """Score how distinguishable the design angles are under these sensors."""
    measurements = inputs["measurements"]

    alpha_deg = inputs["alpha_deg"]

    delta_alpha_min_deg = inputs["delta_alpha_min_deg"]

    tau = inputs["tau"]

    distances = _pair_distances(measurements)

    mask = _retained_mask(alpha_deg, delta_alpha_min_deg)

    # Excluded pairs are pushed to +inf so they contribute exp(-inf) = 0 to the
    # sum. Doing this with a `where` keeps the masked entries out of the
    # gradient as well, rather than merely scaling them by zero.
    neg_scaled = jnp.where(
        mask,
        -distances / tau,
        -jnp.inf,
    )

    n_pairs = jnp.sum(mask)

    # Normalized soft minimum: a log-MEAN-exp rather than a log-sum-exp.
    #
    #     D = -tau * log( (1/|P|) sum exp(-d/tau) )
    #       = -tau * ( logsumexp(-d/tau) - log|P| ).
    #
    # Subtracting log|P| removes the -tau*log|P| offset that the unnormalized
    # form carries, so D lands on the scale of the distances themselves: if
    # every retained pair sits at the same distance d0, then D == d0 exactly,
    # and D -> min(d) as tau -> 0. For a fixed mask and tau this differs from
    # the unnormalized score only by a constant, so the gradient with respect
    # to the measurements is untouched.
    #
    # logsumexp is stable by construction -- it subtracts the maximum
    # internally -- so no underflow even when d/tau is large.
    discrimination = -tau * (
        jax.scipy.special.logsumexp(neg_scaled)
        - jnp.log(n_pairs.astype(neg_scaled.dtype))
    )

    min_pair_distance = jnp.min(
        jnp.where(mask, distances, jnp.inf),
    )

    return {
        "discrimination": discrimination,
        "pair_distances": distances,
        "min_pair_distance": min_pair_distance,
        # int32, not int64: x64 is disabled in this runtime, so requesting
        # int64 here only earns a truncation warning on every call.
        "n_pairs": n_pairs.astype(jnp.int32),
    }


#
# Required endpoint
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Evaluate the global discrimination criterion."""
    out = jax_apply(apply_jit, inputs)
    return OutputSchema(**out)


#
# JAX-handled derivative endpoints (no need to modify)
#


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
) -> dict[str, dict[str, Any]]:
    """Full Jacobian of the requested outputs w.r.t. the requested inputs."""
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Forward-mode directional derivative along ``tangent_vector``."""
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Reverse-mode pullback of ``cotangent_vector`` onto the inputs.

    This is the endpoint the sensor-design optimizer uses: seeding
    ``discrimination`` with 1.0 returns dD/d(measurements) in a single call.
    """
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs: Any) -> dict:
    """Infer output shapes and dtypes without running the computation."""
    return jax_abstract_eval(apply_jit, abstract_inputs)
