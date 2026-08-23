"""Train the WakeSurrogate model."""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from immersa_tesseract_inference.wake_surrogate_config import (
    SENSOR_TIMES,
    TRAIN_ALPHAS_BY_VERSION,
)


# ============================================================
# Paths
# ============================================================

DATA_ROOT = Path(
    "data/wake_surrogate"
)

MODEL_ROOT = Path(
    "models/wake_surrogate"
)


# ============================================================
# Model-version configuration
# ============================================================

MODEL_CONFIGS = {
    "v1": {
        "label": "V1 SPATIAL FOURIER",
        "directory": (
            MODEL_ROOT
            / "v1_spatial_fourier"
        ),
    },
    "v2": {
        "label": "V2 HIGH-AOA REFINEMENT",
        "directory": (
            MODEL_ROOT
            / "v2_high_aoa_refinement"
        ),
    },
    "v3": {
        "label": "V3 BROAD AOA REFINEMENT",
        "directory": (
            MODEL_ROOT
            / "v3_broad_aoa_refinement"
        ),
    },
}


# ============================================================
# Training configuration
# ============================================================

SEED = 5

WIDTH = 128
DEPTH = 3

FOURIER_FREQUENCIES = jnp.array(
    [
        1.0,
        2.0,
        4.0,
        8.0,
    ],
    dtype=jnp.float32,
)

BATCH_SIZE = 512

LEARNING_RATE = 1.0e-3

MAX_EPOCHS = 1000

PATIENCE = 100

OPTIMIZER = optax.adam(
    LEARNING_RATE
)


# ============================================================
# Command-line arguments
# ============================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train one version of the "
            "WakeSurrogate model."
        )
    )

    parser.add_argument(
        "--version",
        required=True,
        choices=(
            "v1",
            "v2",
            "v3",
        ),
        help=(
            "WakeSurrogate version to train."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. "
            "If omitted, the canonical model "
            "directory for the selected version "
            "is used."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow overwriting an existing "
            "best_model.eqx."
        ),
    )

    return parser.parse_args()


# ============================================================
# Data loading
# ============================================================


def load_split(
    split: str,
    allowed_alphas: tuple[float, ...] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[float],
]:
    """
    Load one dataset split.

    For the training split, allowed_alphas determines
    which already-generated CFD cases are used.

    Validation data are loaded completely.
    """

    split_dir = (
        DATA_ROOT
        / split
    )

    files = sorted(
        split_dir.glob(
            "*.npz"
        )
    )

    if not files:
        raise RuntimeError(
            f"No files found in {split_dir}."
        )

    if allowed_alphas is not None:
        allowed_array = np.asarray(
            allowed_alphas,
            dtype=np.float64,
        )
    else:
        allowed_array = None

    all_inputs = []
    all_targets = []
    selected_alphas = []

    for path in files:

        with np.load(
            path
        ) as data:

            alpha_deg = float(
                np.asarray(
                    data["alpha_deg"]
                ).reshape(-1)[0]
            )

            if allowed_array is not None:

                match = np.any(
                    np.isclose(
                        alpha_deg,
                        allowed_array,
                        rtol=0.0,
                        atol=1.0e-8,
                    )
                )

                if not match:
                    continue

            all_inputs.append(
                np.asarray(
                    data["inputs"]
                )
            )

            all_targets.append(
                np.asarray(
                    data["targets"]
                )
            )

            selected_alphas.append(
                alpha_deg
            )

    if not all_inputs:
        raise RuntimeError(
            f"No matching files found "
            f"for split '{split}'."
        )

    x = np.concatenate(
        all_inputs,
        axis=0,
    )

    y = np.concatenate(
        all_targets,
        axis=0,
    )

    selected_alphas = sorted(
        set(
            selected_alphas
        )
    )

    return (
        x,
        y,
        selected_alphas,
    )


def validate_training_alphas(
    selected_alphas: list[float],
    expected_alphas: tuple[float, ...],
) -> None:
    """
    Verify that exactly the requested training AoAs
    were loaded.
    """

    selected = np.asarray(
        selected_alphas,
        dtype=np.float64,
    )

    expected = np.asarray(
        sorted(
            expected_alphas
        ),
        dtype=np.float64,
    )

    if (
        selected.shape
        != expected.shape
    ):
        raise RuntimeError(
            "Incorrect number of training AoAs.\n"
            f"Expected {len(expected)}, "
            f"loaded {len(selected)}.\n"
            f"Expected: {expected.tolist()}\n"
            f"Loaded:   {selected.tolist()}"
        )

    if not np.allclose(
        selected,
        expected,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise RuntimeError(
            "Loaded AoAs do not match the "
            "requested model version.\n"
            f"Expected: {expected.tolist()}\n"
            f"Loaded:   {selected.tolist()}"
        )


# ============================================================
# Input normalization
# ============================================================


def normalize_inputs(
    x: np.ndarray,
) -> np.ndarray:
    """Map physical inputs approximately to [-1, 1]."""

    x = x.copy()

    # alpha:
    # [0, 90] -> [-1, 1]

    x[:, 0] = (
        2.0
        * x[:, 0]
        / 90.0
        - 1.0
    )

    # x:
    # [1, 3] -> [-1, 1]

    x[:, 1] = (
        x[:, 1]
        - 2.0
    )

    # y already lies in [-1, 1].

    return x


# ============================================================
# Model
# ============================================================


class WakeSurrogate(eqx.Module):
    """
    MLP wake surrogate.

    Physical inputs:
        [alpha, x, y]

    Fourier encoding:
        x and y only

    alpha remains a smooth raw parameter.

    Outputs:
        [ux(t1), uy(t1), ...,
         ux(t5), uy(t5)]
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        key: jax.Array,
    ):
        n_original = 3

        # Only x and y receive Fourier features.
        n_spatial = 2

        n_fourier = (
            2
            * n_spatial
            * len(
                FOURIER_FREQUENCIES
            )
        )

        # 3 raw inputs
        # + 8 sine features
        # + 8 cosine features
        # = 19 total features.

        feature_size = (
            n_original
            + n_fourier
        )

        self.mlp = eqx.nn.MLP(
            in_size=feature_size,
            out_size=10,
            width_size=WIDTH,
            depth=DEPTH,
            activation=jax.nn.silu,
            final_activation=lambda x: x,
            key=key,
        )

    def fourier_features(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """
        Keep alpha smooth and Fourier-encode
        only spatial coordinates x and y.
        """

        # x[0] = normalized alpha
        # x[1] = normalized spatial x
        # x[2] = normalized spatial y

        spatial = x[
            1:
        ]

        angles = (
            jnp.pi
            * spatial[:, None]
            * FOURIER_FREQUENCIES[
                None,
                :,
            ]
        )

        sin_features = (
            jnp.sin(
                angles
            ).reshape(-1)
        )

        cos_features = (
            jnp.cos(
                angles
            ).reshape(-1)
        )

        return jnp.concatenate(
            [
                x,
                sin_features,
                cos_features,
            ]
        )

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        """Predict ten normalized velocity values."""

        features = (
            self.fourier_features(
                x
            )
        )

        return self.mlp(
            features
        )


# ============================================================
# Prediction and loss
# ============================================================


def batch_predict(
    model: WakeSurrogate,
    x: jax.Array,
) -> jax.Array:
    """Evaluate model over a batch."""

    return jax.vmap(
        model
    )(
        x
    )


def loss_fn(
    model: WakeSurrogate,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """
    Mean-squared error in normalized target space.
    """

    prediction = batch_predict(
        model,
        x,
    )

    return jnp.mean(
        (
            prediction
            - y
        )
        ** 2
    )


# ============================================================
# Training step
# ============================================================


@eqx.filter_jit
def train_step(
    model: WakeSurrogate,
    optimizer_state,
    x_batch: jax.Array,
    y_batch: jax.Array,
):
    """Perform one Adam update."""

    loss, grads = (
        eqx.filter_value_and_grad(
            loss_fn
        )(
            model,
            x_batch,
            y_batch,
        )
    )

    updates, optimizer_state = (
        OPTIMIZER.update(
            grads,
            optimizer_state,
            eqx.filter(
                model,
                eqx.is_inexact_array,
            ),
        )
    )

    model = eqx.apply_updates(
        model,
        updates,
    )

    return (
        model,
        optimizer_state,
        loss,
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Train and validate one WakeSurrogate version."""

    args = parse_arguments()

    model_config = (
        MODEL_CONFIGS[
            args.version
        ]
    )

    train_alphas = (
        TRAIN_ALPHAS_BY_VERSION[
            args.version
        ]
    )

    if args.output_dir is None:

        model_root = (
            model_config[
                "directory"
            ]
        )

    else:

        model_root = (
            args.output_dir
        )

    model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        model_root
        / "best_model.eqx"
    )

    normalization_path = (
        model_root
        / "normalization.npz"
    )

    # --------------------------------------------------------
    # Protect already-trained models.
    # --------------------------------------------------------

    if (
        best_model_path.exists()
        and not args.overwrite
    ):
        raise RuntimeError(
            "A trained model already exists:\n"
            f"{best_model_path}\n\n"
            "Use --output-dir for a reproduction run, "
            "or --overwrite if replacement is intentional."
        )

    # ========================================================
    # Header
    # ========================================================

    print(
        "=" * 78
    )

    print(
        "WAKE SURROGATE TRAINING - "
        f"{model_config['label']}"
    )

    print(
        "=" * 78
    )

    print(
        f"Version          : "
        f"{args.version}"
    )

    print(
        f"Output directory : "
        f"{model_root}"
    )

    print(
        f"JAX devices      : "
        f"{jax.devices()}"
    )

    print(
        "Precision        : float32"
    )

    print(
        "Architecture     : MLP"
    )

    print(
        f"Width            : "
        f"{WIDTH}"
    )

    print(
        f"Depth            : "
        f"{DEPTH}"
    )

    print(
        "Activation       : SiLU"
    )

    print(
        "Fourier encoding : x, y only"
    )

    print(
        "AoA encoding     : "
        "raw normalized alpha only"
    )

    print(
        f"Fourier freq.    : "
        f"{np.asarray(FOURIER_FREQUENCIES).tolist()}"
    )

    print(
        f"Optimizer        : Adam"
    )

    print(
        f"Learning rate    : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Batch size       : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Seed             : "
        f"{SEED}"
    )

    # ========================================================
    # Load data
    # ========================================================

    (
        x_train,
        y_train,
        selected_train_alphas,
    ) = load_split(
        "train",
        allowed_alphas=train_alphas,
    )

    (
        x_valid,
        y_valid,
        selected_valid_alphas,
    ) = load_split(
        "validation"
    )

    validate_training_alphas(
        selected_train_alphas,
        train_alphas,
    )

    print()

    print(
        "Training AoAs "
        f"({len(selected_train_alphas)}):"
    )

    print(
        selected_train_alphas
    )

    print()

    print(
        "Validation AoAs "
        f"({len(selected_valid_alphas)}):"
    )

    print(
        selected_valid_alphas
    )

    print()

    print(
        f"Train:      "
        f"{x_train.shape} "
        f"-> {y_train.shape}"
    )

    print(
        f"Validation: "
        f"{x_valid.shape} "
        f"-> {y_valid.shape}"
    )

    # ========================================================
    # Normalize inputs
    # ========================================================

    x_train = normalize_inputs(
        x_train
    )

    x_valid = normalize_inputs(
        x_valid
    )

    # ========================================================
    # Normalize targets
    #
    # IMPORTANT:
    # statistics come ONLY from the training set.
    # ========================================================

    y_mean = np.mean(
        y_train,
        axis=0,
        keepdims=True,
    )

    y_std = np.std(
        y_train,
        axis=0,
        keepdims=True,
    )

    y_std = np.maximum(
        y_std,
        1.0e-8,
    )

    y_train = (
        y_train
        - y_mean
    ) / y_std

    y_valid = (
        y_valid
        - y_mean
    ) / y_std

    # ========================================================
    # Convert to float32 JAX arrays
    # ========================================================

    x_train = jnp.asarray(
        x_train,
        dtype=jnp.float32,
    )

    y_train = jnp.asarray(
        y_train,
        dtype=jnp.float32,
    )

    x_valid = jnp.asarray(
        x_valid,
        dtype=jnp.float32,
    )

    y_valid = jnp.asarray(
        y_valid,
        dtype=jnp.float32,
    )

    # ========================================================
    # Initialize model
    # ========================================================

    key = jax.random.PRNGKey(
        SEED
    )

    (
        model_key,
        shuffle_key,
    ) = jax.random.split(
        key
    )

    model = WakeSurrogate(
        model_key
    )

    optimizer_state = (
        OPTIMIZER.init(
            eqx.filter(
                model,
                eqx.is_inexact_array,
            )
        )
    )

    # ========================================================
    # Training
    # ========================================================

    n_train = (
        x_train.shape[0]
    )

    best_valid_loss = (
        np.inf
    )

    best_epoch = 0

    epochs_without_improvement = 0

    print(
        "\nStarting training...\n"
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        (
            shuffle_key,
            epoch_key,
        ) = jax.random.split(
            shuffle_key
        )

        permutation = np.asarray(
            jax.random.permutation(
                epoch_key,
                n_train,
            )
        )

        batch_losses = []

        for start in range(
            0,
            n_train,
            BATCH_SIZE,
        ):

            idx = permutation[
                start:
                start + BATCH_SIZE
            ]

            x_batch = (
                x_train[
                    idx
                ]
            )

            y_batch = (
                y_train[
                    idx
                ]
            )

            (
                model,
                optimizer_state,
                batch_loss,
            ) = train_step(
                model,
                optimizer_state,
                x_batch,
                y_batch,
            )

            batch_losses.append(
                float(
                    batch_loss
                )
            )

        train_loss = float(
            np.mean(
                batch_losses
            )
        )

        valid_loss = float(
            loss_fn(
                model,
                x_valid,
                y_valid,
            )
        )

        # ----------------------------------------------------
        # Save best model according to validation loss.
        # ----------------------------------------------------

        if (
            valid_loss
            < best_valid_loss
        ):

            best_valid_loss = (
                valid_loss
            )

            best_epoch = (
                epoch
            )

            epochs_without_improvement = 0

            eqx.tree_serialise_leaves(
                best_model_path,
                model,
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Progress output
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"epoch={epoch:4d} "
                f"| train={train_loss:.6e} "
                f"| valid={valid_loss:.6e} "
                f"| best={best_valid_loss:.6e}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping: "
                "no validation improvement for "
                f"{PATIENCE} epochs."
            )

            break

    # ========================================================
    # Save normalization metadata
    # ========================================================

    np.savez(
        normalization_path,

        y_mean=y_mean,

        y_std=y_std,

        alpha_min=0.0,

        alpha_max=90.0,

        x_min=1.0,

        x_max=3.0,

        y_min=-1.0,

        y_max=1.0,

        sensor_times=np.asarray(
            SENSOR_TIMES,
            dtype=np.float64,
        ),
    )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Version:              "
        f"{args.version}"
    )

    print(
        f"Training AoAs:        "
        f"{len(selected_train_alphas)}"
    )

    print(
        f"Training samples:     "
        f"{n_train}"
    )

    print(
        f"Best epoch:           "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_valid_loss:.6e}"
    )

    print(
        f"Model:                "
        f"{best_model_path}"
    )

    print(
        f"Normalization:        "
        f"{normalization_path}"
    )


if __name__ == "__main__":
    main()