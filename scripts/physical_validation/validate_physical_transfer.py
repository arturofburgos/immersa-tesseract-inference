"""Does the frozen surrogate sensor design transfer to real Immersa CFD?

Four checks, in increasing scientific weight:

1. Bank fidelity -- persisted fields must reproduce a direct T1 -> T2 evaluation.
2. Sensor-position derivatives -- T3's d(measurement)/d(sensor) against the
   physical d(T2[T1])/d(sensor).
3. Design gradient -- the full four-component dD/ds, surrogate versus physical.
4. The transfer test -- evaluate the *same* frozen T4 functional on real CFD
   measurements at the baseline and at the frozen design.

Sensors are passive, so the physical sensor-position derivative does not
differentiate through T1 at all: T1 supplies the mechanistic state and T2
supplies the entire sensor-coordinate dependence. The chain is

    T4 VJP -> T2 sensor VJP        (physical)
    T4 VJP -> T3 sensor VJP        (surrogate)

which is why the two are directly comparable.

The frozen tau and delta_alpha_min are used unchanged for the primary
comparison; nothing is recalibrated against CFD.
"""

import json
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.cfd_bank import (
    BASELINE_LAYOUT,
    DESIGN_GRID_DEG,
    DT,
    OBSERVATION_TIMES,
    RE,
    SNAPSHOT_FREQ,
    TF,
    H,
    load_flow,
    observation_sensor_jacobian,
    observe_bank,
)
from immersa_tesseract_inference.sensor_design import (
    effective_pair_count,
    retained_pair_mask,
    softmin_weights,
    unpack_layout,
)
from tesseract_core import Tesseract

FORWARD_IMAGE = "immersa_tesseract_inference_immersa_forward"
OBSERVATION_IMAGE = "immersa_tesseract_inference_wake_observation"
SURROGATE_IMAGE = "immersa_tesseract_inference_wake_surrogate"
DESIGN_IMAGE = "immersa_tesseract_inference_sensor_array_design"

SELECTION_JSON = Path("results/sensor_design/optimization/s_star_surrogate.json")

OUTPUT_DIR = Path("results/sensor_design/physical_validation")

# Angles used for the direct-versus-bank fidelity check.
BANK_CHECK_ANGLES = (20.0, 50.0, 85.0, 63.0)


def assemble_sensor_jacobian(
    jac_x: np.ndarray,
    jac_y: np.ndarray,
) -> np.ndarray:
    """Fold per-coordinate Jacobians into one (Ns, 5, 2, 4) tensor.

    Both endpoints return d(measurements)/d(sensor_x) with shape
    (Ns, 5, 2, Ns), which is block-diagonal because sensor i only sees its own
    coordinates. Only the diagonal blocks carry information, and they are laid
    out here against the design vector [x1, y1, x2, y2].
    """
    n_sensors = jac_x.shape[0]

    assembled = np.zeros((n_sensors, 5, 2, 2 * n_sensors), dtype=np.float64)

    for sensor in range(n_sensors):
        assembled[sensor, :, :, 2 * sensor] = jac_x[sensor, :, :, sensor]
        assembled[sensor, :, :, 2 * sensor + 1] = jac_y[sensor, :, :, sensor]

    return assembled


def compare_tensors(surrogate: np.ndarray, physical: np.ndarray) -> dict:
    """Agreement metrics between a surrogate and a physical tensor."""
    a = np.asarray(surrogate, dtype=np.float64).reshape(-1)
    b = np.asarray(physical, dtype=np.float64).reshape(-1)

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    difference = float(np.linalg.norm(a - b))

    cosine = (
        float(a @ b / (norm_a * norm_b)) if norm_a > 0.0 and norm_b > 0.0 else np.nan
    )

    return {
        "norm_surrogate": norm_a,
        "norm_physical": norm_b,
        "absolute_error": difference,
        "relative_l2_error": difference / norm_b if norm_b > 0.0 else np.nan,
        "cosine_similarity": cosine,
        "magnitude_ratio": norm_a / norm_b if norm_b > 0.0 else np.nan,
    }


def surrogate_measurements(
    surrogate: object,
    design: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """T3 measurements over an AoA grid, shape (n_alpha, Ns, 5, 2)."""
    sensor_x, sensor_y = unpack_layout(design)

    return np.stack(
        [
            np.asarray(
                surrogate.apply(
                    {
                        "angle_of_attack_deg": float(alpha),
                        "sensor_x": sensor_x.astype(np.float32),
                        "sensor_y": sensor_y.astype(np.float32),
                    }
                )["measurements"],
                dtype=np.float64,
            )
            for alpha in alphas
        ]
    )


def physical_measurements(
    observation: object,
    design: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """T1 -> T2 measurements over an AoA grid, shape (n_alpha, Ns, 5, 2)."""
    sensor_x, sensor_y = unpack_layout(design)

    return np.stack(
        [
            observe_bank(observation, load_flow(alpha), sensor_x, sensor_y)
            for alpha in alphas
        ]
    )


def score_batch(
    design_tess: object, batch: np.ndarray, tau: float, delta: float
) -> dict:
    """Evaluate the frozen T4 functional on a measurement batch."""
    outputs = design_tess.apply(
        {
            "measurements": batch.astype(np.float32),
            "alpha_deg": DESIGN_GRID_DEG.astype(np.float32),
            "delta_alpha_min_deg": float(delta),
            "tau": float(tau),
        }
    )

    distances = np.asarray(outputs["pair_distances"], dtype=np.float64)

    mask = retained_pair_mask(DESIGN_GRID_DEG, delta)

    retained = distances[mask]

    weights = np.sort(softmin_weights(retained, tau))[::-1]

    # Which AoA pair is hardest to tell apart.
    masked = np.where(mask, distances, np.inf)
    i, j = np.unravel_index(np.argmin(masked), masked.shape)

    return {
        "D": float(np.asarray(outputs["discrimination"])),
        "hard_min": float(np.asarray(outputs["min_pair_distance"])),
        "n_pairs": int(np.asarray(outputs["n_pairs"])),
        "n_eff": effective_pair_count(retained, tau),
        "top1_weight": float(weights[0]),
        "top10_weight": float(weights[:10].sum()),
        "hardest_pair_deg": [
            float(DESIGN_GRID_DEG[i]),
            float(DESIGN_GRID_DEG[j]),
        ],
        "pair_distances": distances,
    }


def design_gradient_from_cotangent(
    tess: object,
    cotangent: np.ndarray,
    alphas: np.ndarray,
    design: np.ndarray,
    *,
    physical: bool,
) -> np.ndarray:
    """Pull a measurement cotangent back onto [x1, y1, x2, y2].

    One reverse-mode call per angle, seeded with that angle's slice, summed
    over the grid. ``physical`` selects the WakeObservation route (real fields,
    held fixed) rather than the WakeSurrogate route.
    """
    sensor_x, sensor_y = unpack_layout(design)

    grad_x = np.zeros(sensor_x.size, dtype=np.float64)
    grad_y = np.zeros(sensor_y.size, dtype=np.float64)

    for index, alpha in enumerate(alphas):
        if physical:
            flow = load_flow(alpha)
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
                "sensor_times": OBSERVATION_TIMES,
            }
            seed = cotangent[index]
        else:
            inputs = {
                "angle_of_attack_deg": float(alpha),
                "sensor_x": sensor_x.astype(np.float32),
                "sensor_y": sensor_y.astype(np.float32),
            }
            seed = cotangent[index].astype(np.float32)

        outputs = tess.vector_jacobian_product(
            inputs,
            vjp_inputs=["sensor_x", "sensor_y"],
            vjp_outputs=["measurements"],
            cotangent_vector={"measurements": seed},
        )

        grad_x += np.asarray(outputs["sensor_x"], dtype=np.float64)
        grad_y += np.asarray(outputs["sensor_y"], dtype=np.float64)

    return np.array([grad_x[0], grad_y[0], grad_x[1], grad_y[1]])


def main() -> None:
    """Run every physical-transfer check and write the metrics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection = json.loads(SELECTION_JSON.read_text())

    s_star = np.array(selection["layout_vector"], dtype=np.float64)
    tau = float(selection["tau"])
    delta = float(selection["delta_alpha_min_deg"])

    layouts = {"baseline": BASELINE_LAYOUT, "s_star_surrogate": s_star}

    report: dict = {
        "frozen_design": selection["layout_vector"],
        "tau": tau,
        "delta_alpha_min_deg": delta,
        "configuration": {
            "h": H,
            "dt": DT,
            "tf": TF,
            "Re": RE,
            "snapshot_freq": SNAPSHOT_FREQ,
            "observation_times": OBSERVATION_TIMES.tolist(),
        },
    }

    print("=" * 78)
    print("Physical validation of the frozen surrogate sensor design")
    print("=" * 78)
    print(f"s_star_surrogate : {s_star.tolist()}")
    print(f"frozen tau       : {tau}")
    print(f"delta_alpha_min  : {delta} deg")
    print()

    with (
        Tesseract.from_image(OBSERVATION_IMAGE) as observation,
        Tesseract.from_image(SURROGATE_IMAGE) as surrogate,
        Tesseract.from_image(DESIGN_IMAGE) as design_tess,
    ):
        # ================================================
        # 1. Bank fidelity
        # ================================================

        print("-" * 78)
        print("1. Bank fidelity: direct T1 -> T2 versus persisted bank")
        print("-" * 78)

        fidelity = []

        with Tesseract.from_image(FORWARD_IMAGE) as forward:
            for alpha in BANK_CHECK_ANGLES:
                fresh = forward.apply(
                    {
                        "angle_of_attack_deg": float(alpha),
                        "h": H,
                        "dt": DT,
                        "tf": TF,
                        "Re": RE,
                        "snapshot_freq": SNAPSHOT_FREQ,
                    }
                )

                banked = load_flow(alpha)

                for name, layout in layouts.items():
                    sensor_x, sensor_y = unpack_layout(layout)

                    direct = np.asarray(
                        observation.apply(
                            {
                                "ux": fresh["ux"],
                                "uy": fresh["uy"],
                                "ux_x": fresh["ux_x"],
                                "ux_y": fresh["ux_y"],
                                "uy_x": fresh["uy_x"],
                                "uy_y": fresh["uy_y"],
                                "times": fresh["times"],
                                "sensor_x": sensor_x,
                                "sensor_y": sensor_y,
                                "sensor_times": OBSERVATION_TIMES,
                            }
                        )["measurements"],
                        dtype=np.float64,
                    )

                    from_bank = observe_bank(observation, banked, sensor_x, sensor_y)

                    absolute = float(np.max(np.abs(direct - from_bank)))
                    relative = absolute / float(np.max(np.abs(direct)))

                    fidelity.append(
                        {
                            "alpha_deg": float(alpha),
                            "layout": name,
                            "max_absolute_difference": absolute,
                            "max_relative_difference": relative,
                        }
                    )

                    print(
                        f"  alpha={alpha:5.1f}  {name:16s} "
                        f"max|diff|={absolute:.3e}  rel={relative:.3e}"
                    )

        report["bank_fidelity"] = fidelity

        # ================================================
        # 2 + 3. Derivatives and the transfer test
        # ================================================

        per_layout: dict = {}

        for name, layout in layouts.items():
            print()
            print("-" * 78)
            print(f"Layout: {name}  {np.round(layout, 6).tolist()}")
            print("-" * 78)

            sensor_x, sensor_y = unpack_layout(layout)

            # ---- sensor-position Jacobians, all 27 design angles ----

            per_alpha = []

            for alpha in DESIGN_GRID_DEG:
                surrogate_jac = surrogate.jacobian(
                    {
                        "angle_of_attack_deg": float(alpha),
                        "sensor_x": sensor_x.astype(np.float32),
                        "sensor_y": sensor_y.astype(np.float32),
                    },
                    jac_inputs=["sensor_x", "sensor_y"],
                    jac_outputs=["measurements"],
                )

                surrogate_tensor = assemble_sensor_jacobian(
                    np.asarray(
                        surrogate_jac["measurements"]["sensor_x"], dtype=np.float64
                    ),
                    np.asarray(
                        surrogate_jac["measurements"]["sensor_y"], dtype=np.float64
                    ),
                )

                physical_jac = observation_sensor_jacobian(
                    observation, load_flow(alpha), sensor_x, sensor_y
                )

                physical_tensor = assemble_sensor_jacobian(
                    physical_jac["sensor_x"], physical_jac["sensor_y"]
                )

                metrics = compare_tensors(surrogate_tensor, physical_tensor)
                metrics["alpha_deg"] = float(alpha)

                per_alpha.append(metrics)

            relative_errors = np.array([m["relative_l2_error"] for m in per_alpha])
            cosines = np.array([m["cosine_similarity"] for m in per_alpha])
            ratios = np.array([m["magnitude_ratio"] for m in per_alpha])

            worst = per_alpha[int(np.argmax(relative_errors))]

            jacobian_summary = {
                "median_relative_l2_error": float(np.median(relative_errors)),
                "mean_relative_l2_error": float(np.mean(relative_errors)),
                "max_relative_l2_error": float(np.max(relative_errors)),
                "min_cosine_similarity": float(np.min(cosines)),
                "median_cosine_similarity": float(np.median(cosines)),
                "median_magnitude_ratio": float(np.median(ratios)),
                "worst_alpha_deg": worst["alpha_deg"],
                "per_alpha": per_alpha,
            }

            print("  sensor Jacobian, T3 vs physical T2, over 27 design angles:")
            print(
                f"    relative L2 error  median {np.median(relative_errors):.4f}  "
                f"mean {np.mean(relative_errors):.4f}  "
                f"max {np.max(relative_errors):.4f} (alpha={worst['alpha_deg']:.1f})"
            )
            print(
                f"    cosine similarity  median {np.median(cosines):.6f}  "
                f"min {np.min(cosines):.6f}"
            )
            print(f"    magnitude ratio    median {np.median(ratios):.4f}")

            # ---- T4 on surrogate and on physical measurements ----

            surrogate_batch = surrogate_measurements(surrogate, layout, DESIGN_GRID_DEG)
            physical_batch = physical_measurements(observation, layout, DESIGN_GRID_DEG)

            surrogate_score = score_batch(design_tess, surrogate_batch, tau, delta)
            physical_score = score_batch(design_tess, physical_batch, tau, delta)

            print(
                f"  T4 discrimination   surrogate {surrogate_score['D']:.8f}   "
                f"physical {physical_score['D']:.8f}"
            )
            print(
                f"  hard minimum        surrogate {surrogate_score['hard_min']:.8f}   "
                f"physical {physical_score['hard_min']:.8f}"
            )
            print(f"  hardest physical pair: {physical_score['hardest_pair_deg']} deg")

            # ---- design gradient, surrogate versus physical ----

            def cotangent_for(batch: np.ndarray) -> np.ndarray:
                outputs = design_tess.vector_jacobian_product(
                    {
                        "measurements": batch.astype(np.float32),
                        "alpha_deg": DESIGN_GRID_DEG.astype(np.float32),
                        "delta_alpha_min_deg": float(delta),
                        "tau": float(tau),
                    },
                    vjp_inputs=["measurements"],
                    vjp_outputs=["discrimination"],
                    cotangent_vector={"discrimination": 1.0},
                )
                return np.asarray(outputs["measurements"], dtype=np.float64)

            surrogate_gradient = design_gradient_from_cotangent(
                surrogate,
                cotangent_for(surrogate_batch),
                DESIGN_GRID_DEG,
                layout,
                physical=False,
            )

            physical_gradient = design_gradient_from_cotangent(
                observation,
                cotangent_for(physical_batch),
                DESIGN_GRID_DEG,
                layout,
                physical=True,
            )

            gradient_metrics = compare_tensors(surrogate_gradient, physical_gradient)

            print("  design gradient dD/ds:")
            print(f"    surrogate {np.round(surrogate_gradient, 6).tolist()}")
            print(f"    physical  {np.round(physical_gradient, 6).tolist()}")
            print(
                f"    relative L2 {gradient_metrics['relative_l2_error']:.4f}   "
                f"cosine {gradient_metrics['cosine_similarity']:.6f}"
            )

            per_layout[name] = {
                "layout": layout.tolist(),
                "sensor_jacobian": jacobian_summary,
                "surrogate_score": {
                    k: v for k, v in surrogate_score.items() if k != "pair_distances"
                },
                "physical_score": {
                    k: v for k, v in physical_score.items() if k != "pair_distances"
                },
                "design_gradient": {
                    "surrogate": surrogate_gradient.tolist(),
                    "physical": physical_gradient.tolist(),
                    **gradient_metrics,
                },
            }

            np.savez_compressed(
                OUTPUT_DIR / f"pair_distances_{name}.npz",
                surrogate=surrogate_score["pair_distances"],
                physical=physical_score["pair_distances"],
                alpha_deg=DESIGN_GRID_DEG,
            )

    report["layouts"] = per_layout

    # ================================================
    # Transfer summary
    # ================================================

    base_phys = per_layout["baseline"]["physical_score"]
    star_phys = per_layout["s_star_surrogate"]["physical_score"]
    base_surr = per_layout["baseline"]["surrogate_score"]
    star_surr = per_layout["s_star_surrogate"]["surrogate_score"]

    report["transfer"] = {
        "surrogate_D_improvement": (star_surr["D"] - base_surr["D"]) / base_surr["D"],
        "physical_D_improvement": (star_phys["D"] - base_phys["D"]) / base_phys["D"],
        "surrogate_hard_min_improvement": (
            star_surr["hard_min"] - base_surr["hard_min"]
        )
        / base_surr["hard_min"],
        "physical_hard_min_improvement": (star_phys["hard_min"] - base_phys["hard_min"])
        / base_phys["hard_min"],
    }

    print()
    print("=" * 78)
    print("Surrogate-to-physics transfer")
    print("=" * 78)
    print(f"{'quantity':26s} {'surrogate':>14s} {'physical':>14s}")
    print(f"{'D baseline':26s} {base_surr['D']:14.8f} {base_phys['D']:14.8f}")
    print(f"{'D s_star':26s} {star_surr['D']:14.8f} {star_phys['D']:14.8f}")
    print(
        f"{'D improvement':26s} "
        f"{report['transfer']['surrogate_D_improvement']:13.2%} "
        f"{report['transfer']['physical_D_improvement']:13.2%}"
    )
    print(
        f"{'hard-min improvement':26s} "
        f"{report['transfer']['surrogate_hard_min_improvement']:13.2%} "
        f"{report['transfer']['physical_hard_min_improvement']:13.2%}"
    )

    (OUTPUT_DIR / "physical_transfer_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(f"\nWrote {OUTPUT_DIR / 'physical_transfer_metrics.json'}")


if __name__ == "__main__":
    main()
