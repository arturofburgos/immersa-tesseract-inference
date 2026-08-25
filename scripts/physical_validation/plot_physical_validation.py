"""Figures for the physical validation of the frozen surrogate design."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from immersa_tesseract_inference.cfd_bank import DESIGN_GRID_DEG
from immersa_tesseract_inference.sensor_design import retained_pair_mask

BASELINE_COLOR = "#c1121f"
STAR_COLOR = "#1d3557"
SURROGATE_COLOR = "#8d99ae"

ALPHA_TRUE = 63.0
DELTA_ALPHA_MIN_DEG = 7.5

VALIDATION_DIR = Path("results/sensor_design/physical_validation")
FIGURE_DIR = VALIDATION_DIR / "figures"


def save(fig: plt.Figure, stem: str) -> list[Path]:
    """Write a figure as PNG and PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    paths = []

    for suffix, kwargs in ((".png", {"dpi": 200}), (".pdf", {})):
        path = FIGURE_DIR / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)

    plt.close(fig)

    return paths


def figure_transfer(metrics: dict) -> list[Path]:
    """Figure 1: does the T4 gain survive the move to real CFD?"""
    mask = retained_pair_mask(DESIGN_GRID_DEG, DELTA_ALPHA_MIN_DEG)

    distances = {
        name: np.load(VALIDATION_DIR / f"pair_distances_{name}.npz")
        for name in ("baseline", "s_star_surrogate")
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    # Left: predicted versus realised improvement.
    ax = axes[0]

    transfer = metrics["transfer"]

    labels = [r"$D_\tau$", "hard minimum"]
    predicted = [
        100 * transfer["surrogate_D_improvement"],
        100 * transfer["surrogate_hard_min_improvement"],
    ]
    realised = [
        100 * transfer["physical_D_improvement"],
        100 * transfer["physical_hard_min_improvement"],
    ]

    positions = np.arange(len(labels))
    width = 0.36

    ax.bar(
        positions - width / 2,
        predicted,
        width,
        color=SURROGATE_COLOR,
        label="surrogate prediction",
    )
    ax.bar(
        positions + width / 2,
        realised,
        width,
        color=STAR_COLOR,
        label="real CFD",
    )

    for position, value in zip(positions - width / 2, predicted, strict=True):
        ax.text(position, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)
    for position, value in zip(positions + width / 2, realised, strict=True):
        ax.text(position, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("improvement over baseline [%]")
    ax.set_title("Predicted versus realised improvement")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6, axis="y")

    # Right: the physical lower tail T4 was trying to lift.
    ax = axes[1]

    count = 40

    for name, color, label in (
        ("baseline", BASELINE_COLOR, "baseline"),
        ("s_star_surrogate", STAR_COLOR, r"$s^\star$ (frozen)"),
    ):
        retained = distances[name]["physical"][mask]
        ax.plot(
            np.arange(1, count + 1),
            np.sort(retained)[:count],
            marker="o",
            markersize=4,
            color=color,
            label=label,
        )

    ax.set_xlabel("rank of the hardest retained pairs")
    ax.set_ylabel(r"physical $d_{ij}$")
    ax.set_title("Real-CFD pair-distance lower tail")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout()

    return save(fig, "F1_physical_transfer")


def figure_landscape() -> list[Path]:
    """Figure 2: the sealed 63 degree real-CFD inverse landscape."""
    rows = list(
        csv.DictReader((VALIDATION_DIR / "physical_landscape_63deg.csv").open())
    )

    alphas = np.array([float(r["alpha_deg"]) for r in rows])
    baseline = np.array([float(r["objective_per_scalar_baseline"]) for r in rows])
    optimized = np.array([float(r["objective_per_scalar_s_star"]) for r in rows])

    metrics = json.loads(
        (VALIDATION_DIR / "physical_landscape_metrics.json").read_text()
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    ax.semilogy(alphas, baseline, color=BASELINE_COLOR, linewidth=1.9, label="baseline")
    ax.semilogy(
        alphas, optimized, color=STAR_COLOR, linewidth=1.9, label=r"$s^\star$ (frozen)"
    )

    ax.axvline(ALPHA_TRUE, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(
        ALPHA_TRUE + 0.6,
        ax.get_ylim()[1] * 0.3,
        r"truth $63^\circ$",
        fontsize=9,
        rotation=90,
        va="top",
    )

    for name, color, marker in (
        ("baseline", BASELINE_COLOR, "v"),
        ("s_star_surrogate", STAR_COLOR, "^"),
    ):
        info = metrics["layouts"][name]
        curve = baseline if name == "baseline" else optimized

        for false_alpha in info["false_minima_deg"]:
            index = int(np.where(alphas == false_alpha)[0][0])
            ax.scatter(
                false_alpha,
                curve[index],
                marker=marker,
                s=60,
                color=color,
                zorder=5,
                edgecolor="white",
                linewidth=0.8,
            )

    ax.set_xlabel(r"candidate angle of attack $\alpha$ [deg]")
    ax.set_ylabel(r"$J(\alpha)/N$")
    ax.set_title(
        "Real-CFD inverse landscape at the sealed truth (markers: false local minima)"
    )
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(alpha=0.25, linewidth=0.6, which="both")

    return save(fig, "F2_physical_landscape_63deg")


def figure_derivatives(metrics: dict) -> list[Path]:
    """Figure 3: how well T3's sensor derivatives match the physical ones."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    for name, color, label in (
        ("baseline", BASELINE_COLOR, "baseline"),
        ("s_star_surrogate", STAR_COLOR, r"$s^\star$ (frozen)"),
    ):
        per_alpha = metrics["layouts"][name]["sensor_jacobian"]["per_alpha"]

        alphas = np.array([m["alpha_deg"] for m in per_alpha])
        cosines = np.array([m["cosine_similarity"] for m in per_alpha])
        errors = np.array([m["relative_l2_error"] for m in per_alpha])

        axes[0].plot(
            alphas, cosines, marker="o", markersize=4, color=color, label=label
        )
        axes[1].plot(alphas, errors, marker="o", markersize=4, color=color, label=label)

    axes[0].set_ylabel("cosine similarity")
    axes[0].set_title("Direction agreement, T3 versus physical")
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)

    axes[1].set_ylabel(r"relative $L_2$ error")
    axes[1].set_title("Magnitude disagreement")

    for ax in axes:
        ax.set_xlabel(r"angle of attack $\alpha$ [deg]")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)

    fig.suptitle(
        r"Sensor-position Jacobian $\nabla_{\mathbf{s}}m$: "
        "learned surrogate versus real CFD observation",
        fontsize=10,
    )

    fig.tight_layout()

    return save(fig, "F3_derivative_transfer")


def main() -> None:
    """Build all physical-validation figures."""
    metrics = json.loads(
        (VALIDATION_DIR / "physical_transfer_metrics.json").read_text()
    )

    written = []
    written += figure_transfer(metrics)
    written += figure_landscape()
    written += figure_derivatives(metrics)

    for path in written:
        print(f"Wrote {path}")

    # A compact machine-readable headline table alongside the figures.
    transfer = metrics["transfer"]
    landscape = json.loads(
        (VALIDATION_DIR / "physical_landscape_metrics.json").read_text()
    )

    with (VALIDATION_DIR / "headline_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "surrogate", "physical"])
        writer.writerow(
            [
                "D_tau improvement",
                transfer["surrogate_D_improvement"],
                transfer["physical_D_improvement"],
            ]
        )
        writer.writerow(
            [
                "hard_min improvement",
                transfer["surrogate_hard_min_improvement"],
                transfer["physical_hard_min_improvement"],
            ]
        )
        writer.writerow(
            [
                "true_to_best_false_margin",
                "",
                landscape["layouts"]["s_star_surrogate"]["true_to_best_false_margin"],
            ]
        )
        writer.writerow(
            [
                "margin_improvement_factor",
                "",
                landscape["margin_improvement_factor"],
            ]
        )

    print(f"Wrote {VALIDATION_DIR / 'headline_metrics.csv'}")


if __name__ == "__main__":
    main()
