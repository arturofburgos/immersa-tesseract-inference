"""Figures for the Phase-II refined design and mechanistic refinement."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REFINED_DIR = Path("results/sensor_design/refined_design")
FIGURE_DIR = REFINED_DIR / "figures"

COLORS = {
    "baseline": "#c1121f",
    "s_star_surrogate_v1": "#8d99ae",
    "s_star_surrogate_v2": "#1d3557",
    "s_star_cfd_refined": "#2a9d8f",
}
LABELS = {
    "baseline": "baseline",
    "s_star_surrogate_v1": r"$s^\star$ v1 (Phase I, 2.5$^\circ$)",
    "s_star_surrogate_v2": r"$s^\star$ v2 (Phase II, 1$^\circ$)",
    "s_star_cfd_refined": r"$s^\star$ CFD-refined",
}
ALPHA_TRUE = 63.0


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


def figure_critical_pair(metrics: dict) -> list[Path]:
    """The alias Phase I could not fix, across all four designs."""
    order = [
        "baseline",
        "s_star_surrogate_v1",
        "s_star_surrogate_v2",
        "s_star_cfd_refined",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    values = [metrics[n]["critical_pair_distance"] for n in order]
    d_values = [metrics[n]["D_tau"] for n in order]

    for ax, series, title, ylabel in (
        (axes[0], values, r"Physical $d(63^\circ, 83^\circ)$", r"$d_{ij}$"),
        (axes[1], d_values, r"Physical $D_\tau$", r"$D_\tau$"),
    ):
        bars = ax.bar(
            range(len(order)),
            series,
            color=[COLORS[n] for n in order],
        )
        base = series[0]
        for index, (bar, value) in enumerate(zip(bars, series, strict=True)):
            change = (value - base) / base
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4f}\n({change:+.0%})" if index else f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["baseline", "v1", "v2", "CFD-ref"], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(series) * 1.28)
        ax.grid(alpha=0.25, linewidth=0.6, axis="y")

    fig.suptitle(
        "Real CFD: refining the AoA design grid fixes the diagnosed alias",
        fontsize=10.5,
    )
    fig.tight_layout()
    return save(fig, "P1_critical_pair_and_score")


def figure_landscape() -> list[Path]:
    """Known-63 landscape for every design."""
    rows = list(
        csv.DictReader(
            (REFINED_DIR / "known_63_diagnostic" / "known63_landscapes.csv").open()
        )
    )
    alphas = np.array([float(r["alpha_deg"]) for r in rows])

    metrics = json.loads(
        (REFINED_DIR / "known_63_diagnostic" / "known63_metrics.json").read_text()
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    for name in COLORS:
        column = f"J_per_scalar_{name}"
        if column not in rows[0]:
            continue
        curve = np.array([float(r[column]) for r in rows])
        ax.semilogy(
            alphas,
            curve,
            color=COLORS[name],
            linewidth=1.9,
            label=LABELS[name],
        )
        info = metrics["layouts"][name]
        index = int(np.where(alphas == info["best_false_minimum_alpha_deg"])[0][0])
        ax.scatter(
            alphas[index],
            curve[index],
            s=70,
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.9,
            zorder=6,
        )

    ax.axvline(ALPHA_TRUE, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel(r"candidate angle of attack $\alpha$ [deg]")
    ax.set_ylabel(r"$J(\alpha)/N$")
    ax.set_title("Known-63 inverse landscape (diagnostic; markers: best false minimum)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25, linewidth=0.6, which="both")

    return save(fig, "P2_known63_landscape")


def figure_layouts(metrics: dict) -> list[Path]:
    """Where each design put its probes."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    ax.add_patch(
        plt.Rectangle(
            (1.0, -1.0), 2.0, 2.0, fill=False, edgecolor="black", linewidth=1.2
        )
    )

    for name, color in COLORS.items():
        layout = metrics[name]["layout"]
        ax.plot(
            [layout[0], layout[2]],
            [layout[1], layout[3]],
            color=color,
            linewidth=2.1,
            zorder=3,
        )
        ax.scatter(
            [layout[0], layout[2]],
            [layout[1], layout[3]],
            s=110,
            color=color,
            zorder=4,
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.set_xlim(0.85, 3.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Sensor layouts across both phases")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(
        handles=[
            Line2D([], [], color=c, marker="o", linewidth=2.1, label=LABELS[n])
            for n, c in COLORS.items()
        ],
        frameon=False,
        fontsize=8.5,
        loc="upper right",
    )

    return save(fig, "P3_layouts")


def main() -> None:
    """Build the Phase-II figures."""
    physical = json.loads(
        (
            REFINED_DIR / "physical_refinement" / "physical_refinement_metrics.json"
        ).read_text()
    )

    metrics = dict(physical["evaluations"])
    metrics["s_star_cfd_refined"] = {
        k.replace("physical_", ""): v
        for k, v in physical["s_star_cfd_refined"].items()
        if k.startswith("physical_")
    }
    metrics["s_star_cfd_refined"]["layout"] = physical["s_star_cfd_refined"][
        "layout_vector"
    ]

    written = []
    written += figure_critical_pair(metrics)
    written += figure_landscape()
    written += figure_layouts(metrics)

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
