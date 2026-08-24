"""Publication figures for the T4 sensor-design campaign."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from immersa_tesseract_inference.sensor_design import (
    BASELINE_LAYOUT,
    DESIGN_BOUNDS,
    SensorDesignPipeline,
    retained_pair_mask,
)
from matplotlib.lines import Line2D

OPTIMIZATION_DIR = Path("results/sensor_design/optimization")
RANDOM_DIR = Path("results/sensor_design/random_baseline")
FIGURE_DIR = Path("results/sensor_design/figures")

BASELINE_COLOR = "#c1121f"
STAR_COLOR = "#1d3557"
OTHER_COLOR = "#8d99ae"
RANDOM_COLOR = "#457b9d"


def save(fig: plt.Figure, stem: str) -> list[Path]:
    """Write a figure as both PNG and PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    paths = []

    for suffix, kwargs in ((".png", {"dpi": 200}), (".pdf", {})):
        path = FIGURE_DIR / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)

    plt.close(fig)

    return paths


def plot_trajectories(summary: list[dict], best_start: str) -> list[Path]:
    """Plot A: discrimination against optimizer evaluation for every start."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for row in summary:
        name = row["start"]
        path = OPTIMIZATION_DIR / "trajectories" / f"{name}.csv"

        with path.open() as handle:
            trajectory = list(csv.DictReader(handle))

        scores = [float(r["D_tau"]) for r in trajectory]

        is_best = name == best_start

        ax.plot(
            range(len(scores)),
            scores,
            color=STAR_COLOR if is_best else OTHER_COLOR,
            linewidth=2.0 if is_best else 1.0,
            alpha=1.0 if is_best else 0.55,
            zorder=3 if is_best else 2,
            label=f"best: {name}" if is_best else None,
        )

    ax.axhline(
        float(summary[0]["D_tau_initial"])
        if summary[0]["start"] == "F01_baseline"
        else np.nan,
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=1.3,
        zorder=1,
        label="baseline layout",
    )

    ax.set_xlabel("objective / gradient evaluation")
    ax.set_ylabel(r"discrimination $D_\tau$")
    ax.set_title("Multistart optimization trajectories (20 starts)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    return save(fig, "A_optimization_trajectories")


def plot_random_distribution(
    scores: np.ndarray,
    d_baseline: float,
    d_star: float,
) -> list[Path]:
    """Plot B: where the optimized design sits among 10000 random layouts."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    ax.hist(scores, bins=80, color=RANDOM_COLOR, alpha=0.85, edgecolor="none")
    ax.set_xlabel(r"$D_\tau$")
    ax.set_ylabel("random layouts")
    ax.set_title(f"Distribution over {scores.size:,} random layouts")

    ax = axes[1]
    ordered = np.sort(scores)
    ax.plot(
        ordered,
        np.arange(1, ordered.size + 1) / ordered.size,
        color=RANDOM_COLOR,
        linewidth=1.8,
    )
    ax.set_xlabel(r"$D_\tau$")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Empirical CDF")

    for axis in axes:
        axis.axvline(
            d_baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.5, zorder=4
        )
        axis.axvline(
            scores.max(), color="#e07a5f", linestyle=":", linewidth=1.5, zorder=4
        )
        axis.axvline(d_star, color=STAR_COLOR, linewidth=2.0, zorder=5)
        axis.grid(alpha=0.25, linewidth=0.6)

    handles = [
        Line2D([], [], color=BASELINE_COLOR, linestyle="--", label="baseline"),
        Line2D([], [], color="#e07a5f", linestyle=":", label="best random"),
        Line2D([], [], color=STAR_COLOR, linewidth=2.0, label=r"$s^\star$ (gradient)"),
    ]

    axes[1].legend(handles=handles, frameon=False, fontsize=9, loc="center left")

    fig.tight_layout()

    return save(fig, "B_random_layout_distribution")


def plot_layouts(summary: list[dict], s_star: np.ndarray) -> list[Path]:
    """Plot C: where every converged layout put its two sensors."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    (x_low, x_high), (y_low, y_high) = DESIGN_BOUNDS[0], DESIGN_BOUNDS[1]

    ax.add_patch(
        plt.Rectangle(
            (x_low, y_low),
            x_high - x_low,
            y_high - y_low,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            linestyle="-",
            zorder=1,
        )
    )

    for row in summary:
        layout = [float(row[f"{c}_final"]) for c in ("x1", "y1", "x2", "y2")]
        ax.plot(
            [layout[0], layout[2]],
            [layout[1], layout[3]],
            color=OTHER_COLOR,
            linewidth=0.9,
            alpha=0.7,
            zorder=2,
        )
        ax.scatter(
            [layout[0], layout[2]],
            [layout[1], layout[3]],
            s=26,
            color=OTHER_COLOR,
            alpha=0.8,
            zorder=3,
        )

    ax.plot(
        [BASELINE_LAYOUT[0], BASELINE_LAYOUT[2]],
        [BASELINE_LAYOUT[1], BASELINE_LAYOUT[3]],
        color=BASELINE_COLOR,
        linewidth=2.0,
        zorder=4,
    )
    ax.scatter(
        [BASELINE_LAYOUT[0], BASELINE_LAYOUT[2]],
        [BASELINE_LAYOUT[1], BASELINE_LAYOUT[3]],
        s=110,
        marker="s",
        color=BASELINE_COLOR,
        zorder=5,
    )

    ax.plot(
        [s_star[0], s_star[2]],
        [s_star[1], s_star[3]],
        color=STAR_COLOR,
        linewidth=2.4,
        zorder=6,
    )
    ax.scatter(
        [s_star[0], s_star[2]],
        [s_star[1], s_star[3]],
        s=170,
        marker="*",
        color=STAR_COLOR,
        zorder=7,
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Converged sensor layouts (plate mid-chord at the origin)")
    ax.set_xlim(x_low - 0.15, x_high + 0.15)
    ax.set_ylim(y_low - 0.15, y_high + 0.15)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linewidth=0.6)

    handles = [
        Line2D(
            [], [], color=BASELINE_COLOR, marker="s", linewidth=2.0, label="baseline"
        ),
        Line2D(
            [],
            [],
            color=STAR_COLOR,
            marker="*",
            markersize=12,
            linewidth=2.4,
            label=r"$s^\star$ (gradient)",
        ),
        Line2D([], [], color=OTHER_COLOR, marker="o", label="other converged starts"),
    ]

    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")

    return save(fig, "C_sensor_layouts")


def plot_pair_distances(
    baseline_distances: np.ndarray,
    star_distances: np.ndarray,
) -> list[Path]:
    """Plot D: what the optimized layout did to the confusable pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    bins = np.linspace(
        0.0,
        max(baseline_distances.max(), star_distances.max()),
        60,
    )
    ax.hist(
        baseline_distances,
        bins=bins,
        color=BASELINE_COLOR,
        alpha=0.55,
        label="baseline",
    )
    ax.hist(
        star_distances,
        bins=bins,
        color=STAR_COLOR,
        alpha=0.55,
        label=r"$s^\star$",
    )
    ax.set_xlabel(r"normalized pair distance $d_{ij}$")
    ax.set_ylabel("retained AoA pairs")
    ax.set_title("Pair-distance distribution")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)

    ax = axes[1]
    count = 40
    ax.plot(
        np.arange(1, count + 1),
        np.sort(baseline_distances)[:count],
        marker="o",
        markersize=4,
        color=BASELINE_COLOR,
        label="baseline",
    )
    ax.plot(
        np.arange(1, count + 1),
        np.sort(star_distances)[:count],
        marker="o",
        markersize=4,
        color=STAR_COLOR,
        label=r"$s^\star$",
    )
    ax.set_xlabel("rank of the hardest retained pairs")
    ax.set_ylabel(r"$d_{ij}$")
    ax.set_title("Lower tail: the pairs that limit inference")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout()

    return save(fig, "D_pair_discrimination")


def main() -> None:
    """Build all four campaign figures."""
    with (OPTIMIZATION_DIR / "multistart_summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))

    selection = json.loads((OPTIMIZATION_DIR / "s_star_surrogate.json").read_text())
    random_summary = json.loads(
        (RANDOM_DIR / "random_baseline_summary.json").read_text()
    )

    s_star = np.array(selection["layout_vector"], dtype=np.float64)

    stored = np.load(RANDOM_DIR / "random_layout_scores.npz")

    written: list[Path] = []

    written += plot_trajectories(summary, selection["source_run"])
    written += plot_random_distribution(
        stored["scores"],
        random_summary["D_baseline"],
        random_summary["D_star_surrogate"],
    )
    written += plot_layouts(summary, s_star)

    with SensorDesignPipeline() as pipeline:
        tau = selection["tau"]
        mask = retained_pair_mask(pipeline.alpha_grid_deg, pipeline.delta_alpha_min_deg)

        baseline_distances = pipeline.discrimination(
            pipeline.measurements(BASELINE_LAYOUT), tau
        )["pair_distances"][mask]

        star_distances = pipeline.discrimination(pipeline.measurements(s_star), tau)[
            "pair_distances"
        ][mask]

    written += plot_pair_distances(baseline_distances, star_distances)

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
