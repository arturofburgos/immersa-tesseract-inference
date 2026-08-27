"""Figures for the sensor budget versus placement ablation."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/sensor_budget_ablation")
FIGURE_DIR = RESULTS_DIR / "figures"

NAIVE_COLOR = "#c1121f"
OPTIMIZED_COLOR = "#1d3557"


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


def load_rows() -> dict:
    """Index the ablation table by (budget, family)."""
    rows = list(csv.DictReader((RESULTS_DIR / "budget_ablation.csv").open()))
    return {(int(r["n_sensors"]), r["family"]): r for r in rows}


def figure_budget_curve(lookup: dict, counts: list[int]) -> list[Path]:
    """The headline: discrimination against budget for both families."""
    naive = [float(lookup[(n, "naive")]["physical_hard_min"]) for n in counts]
    optimized = [float(lookup[(n, "optimized")]["physical_hard_min"]) for n in counts]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))

    ax = axes[0]
    ax.plot(
        counts,
        naive,
        marker="s",
        markersize=8,
        color=NAIVE_COLOR,
        linewidth=2.0,
        label="naive vertical probes",
    )
    ax.plot(
        counts,
        optimized,
        marker="o",
        markersize=8,
        color=OPTIMIZED_COLOR,
        linewidth=2.0,
        label="differentiably optimized",
    )

    # The prespecified comparison: does Ns=2 optimized clear the naive curve?
    two_opt = optimized[counts.index(2)]
    ax.axhline(two_opt, color=OPTIMIZED_COLOR, linestyle=":", linewidth=1.4, alpha=0.8)
    ax.annotate(
        r"$N_s=2$ optimized",
        xy=(counts[-1], two_opt),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=OPTIMIZED_COLOR,
    )

    ax.set_xlabel("sensor budget $N_s$")
    ax.set_ylabel("real-CFD hard minimum  " r"$\min_{(i,j)} d_{ij}$")
    ax.set_title("Global discrimination against budget")
    ax.set_xticks(counts)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.6)

    # Right: how many naive probes an optimized pair is worth.
    ax = axes[1]
    gains = [(optimized[i] - naive[i]) / naive[i] * 100 for i in range(len(counts))]
    bars = ax.bar([str(n) for n in counts], gains, color=OPTIMIZED_COLOR)
    for bar, gain in zip(bars, gains, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            gain,
            f"{gain:+.0f}%",
            ha="center",
            va="bottom" if gain >= 0 else "top",
            fontsize=9,
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlabel("sensor budget $N_s$")
    ax.set_ylabel("placement gain over naive [%]")
    ax.set_title("What placement buys at each budget")
    ax.grid(alpha=0.25, linewidth=0.6, axis="y")

    fig.suptitle(
        "Real CFD: sensor placement versus sensor count "
        "(no new CFD, same 66-angle bank)",
        fontsize=10.5,
    )
    fig.tight_layout()
    return save(fig, "S1_budget_versus_placement")


def figure_layouts(designs: dict, counts: list[int]) -> list[Path]:
    """Where each family puts its probes."""
    fig, axes = plt.subplots(
        2, len(counts), figsize=(3.0 * len(counts), 5.6), sharex=True, sharey=True
    )

    for column, n_sensors in enumerate(counts):
        for row, (family, color) in enumerate(
            (("naive", NAIVE_COLOR), ("optimized", OPTIMIZED_COLOR))
        ):
            ax = axes[row, column]
            layout = np.array(designs[f"Ns{n_sensors}_{family}"])
            ax.add_patch(
                plt.Rectangle(
                    (1.0, -1.0), 2.0, 2.0, fill=False, edgecolor="black", linewidth=1.0
                )
            )
            ax.scatter(
                layout[0::2],
                layout[1::2],
                s=70,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            ax.set_xlim(0.85, 3.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2, linewidth=0.5)
            if row == 0:
                ax.set_title(f"$N_s={n_sensors}$", fontsize=11)
            if column == 0:
                ax.set_ylabel(f"{family}\n$y$", fontsize=10)
            if row == 1:
                ax.set_xlabel("$x$")

    fig.suptitle(
        "Probe layouts: conventional vertical array versus optimized", fontsize=11
    )
    fig.tight_layout()
    return save(fig, "S2_layouts")


def figure_gradient_scaling() -> list[Path]:
    """Reverse-mode gradient cost against a central finite difference."""
    rows = list(csv.DictReader((RESULTS_DIR / "gradient_scaling.csv").open()))

    counts = [int(r["n_sensors"]) for r in rows]
    dims = [int(r["design_dimension"]) for r in rows]
    reverse = [float(r["reverse_mode_seconds"]) for r in rows]
    finite = [float(r["central_difference_seconds"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))

    ax.plot(
        dims,
        reverse,
        marker="o",
        markersize=8,
        color=OPTIMIZED_COLOR,
        linewidth=2.0,
        label="reverse mode (T4 VJP -> T3 VJP)",
    )
    ax.plot(
        dims,
        finite,
        marker="s",
        markersize=8,
        color=NAIVE_COLOR,
        linewidth=2.0,
        label="central finite differences",
    )

    for dim, r, f, n in zip(dims, reverse, finite, counts, strict=True):
        ax.annotate(
            f"$N_s$={n}\n{f / r:.0f}x",
            xy=(dim, f),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=NAIVE_COLOR,
        )

    ax.set_xlabel("design variables  $2N_s$")
    ax.set_ylabel("seconds per gradient")
    ax.set_xticks(dims)
    ax.set_title("Gradient cost: reverse mode is independent of design dimension")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.6)

    return save(fig, "S3_gradient_scaling")


def main() -> None:
    """Build the ablation figures."""
    summary = json.loads((RESULTS_DIR / "budget_ablation_summary.json").read_text())
    lookup = load_rows()
    counts = sorted({n for n, _ in lookup})

    written = figure_budget_curve(lookup, counts)
    written += figure_layouts(summary["designs"], counts)
    written += figure_gradient_scaling()

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
