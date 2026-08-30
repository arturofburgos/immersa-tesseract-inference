"""README presentation figures derived from the frozen budget ablation.

Everything here reads frozen artifacts; no scientific quantity is recomputed and
no value is hard-coded that already exists in a result file. Outputs are written
alongside the frozen results they come from, in
results/sensor_budget_ablation/figures/.

    python scripts/budget_ablation/plot_readme_figures.py
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/sensor_budget_ablation")
FIGURE_DIR = RESULTS_DIR / "figures"

NAIVE_COLOR = "#d62828"
OPTIMIZED_COLOR = "#0b6e6e"


def save(fig: plt.Figure, stem: str, *, pdf: bool = True) -> list[Path]:
    """Write a figure as PNG and optionally PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written = [FIGURE_DIR / f"{stem}.png"]
    fig.savefig(written[0], dpi=170, bbox_inches="tight", facecolor="white")
    if pdf:
        written.append(FIGURE_DIR / f"{stem}.pdf")
        fig.savefig(written[-1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return written


def load_frozen() -> tuple[dict, dict]:
    """Frozen ablation summary and the per-row table."""
    summary = json.loads((RESULTS_DIR / "budget_ablation_summary.json").read_text())
    rows = {
        (int(r["n_sensors"]), r["family"]): r
        for r in csv.DictReader((RESULTS_DIR / "budget_ablation.csv").open())
    }
    return summary, rows


def plate_endpoints(
    alpha_deg: float, length: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Flat-plate endpoints, matching the solver's own convention.

    ImmersaSolver builds markers at mid_chord + s * (cos(a), sin(a)) with
    a = deg2rad(-angle_of_attack_deg), so the chord runs along (cos, -sin).
    """
    angle = np.deg2rad(-alpha_deg)
    direction = np.array([np.cos(angle), np.sin(angle)])
    return -0.5 * length * direction, 0.5 * length * direction


def budget_two_panel(summary: dict, rows: dict) -> list[Path]:
    """Per-measurement efficiency beside total discrimination."""
    counts = sorted({n for n, _ in rows})

    per_naive = [float(rows[(n, "naive")]["physical_hard_min"]) for n in counts]
    per_opt = [float(rows[(n, "optimized")]["physical_hard_min"]) for n in counts]
    scalars = [int(rows[(n, "naive")]["n_scalar"]) for n in counts]
    tot_naive = [p * s for p, s in zip(per_naive, scalars, strict=True)]
    tot_opt = [p * s for p, s in zip(per_opt, scalars, strict=True)]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    for ax, naive_series, opt_series, title, ylabel in (
        (
            axes[0],
            per_naive,
            per_opt,
            "Worst-case discrimination PER SCALAR MEASUREMENT",
            r"$\min_{(i,j)}\, d_{ij}$",
        ),
        (
            axes[1],
            tot_naive,
            tot_opt,
            "TOTAL worst-case discrimination",
            r"$\min_{(i,j)}\, d_{ij}\;\times\;10N_s$",
        ),
    ):
        ax.plot(
            counts,
            naive_series,
            marker="s",
            markersize=10,
            color=NAIVE_COLOR,
            linewidth=2.6,
            label="conventional rake",
        )
        ax.plot(
            counts,
            opt_series,
            marker="o",
            markersize=10,
            color=OPTIMIZED_COLOR,
            linewidth=2.6,
            label="optimized",
        )
        ax.set_xlabel("sensor budget  $N_s$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xticks(counts)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.25, linewidth=0.7)
        ax.legend(frameon=False, fontsize=11)

    fig.suptitle(
        "Placement beats count per measurement; count still adds total discrimination",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, "R1_budget_readme")


def layouts_readme(summary: dict) -> list[Path]:
    """What optimization discovered: a near/far split, not a denser rake."""
    counts = [1, 2, 3, 5, 8]

    fig, axes = plt.subplots(
        2, len(counts), figsize=(15.0, 5.4), sharex=True, sharey=True
    )

    for column, n_sensors in enumerate(counts):
        for row, (family, color, marker) in enumerate(
            (("naive", NAIVE_COLOR, "s"), ("optimized", OPTIMIZED_COLOR, "o"))
        ):
            ax = axes[row, column]
            layout = np.array(summary["designs"][f"Ns{n_sensors}_{family}"])

            if family == "optimized":
                ax.axvspan(1.0, 1.55, color=OPTIMIZED_COLOR, alpha=0.09, lw=0)
                ax.axvspan(2.6, 3.0, color=OPTIMIZED_COLOR, alpha=0.09, lw=0)

            ax.add_patch(
                plt.Rectangle(
                    (1.0, -1.0),
                    2.0,
                    2.0,
                    fill=False,
                    edgecolor="#999999",
                    linewidth=1.1,
                )
            )
            ax.scatter(
                layout[0::2],
                layout[1::2],
                s=95,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=1.1,
                zorder=4,
            )

            ax.set_xlim(0.8, 3.2)
            ax.set_ylim(-1.15, 1.15)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=10)
            ax.grid(alpha=0.18, linewidth=0.5)

            if row == 0:
                ax.set_title(f"$N_s={n_sensors}$", fontsize=14, fontweight="bold")
            if column == 0:
                ax.set_ylabel(
                    "conventional\n$y$" if family == "naive" else "optimized\n$y$",
                    fontsize=12,
                )
            if row == 1:
                ax.set_xlabel("$x$", fontsize=12)

    axes[1, 2].text(
        1.27,
        0.86,
        "near\nwake",
        ha="center",
        fontsize=9.5,
        color=OPTIMIZED_COLOR,
        fontweight="bold",
    )
    axes[1, 2].text(
        2.8,
        0.86,
        "far\nwake",
        ha="center",
        fontsize=9.5,
        color=OPTIMIZED_COLOR,
        fontweight="bold",
    )

    fig.suptitle(
        "As sensor budget grows, optimized layouts split sensing between "
        "near and far wake rather than densifying one cross-stream rake",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, "R2_layouts_readme")


def gradient_scaling_readme() -> list[Path]:
    """Reverse-mode cost stays flat as the design dimension grows."""
    rows = list(csv.DictReader((RESULTS_DIR / "gradient_scaling.csv").open()))

    counts = [int(r["n_sensors"]) for r in rows]
    dims = [int(r["design_dimension"]) for r in rows]
    reverse = [float(r["reverse_mode_seconds"]) for r in rows]
    finite = [float(r["central_difference_seconds"]) for r in rows]
    speedup = [float(r["speedup"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))

    ax.plot(
        dims,
        finite,
        marker="s",
        markersize=11,
        color=NAIVE_COLOR,
        linewidth=2.6,
        label="central finite differences",
        zorder=3,
    )
    ax.plot(
        dims,
        reverse,
        marker="o",
        markersize=11,
        color=OPTIMIZED_COLOR,
        linewidth=2.6,
        label="reverse mode  (T4 VJP $\\to$ T3 VJP)",
        zorder=4,
    )

    for dim, count, fast, ratio in zip(dims, counts, finite, speedup, strict=True):
        ax.annotate(
            f"$N_s$={count}\n{ratio:.1f}$\\times$",
            xy=(dim, fast),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color=NAIVE_COLOR,
            fontweight="bold",
        )

    ax.set_xlabel("number of continuous sensor-design variables  $2N_s$", fontsize=12.5)
    ax.set_ylabel("gradient evaluation time  [s]", fontsize=12.5)
    ax.set_xticks(dims)
    ax.set_ylim(-0.15, max(finite) * 1.32)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=11.5, loc="upper left")
    ax.set_title(
        "Reverse-mode cost is flat in design dimension; "
        f"{speedup[-1]:.1f}$\\times$ at {dims[-1]} variables",
        fontsize=13.5,
        fontweight="bold",
        pad=14,
    )

    return save(fig, "R3_gradient_scaling_readme")


def main() -> None:
    """Build every README figure derived from the budget ablation."""
    summary, rows = load_frozen()

    written = budget_two_panel(summary, rows)
    written += layouts_readme(summary)
    written += gradient_scaling_readme()

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
