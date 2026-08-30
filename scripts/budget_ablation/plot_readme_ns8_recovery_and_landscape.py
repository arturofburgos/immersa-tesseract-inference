"""README figure: does the optimized Ns=8 array recover the hidden angle better?

Both panels use the same matched Ns=8 experiment: the same conventional rake, the
same frozen optimized array, the same truth of 63 degrees and the same real-CFD
bank. The left panel is multistart recovery robustness across ten initial guesses;
the right panel is the inverse-objective landscape for those same two layouts.

The two panels are complementary diagnostics. The landscape is not evidence for the
mechanism behind any individual rescued start: the optimized layout still carries
local minima near 42 and 51 degrees.

Reads frozen artifacts only; runs no CFD, optimization, training or inference.

    python scripts/budget_ablation/plot_readme_ns8_recovery_and_landscape.py
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/sensor_budget_ablation")
FIGURE_DIR = RESULTS_DIR / "figures"

ALPHA_TRUE = 63.0
ALIAS_DEG = 83.0
RESCUED_START_DEG = 40.0

# Same palette as R1 / R2 / R3.
NAIVE_COLOR = "#d62828"
OPTIMIZED_COLOR = "#0b6e6e"

LABELS = {
    "conventional": r"conventional $N_s$=8 rake",
    "optimized": r"optimized $N_s$=8 array",
}
COLORS = {"conventional": NAIVE_COLOR, "optimized": OPTIMIZED_COLOR}
# Nudge the two series apart so coincident recoveries at the truth stay legible.
OFFSETS = {"conventional": -0.85, "optimized": 0.85}


def draw_recovery(ax: plt.Axes) -> None:
    """Recovered angle against initial guess, for both matched layouts."""
    rows = list(csv.DictReader((RESULTS_DIR / "ns8_multistart_recovery.csv").open()))

    runs: dict[str, list[tuple[float, float, bool]]] = {}
    for row in rows:
        runs.setdefault(row["layout"], []).append(
            (
                float(row["initial_angle_deg"]),
                float(row["recovered_angle_deg"]),
                row["success"] == "True",
            )
        )

    ax.axhline(ALPHA_TRUE, color="black", linestyle="--", linewidth=1.4, alpha=0.75)
    ax.text(
        34.0,
        ALPHA_TRUE + 1.6,
        f"truth = {ALPHA_TRUE:g}" + r"$^\circ$",
        fontsize=10.5,
        color="#333333",
    )

    for name in ("conventional", "optimized"):
        entries = sorted(runs[name])
        successes = sum(1 for _, _, ok in entries if ok)

        for start, recovered, ok in entries:
            ax.scatter(
                start + OFFSETS[name],
                recovered,
                s=110,
                marker="o",
                color=COLORS[name] if ok else "white",
                edgecolor=COLORS[name],
                linewidth=2.0,
                zorder=4,
            )

        # One invisible proxy point carries the legend entry and the tally.
        ax.scatter(
            [],
            [],
            s=110,
            marker="o",
            color=COLORS[name],
            edgecolor=COLORS[name],
            linewidth=2.0,
            label=f"{LABELS[name]}   {successes}/{len(entries)}",
        )

    rescued = {
        name: next(r for s, r, _ in sorted(runs[name]) if s == RESCUED_START_DEG)
        for name in ("conventional", "optimized")
    }
    ax.annotate(
        r"$\alpha_0 = 40^\circ$ rescued:"
        "\n"
        f"{rescued['conventional']:.1f}" + r"$^\circ \rightarrow$ "
        f"{rescued['optimized']:.1f}" + r"$^\circ$",
        xy=(RESCUED_START_DEG + OFFSETS["conventional"], rescued["conventional"]),
        xytext=(45.5, 42.0),
        fontsize=10.5,
        color="#333333",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.3},
    )

    ax.text(
        0.985,
        0.055,
        "filled = recovered the truth\nopen = converged to a false basin",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#666666",
        ha="right",
        va="bottom",
    )

    ax.set_xlabel(r"initial guess  $\alpha_0$  [deg]", fontsize=11.5)
    ax.set_ylabel(r"recovered  $\alpha$  [deg]", fontsize=11.5)
    ax.set_title(
        r"(A) Multistart AoA recovery — matched $N_s$=8 budget",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")


def draw_landscape(ax: plt.Axes) -> None:
    """Inverse objective for the same two matched Ns=8 layouts."""
    rows = list(csv.DictReader((RESULTS_DIR / "ns8_inverse_landscape.csv").open()))
    metrics_path = RESULTS_DIR / "ns8_inverse_landscape_metrics.json"
    metrics = json.loads(metrics_path.read_text())

    alphas = np.array([float(r["alpha_deg"]) for r in rows])
    curves = {
        name: np.array([float(r[f"J_per_scalar_{name}"]) for r in rows])
        for name in ("conventional", "optimized")
    }

    for name, curve in curves.items():
        ax.semilogy(
            alphas,
            curve,
            color=COLORS[name],
            linewidth=2.4,
            label=LABELS[name],
            zorder=3,
        )

    ax.axvline(ALPHA_TRUE, color="black", linestyle="--", linewidth=1.4, alpha=0.75)
    ax.text(
        ALPHA_TRUE - 1.2,
        0.55,
        f"true angle {ALPHA_TRUE:g}" + r"$^\circ$",
        fontsize=10.5,
        va="center",
        ha="right",
        fontweight="bold",
    )

    # The strongest false minimum, which is the 83 degree alias for both layouts.
    for name, curve in curves.items():
        alias = metrics["layouts"][name]["best_false_minimum_alpha_deg"]
        index = int(np.where(alphas == alias)[0][0])
        ax.scatter(
            alphas[index],
            curve[index],
            s=130,
            color=COLORS[name],
            edgecolor="white",
            linewidth=1.6,
            zorder=6,
        )

    factor = metrics["margin_factor_optimized_vs_conventional"]
    ax.annotate(
        f"best-false margin near {ALIAS_DEG:g}"
        + r"$^\circ$"
        + f"\nincreased {factor:.2f}$\\times$",
        xy=(ALIAS_DEG, metrics["layouts"]["conventional"]["best_false_minimum_value"]),
        xytext=(68.5, 0.0032),
        fontsize=10.5,
        color="#333333",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.3},
    )

    # A little headroom keeps the peaks clear of the panel title.
    ax.set_ylim(top=2.6)
    ax.set_xlabel(r"candidate angle of attack  $\alpha$  [deg]", fontsize=11.5)
    ax.set_ylabel(r"inverse objective  $J(\alpha)/N$", fontsize=11.5)
    ax.set_title(
        r"(B) Inverse landscape — same $N_s$=8 layouts",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.22, linewidth=0.6, which="both")
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")


def main() -> None:
    """Compose recovery robustness beside false-solution separation."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.2), width_ratios=[1.0, 1.05])
    draw_recovery(axes[0])
    draw_landscape(axes[1])

    fig.suptitle(
        "Optimized sensing improves hidden-AoA recovery and false-solution separation",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(w_pad=3.0)
    fig.text(
        0.5,
        -0.045,
        "Real CFD  ·  truth "
        + r"$\alpha = 63^\circ$"
        + "  ·  same conventional and optimized "
        + r"$N_s$=8"
        + " layouts in both panels.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    for suffix, kwargs in ((".png", {"dpi": 170}), (".pdf", {})):
        path = FIGURE_DIR / f"R6_ns8_recovery_and_landscape_readme{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        print(f"Wrote {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
