"""README figure: why sensor placement matters for inference.

Reads the frozen known-63 diagnostic landscapes and shows the one alias that
dominated this inverse problem. Output goes beside the frozen results it comes
from, in results/sensor_design/refined_design/figures/.

    python scripts/refined_design/plot_readme_alias.py
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DIAGNOSTIC_DIR = Path("results/sensor_design/refined_design/known_63_diagnostic")
FIGURE_DIR = Path("results/sensor_design/refined_design/figures")

ALPHA_TRUE = 63.0
ALIAS_DEG = 83.0

BASELINE_COLOR = "#d62828"
REFINED_COLOR = "#0b6e6e"


def main() -> None:
    """Draw the baseline versus CFD-refined landscape at the known truth."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((DIAGNOSTIC_DIR / "known63_landscapes.csv").open()))
    metrics = json.loads((DIAGNOSTIC_DIR / "known63_metrics.json").read_text())

    alphas = np.array([float(r["alpha_deg"]) for r in rows])
    curves = {
        "baseline": np.array([float(r["J_per_scalar_baseline"]) for r in rows]),
        "s_star_cfd_refined": np.array(
            [float(r["J_per_scalar_s_star_cfd_refined"]) for r in rows]
        ),
    }

    labels = {
        "baseline": "conventional probes",
        "s_star_cfd_refined": "optimized probes",
    }
    colors = {"baseline": BASELINE_COLOR, "s_star_cfd_refined": REFINED_COLOR}

    fig, ax = plt.subplots(figsize=(10.4, 5.2))

    for name, curve in curves.items():
        ax.semilogy(
            alphas,
            curve,
            color=colors[name],
            linewidth=2.6,
            label=labels[name],
            zorder=3,
        )

    ax.axvline(ALPHA_TRUE, color="black", linestyle="--", linewidth=1.5, alpha=0.75)
    ax.text(
        ALPHA_TRUE - 1.2,
        0.62,
        f"true angle {ALPHA_TRUE:g}" + r"$^\circ$",
        fontsize=11.5,
        va="center",
        ha="right",
        fontweight="bold",
    )

    # The alias that trapped the inverse solver.
    for name, curve in curves.items():
        info = metrics["layouts"][name]
        alias = info["best_false_minimum_alpha_deg"]
        index = int(np.where(alphas == alias)[0][0])
        ax.scatter(
            alphas[index],
            curve[index],
            s=150,
            color=colors[name],
            edgecolor="white",
            linewidth=1.6,
            zorder=6,
        )

    base_margin = metrics["layouts"]["baseline"]["true_to_best_false_margin"]
    refined_margin = metrics["layouts"]["s_star_cfd_refined"][
        "true_to_best_false_margin"
    ]
    factor = refined_margin / base_margin

    ax.annotate(
        f"best-false margin near {ALIAS_DEG:g}"
        + r"$^\circ$"
        + f"\nincreased {factor:.2f}$\\times$",
        xy=(ALIAS_DEG, base_margin),
        xytext=(64.8, 0.20),
        fontsize=11.5,
        color="#333333",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.4},
    )

    ax.set_xlabel(r"candidate angle of attack  $\alpha$  [deg]", fontsize=12.5)
    ax.set_ylabel(r"inverse objective  $J(\alpha)/N$", fontsize=12.5)
    ax.set_title(
        f"Optimized probes weaken the dangerous {ALPHA_TRUE:g}"
        + r"$^\circ \leftrightarrow$ "
        + f"{ALIAS_DEG:g}"
        + r"$^\circ$ alias",
        fontsize=14.5,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.22, linewidth=0.6, which="both")
    ax.legend(frameon=False, fontsize=12, loc="lower left")

    fig.text(
        0.5,
        -0.035,
        "Real CFD. Diagnostic case: 63"
        + r"$^\circ$"
        + " lies on the design grid, so this is not a held-out result.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    for suffix, kwargs in ((".png", {"dpi": 170}), (".pdf", {})):
        path = FIGURE_DIR / f"R4_alias_landscape_readme{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        print(f"Wrote {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
