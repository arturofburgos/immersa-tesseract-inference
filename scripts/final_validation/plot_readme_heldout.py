"""README figure: preregistered held-out generalization, including the failure.

Reads the frozen holdout metrics and reports the best-false-margin factor for
each preregistered truth. The 74.5 degree regression is shown, not hidden.

    python scripts/final_validation/plot_readme_heldout.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

VALIDATION_DIR = Path("results/sensor_design/final_physical_validation")
FIGURE_DIR = VALIDATION_DIR / "figures"

IMPROVED_COLOR = "#0b6e6e"
WORSENED_COLOR = "#d62828"


def main() -> None:
    """Draw the per-truth margin factors for the optimized design."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(
        (VALIDATION_DIR / "holdout_landscape_metrics.json").read_text()
    )

    truths = metrics["holdout_alphas_deg"]
    aggregate = metrics["aggregate"]["s_star_cfd_refined"]
    factors = aggregate["margin_factors"]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    colors = [IMPROVED_COLOR if f > 1.0 else WORSENED_COLOR for f in factors]
    bars = ax.bar(
        [f"{t:g}" + r"$^\circ$" for t in truths], factors, color=colors, width=0.55
    )

    for bar, factor in zip(bars, factors, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            factor + 0.045,
            f"{factor:.2f}" + r"$\times$",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=IMPROVED_COLOR if factor > 1.0 else WORSENED_COLOR,
        )

    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--")
    ax.text(
        2.52,
        1.02,
        "no change",
        fontsize=10.5,
        color="#444444",
        va="bottom",
        ha="right",
    )

    geometric = aggregate["geometric_mean_margin_factor"]
    ax.axhline(geometric, color="#555555", linewidth=1.2, linestyle=":")
    ax.text(
        -0.48,
        geometric + 0.055,
        f"geometric mean {geometric:.2f}" + r"$\times$",
        fontsize=10.5,
        color="#555555",
        va="bottom",
    )

    ax.set_ylabel(
        "best-false-minimum margin\nrelative to conventional probes", fontsize=12
    )
    ax.set_xlabel("preregistered unseen truth angle", fontsize=12.5)
    ax.set_ylim(0, max(factors) * 1.22)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.22, linewidth=0.6, axis="y")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_title(
        "Held-out generalization: two of three unseen truths improve",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    fig.text(
        0.5,
        -0.155,
        "Angles fixed before optimization and never used to tune the design. "
        r"74.5$^\circ$ got worse: global discrimination improves held-out"
        "\nperformance on average, but does not guarantee it for every "
        "off-grid truth.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    for suffix, kwargs in ((".png", {"dpi": 170}), (".pdf", {})):
        path = FIGURE_DIR / f"R5_heldout_summary{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        print(f"Wrote {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
