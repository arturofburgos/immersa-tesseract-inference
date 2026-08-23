"""Plot the Tesseract-native T1 -> T2 gradient-based AoA recovery."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

# ============================================================
# Paths
# ============================================================

INPUT_CSV = Path("results/tesseract_gradient_inference/t1_t2_gradient_inference.csv")

OUTPUT_PNG = Path("results/tesseract_gradient_inference/t1_t2_gradient_inference.png")

OUTPUT_PDF = Path("results/tesseract_gradient_inference/t1_t2_gradient_inference.pdf")


# ============================================================
# Configuration
# ============================================================

ALPHA_TRUE = 63.0

TRUTH_COLOR = "#c1121f"
PATH_COLOR = "#1d3557"


# ============================================================
# Figure
# ============================================================


def main() -> None:
    """Draw AoA and loss against Gauss-Newton iteration."""
    history = pd.read_csv(INPUT_CSV)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 3.8),
    )

    # --------------------------------------------------------
    # Panel A: recovered angle of attack
    # --------------------------------------------------------

    ax = axes[0]

    ax.axhline(
        ALPHA_TRUE,
        color=TRUTH_COLOR,
        linestyle="--",
        linewidth=1.4,
        label=rf"truth $\alpha^\star={ALPHA_TRUE:.0f}^\circ$",
        zorder=1,
    )

    ax.plot(
        history["iteration"],
        history["alpha_deg"],
        marker="o",
        markersize=5,
        color=PATH_COLOR,
        linewidth=1.6,
        label="Gauss-Newton iterate",
        zorder=2,
    )

    ax.set_xlabel("Gauss-Newton iteration")
    ax.set_ylabel(r"angle of attack $\alpha$ [deg]")
    ax.set_title("Recovered angle of attack")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)

    # --------------------------------------------------------
    # Panel B: inverse objective
    # --------------------------------------------------------

    ax = axes[1]

    ax.semilogy(
        history["iteration"],
        history["loss"],
        marker="o",
        markersize=5,
        color=PATH_COLOR,
        linewidth=1.6,
        zorder=2,
    )

    ax.set_xlabel("Gauss-Newton iteration")
    ax.set_ylabel(r"$J(\alpha)=\frac{1}{2}\|m(\alpha)-m_{\rm obs}\|^2$")
    ax.set_title("Inverse objective")
    ax.grid(alpha=0.25, linewidth=0.6, which="both")

    # Iterations are integers; suppress fractional ticks.
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(
        "Gradient from the Julia ImmersaForward Tesseract, "
        "propagated through the JAX WakeObservation Tesseract",
        fontsize=10,
    )

    fig.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")

    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
