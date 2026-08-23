from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ============================================================
# Paths
# ============================================================

LANDSCAPE_CSV = Path(
    "results/identifiability/sensor_count/"
    "nonuniform_sensor_count_landscape/"
    "sensor_count_loss_landscapes.csv"
)

MULTISTART_FILES = {
    "Ns1": Path("results/identifiability/multistart/Ns1.csv"),
    "Ns2": Path("results/identifiability/multistart/Ns2.csv"),
    "Ns3": Path("results/identifiability/multistart/Ns3.csv"),
    "Ns5": Path("results/identifiability/multistart/Ns5.csv"),
}

OUTPUT_PNG = Path("results/identifiability/final/identifiability_summary.png")

OUTPUT_PDF = Path("results/identifiability/final/identifiability_summary.pdf")


# ============================================================
# Configuration
# ============================================================

ALPHA_TRUE = 63.0

CASE_LABELS = {
    "Ns1": r"$N_s=1$",
    "Ns2": r"$N_s=2$",
    "Ns3": r"$N_s=3$",
    "Ns5": r"$N_s=5$",
}

MARKERS = {
    "Ns1": "o",
    "Ns2": "s",
    "Ns3": "^",
    "Ns5": "D",
}

# Small x offsets prevent successful points from
# sitting exactly on top of each other.
X_OFFSETS = {
    "Ns1": -0.45,
    "Ns2": -0.15,
    "Ns3": +0.15,
    "Ns5": +0.45,
}

SUCCESS_TOL_DEG = 0.1


# ============================================================
# Load data
# ============================================================

landscape = pd.read_csv(LANDSCAPE_CSV)

multistart = {case: pd.read_csv(path) for case, path in MULTISTART_FILES.items()}


# ============================================================
# Figure
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(12.0, 4.7),
    gridspec_kw={
        "width_ratios": [1.15, 1.0],
        "wspace": 0.12,
    },
)


# ============================================================
# Panel (a): inverse-objective landscapes
# ============================================================

landscape_lines = {}

for case, label in CASE_LABELS.items():
    data = landscape[landscape["case"] == case].sort_values("alpha_deg")

    (line,) = ax1.plot(
        data["alpha_deg"],
        data["objective_per_scalar"],
        marker=MARKERS[case],
        markersize=3.0,
        markevery=3,
        linewidth=1.6,
        label=label,
    )

    landscape_lines[case] = line


# True AoA
ax1.axvline(
    ALPHA_TRUE,
    linestyle="--",
    linewidth=1.35,
    color="black",
)

ax1.text(
    ALPHA_TRUE + 0.8,
    1.01,
    r"$\alpha^\star=63^\circ$",
    rotation=90,
    va="top",
    ha="left",
    fontsize=9,
)


ax1.set_xlabel(r"Angle of attack, $\alpha$ [deg]")

ax1.set_ylabel(r"Normalized objective, $J/N_{\mathrm{scalar}}$")

ax1.set_xlim(20, 85)
ax1.set_ylim(bottom=0)

ax1.grid(
    True,
    alpha=0.20,
)

ax1.legend(
    frameon=False,
    loc="upper left",
    ncol=2,
)

ax1.set_title(
    "(a) Normalized inverse-objective landscape",
    fontsize=11,
)


# ============================================================
# Panel (b): multi-start optimizer results
# ============================================================

success_counts = {}
case_colors = {}


for case, _label in CASE_LABELS.items():
    data = multistart[case].sort_values("initial_angle_deg")

    alpha0 = data["initial_angle_deg"].to_numpy() + X_OFFSETS[case]

    recovered = data["recovered_angle_deg"].to_numpy()

    success = np.abs(recovered - ALPHA_TRUE) <= SUCCESS_TOL_DEG

    success_counts[case] = int(np.count_nonzero(success))

    # Use same color as corresponding landscape curve.
    color = landscape_lines[case].get_color()
    case_colors[case] = color

    # --------------------------------------------------------
    # Successful cases: filled markers
    # --------------------------------------------------------

    ax2.scatter(
        alpha0[success],
        recovered[success],
        marker=MARKERS[case],
        s=56,
        facecolors=color,
        edgecolors=color,
        linewidths=1.0,
        zorder=4,
    )

    # --------------------------------------------------------
    # Failed cases: hollow markers
    # --------------------------------------------------------

    ax2.scatter(
        alpha0[~success],
        recovered[~success],
        marker=MARKERS[case],
        s=66,
        facecolors="none",
        edgecolors=color,
        linewidths=1.7,
        zorder=5,
    )


# ------------------------------------------------------------
# True solution
# ------------------------------------------------------------

ax2.axhline(
    ALPHA_TRUE,
    linestyle="--",
    linewidth=1.35,
    color="black",
    zorder=1,
)

ax2.text(
    35.6,
    ALPHA_TRUE + 1.1,
    r"True AoA: $\hat{\alpha}=63^\circ$",
    fontsize=8.8,
)


# ------------------------------------------------------------
# Identity line
# ------------------------------------------------------------

identity_x = np.linspace(
    33,
    82,
    100,
)

ax2.plot(
    identity_x,
    identity_x,
    linestyle=":",
    linewidth=1.0,
    color="0.55",
    zorder=1,
)


# ============================================================
# Panel (b) legends
# ============================================================

# First legend: sensor configurations + success counts
sensor_handles = []

for case in CASE_LABELS:
    sensor_handles.append(
        Line2D(
            [0],
            [0],
            marker=MARKERS[case],
            linestyle="none",
            markersize=7,
            markerfacecolor=case_colors[case],
            markeredgecolor=case_colors[case],
            label=(f"{CASE_LABELS[case]}: {success_counts[case]}/10"),
        )
    )


legend_sensors = ax2.legend(
    handles=sensor_handles,
    title="Recovered truth",
    frameon=False,
    fontsize=8.5,
    title_fontsize=8.5,
    loc="upper left",
)

ax2.add_artist(legend_sensors)


# Second legend: marker semantics
outcome_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=7,
        markerfacecolor="0.25",
        markeredgecolor="0.25",
        label="Truth recovered",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor="0.25",
        markeredgewidth=1.5,
        label="Failed recovery",
    ),
    Line2D(
        [0],
        [0],
        linestyle=":",
        linewidth=1.0,
        color="0.55",
        label=r"$\hat{\alpha}=\alpha_0$",
    ),
]

ax2.legend(
    handles=outcome_handles,
    frameon=False,
    fontsize=8.3,
    loc="lower right",
)


# ============================================================
# Panel (b) axes
# ============================================================

ax2.set_xlabel(r"Initial guess, $\alpha_0$ [deg]")

ax2.set_ylabel(r"Recovered AoA, $\hat{\alpha}$ [deg]")

ax2.set_xlim(33, 82)
ax2.set_ylim(23, 87)

ax2.set_xticks(np.arange(35, 81, 5))

ax2.set_yticks(np.arange(30, 81, 10))

ax2.grid(
    True,
    alpha=0.20,
)

ax2.set_title(
    "(b) Multi-start AoA recovery",
    fontsize=11,
)


# ============================================================
# Save
# ============================================================

fig.tight_layout()

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)

print("\nMulti-start recovery summary:")

for case in CASE_LABELS:
    print(f"  {CASE_LABELS[case]}: {success_counts[case]}/10")

print(f"\nSaved PNG: {OUTPUT_PNG}")

print(f"Saved PDF: {OUTPUT_PDF}")

plt.show()
