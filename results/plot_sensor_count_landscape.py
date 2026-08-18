from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

csv_path = Path(
    "results/nonuniform_sensor_count_landscape/sensor_count_loss_landscapes.csv"
)

output_path = Path(
    "results/nonuniform_sensor_count_landscape/sensor_count_loss_landscape.png"
)

df = pd.read_csv(csv_path)

case_labels = {
    "Ns1": r"$N_s=1$",
    "Ns2": r"$N_s=2$",
    "Ns3": r"$N_s=3$",
    "Ns5": r"$N_s=5$",
}

fig, ax = plt.subplots(figsize=(8.0, 5.0))

for case, label in case_labels.items():
    data = df[df["case"] == case].sort_values("alpha_deg")

    ax.plot(
        data["alpha_deg"],
        data["objective_per_scalar"],
        marker="o",
        markersize=3,
        linewidth=1.6,
        label=label,
    )

# True AoA
ax.axvline(
    63.0,
    linestyle="--",
    linewidth=1.4,
    label=r"True AoA: $\alpha^\star=63^\circ$",
)

ax.set_xlabel(r"Angle of attack, $\alpha$ [deg]")
ax.set_ylabel(r"Normalized objective, $J/N_{\mathrm{scalar}}$")

ax.set_xlim(20, 85)

ax.legend(frameon=False)

ax.grid(
    True,
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"Saved figure to: {output_path}")
