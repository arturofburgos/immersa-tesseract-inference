"""Final Phase-III figures: unseen truths, recovery, and the full progression."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VALIDATION_DIR = Path("results/sensor_design/final_physical_validation")
FIGURE_DIR = VALIDATION_DIR / "figures"

HOLDOUTS = (33.5, 58.5, 74.5)

COLORS = {
    "baseline": "#c1121f",
    "s_star_surrogate_v2": "#1d3557",
    "s_star_cfd_refined": "#2a9d8f",
}
LABELS = {
    "baseline": "baseline",
    "s_star_surrogate_v2": r"$s^\star$ v2 (surrogate)",
    "s_star_cfd_refined": r"$s^\star$ CFD-refined",
}


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


def figure_a(metrics: dict) -> list[Path]:
    """Three unseen physical landscapes."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), sharey=False)

    for ax, truth in zip(axes, HOLDOUTS, strict=True):
        rows = list(
            csv.DictReader((VALIDATION_DIR / f"holdout_landscape_{truth:g}.csv").open())
        )
        alphas = np.array([float(r["alpha_deg"]) for r in rows])

        for name in COLORS:
            curve = np.array([float(r[f"J_per_scalar_{name}"]) for r in rows])
            ax.semilogy(
                alphas, curve, color=COLORS[name], linewidth=1.8, label=LABELS[name]
            )

            info = metrics["truths"][str(truth)][name]
            index = int(np.where(alphas == info["best_false_minimum_alpha_deg"])[0][0])
            ax.scatter(
                alphas[index],
                curve[index],
                s=64,
                color=COLORS[name],
                edgecolor="white",
                linewidth=0.9,
                zorder=6,
            )

        ax.axvline(truth, color="black", linestyle="--", linewidth=1.3, alpha=0.8)
        ax.set_title(
            rf"unseen truth $\alpha^\star={truth}^\circ$"
            "\n"
            + "  ".join(
                f"{n.split('_')[-1]}:{metrics['truths'][str(truth)][n]['margin_factor_vs_baseline']:.2f}x"
                for n in ("s_star_surrogate_v2", "s_star_cfd_refined")
            ),
            fontsize=10,
        )
        ax.set_xlabel(r"candidate $\alpha$ [deg]")
        ax.grid(alpha=0.25, linewidth=0.6, which="both")

    axes[0].set_ylabel(r"$J(\alpha)/N$")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")

    fig.suptitle(
        "Preregistered unseen truths (markers: best false minimum)", fontsize=11
    )
    fig.tight_layout()
    return save(fig, "A_unseen_truth_landscapes")


def figure_b(summary: dict) -> list[Path]:
    """Recovery success, per start, out of the ten committed initial angles."""
    angles = summary["initial_angles_deg"]

    # Actual per-start outcomes, not just counts: the failures are specific
    # starting angles and the figure must show which ones.
    committed = {
        float(r["initial_angle_deg"]): r["basin"] == "true_basin"
        for r in csv.DictReader(
            Path("results/identifiability/multistart/Ns2.csv").open()
        )
    }
    runs = list(csv.DictReader((VALIDATION_DIR / "known63_multistart.csv").open()))

    outcomes = {"baseline": committed}
    for name in ("s_star_surrogate_v2", "s_star_cfd_refined"):
        outcomes[name] = {
            float(r["initial_angle_deg"]): r["success"] == "True"
            for r in runs
            if r["layout"] == name
        }

    names = ["baseline", "s_star_surrogate_v2", "s_star_cfd_refined"]

    fig, ax = plt.subplots(figsize=(7.8, 4.2))

    for row, name in enumerate(names):
        for column, angle in enumerate(angles):
            ok = outcomes[name][angle]
            ax.scatter(
                column,
                row,
                s=280,
                marker="o",
                color=COLORS[name] if ok else "#f1f3f5",
                edgecolor=COLORS[name] if ok else "#ced4da",
                linewidth=1.5,
                zorder=3,
            )
            if not ok:
                ax.text(
                    column,
                    row,
                    "x",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#868e96",
                    zorder=4,
                )

        ax.text(
            len(angles) + 0.35,
            row,
            f"{sum(outcomes[name].values())}/{len(angles)}",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=COLORS[name],
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS[n] for n in names], fontsize=10)
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels([f"{a:g}" for a in angles], fontsize=9)
    ax.set_xlabel("initial angle of attack [deg]")
    ax.set_xlim(-0.7, len(angles) + 1.4)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_title(
        r"Physical multistart recovery at truth $63^\circ$"
        "\n(filled = converged to the true basin)",
        fontsize=10.5,
    )
    ax.grid(alpha=0.2, linewidth=0.6, axis="x")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    return save(fig, "B_recovery_success")


def figure_c(progression: list[dict]) -> list[Path]:
    """End-to-end physical progression across the four designs."""
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))

    palette = ["#c1121f", "#8d99ae", "#1d3557", "#2a9d8f"]
    short = ["baseline", "v1", "v2", "CFD-ref"]

    panels = [
        ("physical_D_tau", r"physical $D_\tau$", "diagnostic"),
        ("physical_hard_min", "physical hard minimum", "diagnostic"),
        ("d_63_83", r"known $d(63^\circ,83^\circ)$", "diagnostic"),
    ]

    for ax, (key, title, tag) in zip(axes, panels, strict=True):
        values = [row[key] for row in progression]
        bars = ax.bar(range(len(values)), values, color=palette)
        base = values[0]
        for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
            label = f"{value:.4f}"
            if index:
                label += f"\n({(value - base) / base:+.0%})"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(short, fontsize=9)
        ax.set_title(f"{title}\n[{tag}]", fontsize=9.5)
        ax.set_ylim(0, max(values) * 1.3)
        ax.grid(alpha=0.25, linewidth=0.6, axis="y")

    fig.suptitle(
        "End-to-end physical progression  (all three panels are DIAGNOSTIC "
        "metrics on seen angles, not held-out)",
        fontsize=10,
    )
    fig.tight_layout()
    return save(fig, "C_design_progression")


def main() -> None:
    """Build the Phase-III figures."""
    metrics = json.loads(
        (VALIDATION_DIR / "holdout_landscape_metrics.json").read_text()
    )

    written = figure_a(metrics)

    summary_path = VALIDATION_DIR / "known63_multistart_summary.json"
    if summary_path.exists():
        written += figure_b(json.loads(summary_path.read_text()))

    phys = json.loads(
        Path(
            "results/sensor_design/refined_design/physical_refinement/"
            "physical_refinement_metrics.json"
        ).read_text()
    )
    ev = phys["evaluations"]
    refined = {
        k.replace("physical_", ""): v
        for k, v in phys["s_star_cfd_refined"].items()
        if k.startswith("physical_")
    }

    progression = [
        {
            "design": name,
            "physical_D_tau": src["D_tau"],
            "physical_hard_min": src["hard_min"],
            "d_63_83": src["critical_pair_distance"],
        }
        for name, src in (
            ("baseline", ev["baseline"]),
            ("s_star_surrogate_v1", ev["s_star_surrogate_v1"]),
            ("s_star_surrogate_v2", ev["s_star_surrogate_v2"]),
            ("s_star_cfd_refined", refined),
        )
    ]

    with (VALIDATION_DIR / "design_progression.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(progression[0]))
        writer.writeheader()
        writer.writerows(progression)

    written += figure_c(progression)

    for path in written:
        print(f"Wrote {path}")
    print(f"Wrote {VALIDATION_DIR / 'design_progression.csv'}")


if __name__ == "__main__":
    main()
