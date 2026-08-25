"""The sealed 63 degree real-CFD inverse landscape, baseline versus frozen design.

Reproduces the objective of the committed identifiability study exactly:

    r(alpha) = m(alpha) - m_obs
    J(alpha) = 0.5 * ||r||^2
    J/N      = J / n_scalar

with truth observations generated at 63 degrees by the same physical pipeline.
Local minima are grid-local minima of J/N, matching the committed definition.

The baseline landscape doubles as a regression check: it must reproduce the
committed Ns2 curve, since it is the same physics observed at the same probes.
"""

import csv
import json
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.cfd_bank import (
    BASELINE_LAYOUT,
    LANDSCAPE_GRID_DEG,
    load_flow,
    observe_bank,
)
from immersa_tesseract_inference.sensor_design import unpack_layout
from tesseract_core import Tesseract

OBSERVATION_IMAGE = "immersa_tesseract_inference_wake_observation"

ALPHA_TRUE = 63.0

SELECTION_JSON = Path("results/sensor_design/optimization/s_star_surrogate.json")

COMMITTED_LANDSCAPE = Path(
    "results/identifiability/sensor_count/"
    "nonuniform_sensor_count_landscape/sensor_count_loss_landscapes.csv"
)

OUTPUT_DIR = Path("results/sensor_design/physical_validation")
LANDSCAPE_CSV = OUTPUT_DIR / "physical_landscape_63deg.csv"
METRICS_JSON = OUTPUT_DIR / "physical_landscape_metrics.json"

# A false minimum counts as competitive when it sits within this factor of the
# true minimum's basin depth; used only for reporting, not for selection.
COMPETITIVE_FACTOR = 10.0


def local_minima(alphas: np.ndarray, values: np.ndarray) -> list[int]:
    """Grid-local minima, matching the committed study's definition."""
    return [
        i
        for i in range(1, len(alphas) - 1)
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]
    ]


def landscape_for(
    observation: object,
    layout: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Objective and normalized objective over the landscape grid."""
    sensor_x, sensor_y = unpack_layout(layout)

    truth = observe_bank(observation, load_flow(ALPHA_TRUE), sensor_x, sensor_y)

    objective = np.empty(LANDSCAPE_GRID_DEG.size, dtype=np.float64)

    for index, alpha in enumerate(LANDSCAPE_GRID_DEG):
        prediction = observe_bank(observation, load_flow(alpha), sensor_x, sensor_y)
        residual = prediction - truth
        objective[index] = 0.5 * float(np.sum(residual**2))

    return objective, objective / truth.size, int(truth.size)


def describe(alphas: np.ndarray, normalized: np.ndarray) -> dict:
    """Minima structure of one landscape."""
    minima = local_minima(alphas, normalized)

    true_index = int(np.argmin(np.abs(alphas - ALPHA_TRUE)))

    global_index = int(np.argmin(normalized))

    false_minima = [i for i in minima if abs(alphas[i] - ALPHA_TRUE) > 1.5]

    best_false = (
        min(false_minima, key=lambda i: normalized[i]) if false_minima else None
    )

    true_value = float(normalized[true_index])

    return {
        "global_minimum_alpha_deg": float(alphas[global_index]),
        "global_minimum_value": float(normalized[global_index]),
        "global_minimum_is_truth": bool(abs(alphas[global_index] - ALPHA_TRUE) <= 1.5),
        "objective_at_truth": true_value,
        "all_local_minima_deg": [float(alphas[i]) for i in minima],
        "all_local_minima_values": [float(normalized[i]) for i in minima],
        "false_minima_deg": [float(alphas[i]) for i in false_minima],
        "false_minima_values": [float(normalized[i]) for i in false_minima],
        "n_false_minima": len(false_minima),
        "best_false_minimum_alpha_deg": (
            float(alphas[best_false]) if best_false is not None else None
        ),
        "best_false_minimum_value": (
            float(normalized[best_false]) if best_false is not None else None
        ),
        "true_to_best_false_margin": (
            float(normalized[best_false] - true_value)
            if best_false is not None
            else None
        ),
        "n_competitive_false_minima": sum(
            1
            for i in false_minima
            if normalized[i] <= max(true_value, 1e-300) * COMPETITIVE_FACTOR
            or normalized[i] <= 0.05
        ),
    }


def main() -> None:
    """Build both landscapes and compare them."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection = json.loads(SELECTION_JSON.read_text())
    s_star = np.array(selection["layout_vector"], dtype=np.float64)

    layouts = {"baseline": BASELINE_LAYOUT, "s_star_surrogate": s_star}

    print("=" * 78)
    print(f"Real-CFD inverse landscape at the sealed truth {ALPHA_TRUE} deg")
    print("=" * 78)
    print(
        f"grid: {LANDSCAPE_GRID_DEG.size} angles, "
        f"{LANDSCAPE_GRID_DEG[0]:.0f}..{LANDSCAPE_GRID_DEG[-1]:.0f} deg"
    )
    print()

    curves: dict[str, np.ndarray] = {}
    raw_objectives: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    with Tesseract.from_image(OBSERVATION_IMAGE) as observation:
        for name, layout in layouts.items():
            objective, normalized, n_scalar = landscape_for(observation, layout)

            curves[name] = normalized
            raw_objectives[name] = objective
            summary[name] = describe(LANDSCAPE_GRID_DEG, normalized)
            summary[name]["layout"] = layout.tolist()
            summary[name]["n_scalar"] = n_scalar

            info = summary[name]

            print(f"{name}:")
            print(
                f"  global minimum      : {info['global_minimum_alpha_deg']:.1f} deg "
                f"(is truth: {info['global_minimum_is_truth']})"
            )
            print(f"  objective at truth  : {info['objective_at_truth']:.8e}")
            print(f"  local minima        : {info['all_local_minima_deg']}")
            print(f"  false minima        : {info['false_minima_deg']}")
            print(
                f"  best false minimum  : {info['best_false_minimum_alpha_deg']} deg "
                f"at {info['best_false_minimum_value']:.8e}"
            )
            print(f"  true-to-best-false  : {info['true_to_best_false_margin']:.8e}")
            print(f"  competitive false   : {info['n_competitive_false_minima']}")
            print()

    # ------------------------------------------------------------
    # Regression: the baseline must reproduce the committed curve.
    # ------------------------------------------------------------

    regression = None

    if COMMITTED_LANDSCAPE.exists():
        committed = {
            float(r["alpha_deg"]): float(r["objective_per_scalar"])
            for r in csv.DictReader(COMMITTED_LANDSCAPE.open())
            if r["case"] == "Ns2"
        }

        shared = [a for a in LANDSCAPE_GRID_DEG if float(a) in committed]

        old = np.array([committed[float(a)] for a in shared])
        new = np.array(
            [
                curves["baseline"][int(np.where(LANDSCAPE_GRID_DEG == a)[0][0])]
                for a in shared
            ]
        )

        scale = float(np.max(np.abs(old)))

        regression = {
            "n_shared_angles": len(shared),
            "max_absolute_difference": float(np.max(np.abs(new - old))),
            "max_relative_to_range": float(np.max(np.abs(new - old)) / scale),
            "correlation": float(np.corrcoef(old, new)[0, 1]),
        }

        print("-" * 78)
        print("Regression against the committed Ns2 landscape:")
        print(f"  shared angles        : {regression['n_shared_angles']}")
        print(f"  max |difference|     : {regression['max_absolute_difference']:.3e}")
        print(f"  relative to range    : {regression['max_relative_to_range']:.3e}")
        print(f"  correlation          : {regression['correlation']:.10f}")
        print()

    # ------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------

    with LANDSCAPE_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alpha_deg",
                "objective_per_scalar_baseline",
                "objective_per_scalar_s_star",
            ]
        )
        for index, alpha in enumerate(LANDSCAPE_GRID_DEG):
            writer.writerow(
                [alpha, curves["baseline"][index], curves["s_star_surrogate"][index]]
            )

    base, star = summary["baseline"], summary["s_star_surrogate"]

    comparison = {
        "alpha_true_deg": ALPHA_TRUE,
        "layouts": summary,
        "baseline_regression": regression,
        "margin_improvement_factor": (
            star["true_to_best_false_margin"] / base["true_to_best_false_margin"]
            if base["true_to_best_false_margin"]
            else None
        ),
        "false_minima_change": star["n_false_minima"] - base["n_false_minima"],
    }

    METRICS_JSON.write_text(json.dumps(comparison, indent=2) + "\n")

    print("=" * 78)
    print("Baseline versus frozen surrogate design")
    print("=" * 78)
    print(
        f"  false minima        : {base['n_false_minima']} -> {star['n_false_minima']}"
    )
    print(
        f"  best false minimum  : {base['best_false_minimum_value']:.6e} -> "
        f"{star['best_false_minimum_value']:.6e}"
    )
    print(
        f"  true-to-best-false  : {base['true_to_best_false_margin']:.6e} -> "
        f"{star['true_to_best_false_margin']:.6e}"
    )
    if comparison["margin_improvement_factor"]:
        print(f"  margin factor       : {comparison['margin_improvement_factor']:.3f}x")
    print()
    print(f"Wrote {LANDSCAPE_CSV}")
    print(f"Wrote {METRICS_JSON}")


if __name__ == "__main__":
    main()
