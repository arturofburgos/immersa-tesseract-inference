"""Known-63 inverse landscape across all four designs -- a DIAGNOSTIC.

63 degrees was opened in Phase I and now lies on the Phase-II design grid, so
this is no longer a held-out test. It is reported to show whether the refined
formulation attacks the alias Phase I diagnosed, and must not be presented as a
Phase-II generalization result. The preregistered off-grid holdouts remain
unevaluated.

The objective is the committed one:  J(alpha) = 0.5*||m(alpha) - m(63)||^2.
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

REFINED_DIR = Path("results/sensor_design/refined_design")
OUTPUT_DIR = REFINED_DIR / "known_63_diagnostic"

DESIGN_FILES = {
    "s_star_surrogate_v1": (
        Path("results/sensor_design/optimization/s_star_surrogate.json"),
        "layout_vector",
    ),
    "s_star_surrogate_v2": (
        REFINED_DIR / "surrogate_v2" / "s_star_surrogate_v2.json",
        "layout_vector",
    ),
    "s_star_cfd_refined": (
        REFINED_DIR / "physical_refinement" / "s_star_cfd_refined.json",
        "layout_vector",
    ),
}


def local_minima(values: np.ndarray) -> list[int]:
    """Grid-local minima, matching the committed study's definition."""
    return [
        i
        for i in range(1, len(values) - 1)
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]
    ]


def describe(alphas: np.ndarray, normalized: np.ndarray) -> dict:
    """Minima structure of one landscape."""
    minima = local_minima(normalized)
    true_index = int(np.argmin(np.abs(alphas - ALPHA_TRUE)))
    false_minima = [i for i in minima if abs(alphas[i] - ALPHA_TRUE) > 1.5]
    best_false = (
        min(false_minima, key=lambda i: normalized[i]) if false_minima else None
    )

    return {
        "global_minimum_alpha_deg": float(alphas[int(np.argmin(normalized))]),
        "global_minimum_is_truth": bool(
            abs(alphas[int(np.argmin(normalized))] - ALPHA_TRUE) <= 1.5
        ),
        "objective_at_truth": float(normalized[true_index]),
        "all_local_minima_deg": [float(alphas[i]) for i in minima],
        "false_minima_deg": [float(alphas[i]) for i in false_minima],
        "n_false_minima": len(false_minima),
        "best_false_minimum_alpha_deg": (
            float(alphas[best_false]) if best_false is not None else None
        ),
        "best_false_minimum_value": (
            float(normalized[best_false]) if best_false is not None else None
        ),
        "true_to_best_false_margin": (
            float(normalized[best_false] - normalized[true_index])
            if best_false is not None
            else None
        ),
    }


def main() -> None:
    """Evaluate the 63 degree landscape for all four designs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    layouts = {"baseline": BASELINE_LAYOUT}

    for name, (path, key) in DESIGN_FILES.items():
        if path.exists():
            layouts[name] = np.array(
                json.loads(path.read_text())[key], dtype=np.float64
            )
        else:
            print(f"  (skipping {name}: {path} not present)")

    curves: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    print("=" * 78)
    print("Known-63 inverse landscape DIAGNOSTIC (not a held-out test)")
    print("=" * 78)

    with Tesseract.from_image(OBSERVATION_IMAGE) as observation:
        for name, layout in layouts.items():
            sensor_x, sensor_y = unpack_layout(layout)

            truth = observe_bank(observation, load_flow(ALPHA_TRUE), sensor_x, sensor_y)

            objective = np.array(
                [
                    0.5
                    * float(
                        np.sum(
                            (
                                observe_bank(
                                    observation, load_flow(a), sensor_x, sensor_y
                                )
                                - truth
                            )
                            ** 2
                        )
                    )
                    for a in LANDSCAPE_GRID_DEG
                ]
            )

            normalized = objective / truth.size

            curves[name] = normalized
            summary[name] = describe(LANDSCAPE_GRID_DEG, normalized)
            summary[name]["layout"] = layout.tolist()

            info = summary[name]
            print(f"\n{name}:")
            print(f"  false minima       : {info['false_minima_deg']}")
            print(
                f"  best false minimum : {info['best_false_minimum_alpha_deg']} deg "
                f"at {info['best_false_minimum_value']:.8e}"
            )
            print(f"  true-to-best-false : {info['true_to_best_false_margin']:.8e}")
            print(f"  n false basins     : {info['n_false_minima']}")

    with (OUTPUT_DIR / "known63_landscapes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha_deg", *[f"J_per_scalar_{n}" for n in curves]])
        for index, alpha in enumerate(LANDSCAPE_GRID_DEG):
            writer.writerow([alpha, *[curves[n][index] for n in curves]])

    base_margin = summary["baseline"]["true_to_best_false_margin"]

    comparison = {
        "status": (
            "DIAGNOSTIC ONLY. 63 deg was opened in Phase I and lies on the "
            "Phase-II design grid, so this is not a held-out result. The "
            "preregistered off-grid holdouts remain unevaluated."
        ),
        "alpha_true_deg": ALPHA_TRUE,
        "layouts": summary,
        "margin_factor_vs_baseline": {
            name: (
                info["true_to_best_false_margin"] / base_margin if base_margin else None
            )
            for name, info in summary.items()
        },
    }

    (OUTPUT_DIR / "known63_metrics.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )

    print()
    print("margin relative to baseline:")
    for name, factor in comparison["margin_factor_vs_baseline"].items():
        print(f"  {name:22s} {factor:.3f}x")

    print(f"\nWrote {OUTPUT_DIR / 'known63_metrics.json'}")


if __name__ == "__main__":
    main()
