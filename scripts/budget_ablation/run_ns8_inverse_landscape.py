"""Inverse-objective landscape for the matched Ns=8 layouts, from the frozen bank.

Evaluates J(alpha) = 0.5*||m(alpha) - m(63)||^2 for the conventional Ns=8 rake and
the frozen optimized Ns=8 array, using the existing 66-angle real-CFD bank sampled
through WakeObservation. Sensors are passive, so no new CFD is run: the persisted
fields are simply observed at each layout.

The objective, the grid-local minimum rule and the best-false-minimum definition are
the committed ones from scripts/refined_design/run_known63_diagnostic.py, so the
Ns=8 margin is comparable in kind with the two-probe diagnostic.

    python scripts/budget_ablation/run_ns8_inverse_landscape.py
"""

import csv
import json
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.cfd_bank import (
    LANDSCAPE_GRID_DEG,
    load_flow,
    observe_bank,
)
from tesseract_core import Tesseract

OBSERVATION_IMAGE = "immersa_tesseract_inference_wake_observation"
ALPHA_TRUE = 63.0

RESULTS_DIR = Path("results/sensor_budget_ablation")
SUMMARY_PATH = RESULTS_DIR / "budget_ablation_summary.json"

# The matched pair from the frozen budget ablation, identical to the layouts the
# Ns=8 multistart recovery experiment used.
LAYOUT_KEYS = {"conventional": "Ns8_naive", "optimized": "Ns8_optimized"}


def unpack(layout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split interleaved [x1, y1, ..., xN, yN] into sensor_x and sensor_y."""
    flat = np.asarray(layout, dtype=np.float64).ravel()

    return flat[0::2].copy(), flat[1::2].copy()


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
    """Evaluate both matched Ns=8 layouts on the frozen 66-angle bank."""
    designs = json.loads(SUMMARY_PATH.read_text())["designs"]
    layouts = {
        name: np.array(designs[key], dtype=np.float64)
        for name, key in LAYOUT_KEYS.items()
    }

    curves: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    print("=" * 78)
    print("Matched Ns=8 inverse landscape from the frozen bank (no new CFD)")
    print("=" * 78)

    with Tesseract.from_image(OBSERVATION_IMAGE) as observation:
        for name, layout in layouts.items():
            sensor_x, sensor_y = unpack(layout)

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
            summary[name]["n_scalar"] = int(truth.size)

            info = summary[name]
            print(f"\n{name}:")
            print(f"  false minima       : {info['false_minima_deg']}")
            print(
                f"  best false minimum : {info['best_false_minimum_alpha_deg']} deg "
                f"at {info['best_false_minimum_value']:.8e}"
            )
            print(f"  true-to-best-false : {info['true_to_best_false_margin']:.8e}")

    base = summary["conventional"]["true_to_best_false_margin"]
    factor = summary["optimized"]["true_to_best_false_margin"] / base

    csv_path = RESULTS_DIR / "ns8_inverse_landscape.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha_deg", *[f"J_per_scalar_{n}" for n in curves]])
        for index, alpha in enumerate(LANDSCAPE_GRID_DEG):
            writer.writerow([alpha, *[curves[n][index] for n in curves]])

    metrics_path = RESULTS_DIR / "ns8_inverse_landscape_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "experiment": "matched Ns=8 inverse landscape",
                "no_new_cfd": True,
                "evaluation_source": "existing 66-angle real-CFD bank",
                "objective": "J(alpha) = 0.5*||m(alpha) - m(63)||^2, per scalar",
                "status": (
                    "DIAGNOSTIC ONLY. 63 deg lies on the Phase-II design grid, so "
                    "this is not a held-out result."
                ),
                "alpha_true_deg": ALPHA_TRUE,
                "layout_source": str(SUMMARY_PATH),
                "layouts": summary,
                "margin_factor_optimized_vs_conventional": factor,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nmargin factor optimized vs conventional: {factor:.3f}x")
    print(f"Wrote {csv_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
