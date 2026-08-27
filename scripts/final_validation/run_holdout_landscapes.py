"""Inverse landscapes at the three preregistered unseen truths.

For each sealed truth the candidate grid is the existing integer bank plus the
exact truth angle, so J(truth) = 0 is represented rather than interpolated. The
objective is the committed one, unchanged across layouts:

    J(alpha) = 0.5 * ||m(alpha) - m(truth)||^2,   J/N = J / n_scalar

This is the primary unbiased generalization result. Sensors are never moved in
response to anything found here.
"""

import csv
import json
import sys
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

sys.path.insert(0, str(Path(__file__).parent))
from build_holdout_truths import HOLDOUT_ALPHAS, load_holdout_flow

OBSERVATION_IMAGE = "immersa_tesseract_inference_wake_observation"

# A local minimum this close to the truth is the true basin, not an alias.
TRUE_BASIN_TOLERANCE_DEG = 1.5

OUTPUT_DIR = Path("results/sensor_design/final_physical_validation")

LAYOUT_FILES = {
    "s_star_surrogate_v2": Path(
        "results/sensor_design/refined_design/surrogate_v2/s_star_surrogate_v2.json"
    ),
    "s_star_cfd_refined": Path(
        "results/sensor_design/refined_design/physical_refinement/"
        "s_star_cfd_refined.json"
    ),
}


def local_minima(values: np.ndarray) -> list[int]:
    """Grid-local minima, matching the committed study's definition."""
    return [
        i
        for i in range(1, len(values) - 1)
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]
    ]


def describe(alphas: np.ndarray, normalized: np.ndarray, truth: float) -> dict:
    """Minima structure of one landscape."""
    minima = local_minima(normalized)

    true_index = int(np.argmin(np.abs(alphas - truth)))

    false_minima = [
        i for i in minima if abs(alphas[i] - truth) > TRUE_BASIN_TOLERANCE_DEG
    ]

    best_false = (
        min(false_minima, key=lambda i: normalized[i]) if false_minima else None
    )
    nearest_false = (
        min(false_minima, key=lambda i: abs(alphas[i] - truth))
        if false_minima
        else None
    )

    global_index = int(np.argmin(normalized))

    return {
        "objective_at_truth": float(normalized[true_index]),
        "global_minimum_alpha_deg": float(alphas[global_index]),
        "global_minimum_is_truth": bool(
            abs(alphas[global_index] - truth) <= TRUE_BASIN_TOLERANCE_DEG
        ),
        "all_local_minima_deg": [float(alphas[i]) for i in minima],
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
            float(normalized[best_false] - normalized[true_index])
            if best_false is not None
            else None
        ),
        "nearest_false_alias_deg": (
            float(alphas[nearest_false]) if nearest_false is not None else None
        ),
        "nearest_false_alias_value": (
            float(normalized[nearest_false]) if nearest_false is not None else None
        ),
    }


def classify(baseline: dict, candidate: dict) -> str:
    """STRONG / PARTIAL / FAILED for one truth and one layout."""
    base_margin = baseline["true_to_best_false_margin"]
    margin = candidate["true_to_best_false_margin"]

    if base_margin is None or margin is None:
        return "UNDEFINED"

    factor = margin / base_margin

    worse_ambiguity = candidate["n_false_minima"] > baseline["n_false_minima"]

    if factor < 0.95:
        return "FAILED"
    if factor >= 1.25 and not worse_ambiguity:
        return "STRONG"
    return "PARTIAL"


def main() -> None:
    """Build every unseen-truth landscape and classify the outcome."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    layouts = {"baseline": BASELINE_LAYOUT}
    for name, path in LAYOUT_FILES.items():
        layouts[name] = np.array(
            json.loads(path.read_text())["layout_vector"], dtype=np.float64
        )

    print("=" * 78)
    print("Unseen-truth physical inverse landscapes")
    print("=" * 78)
    for name, layout in layouts.items():
        print(f"  {name:22s} {[round(v, 6) for v in layout]}")
    print()

    report: dict = {
        "holdout_alphas_deg": list(HOLDOUT_ALPHAS),
        "held_out": True,
        "note": (
            "Angles preregistered before Phase-II optimization and evaluated "
            "here for the first time. No design was modified in response."
        ),
        "layouts": {k: v.tolist() for k, v in layouts.items()},
        "truths": {},
        "bank_fidelity": [],
    }

    with Tesseract.from_image(OBSERVATION_IMAGE) as observation:
        for truth in HOLDOUT_ALPHAS:
            truth_flow = load_holdout_flow(truth)

            # Candidate grid: the integer bank plus the exact truth point.
            alphas = np.unique(np.concatenate([LANDSCAPE_GRID_DEG, [truth]]).round(6))

            truth_index = int(np.where(alphas == truth)[0][0])

            print("-" * 78)
            print(f"Truth {truth} deg   ({alphas.size} candidate angles)")
            print("-" * 78)

            summary: dict[str, dict] = {}
            curves: dict[str, np.ndarray] = {}

            for name, layout in layouts.items():
                sensor_x, sensor_y = unpack_layout(layout)

                truth_measurements = observe_bank(
                    observation, truth_flow, sensor_x, sensor_y
                )

                # Persist/reload sanity check, once per layout.
                reloaded = observe_bank(
                    observation, load_holdout_flow(truth), sensor_x, sensor_y
                )
                report["bank_fidelity"].append(
                    {
                        "alpha_deg": truth,
                        "layout": name,
                        "max_absolute_difference": float(
                            np.max(np.abs(truth_measurements - reloaded))
                        ),
                    }
                )

                objective = np.empty(alphas.size)

                for index, alpha in enumerate(alphas):
                    flow = truth_flow if index == truth_index else load_flow(alpha)
                    prediction = observe_bank(observation, flow, sensor_x, sensor_y)
                    objective[index] = 0.5 * float(
                        np.sum((prediction - truth_measurements) ** 2)
                    )

                normalized = objective / truth_measurements.size

                curves[name] = normalized
                summary[name] = describe(alphas, normalized, truth)
                summary[name]["layout"] = layout.tolist()

            base = summary["baseline"]

            for name in layouts:
                info = summary[name]
                factor = (
                    info["true_to_best_false_margin"]
                    / base["true_to_best_false_margin"]
                    if base["true_to_best_false_margin"]
                    else None
                )
                info["margin_factor_vs_baseline"] = factor
                info["classification"] = (
                    "reference" if name == "baseline" else classify(base, info)
                )

                print(
                    f"  {name:22s} J(truth)={info['objective_at_truth']:.3e}  "
                    f"global={info['global_minimum_alpha_deg']:.1f} "
                    f"({'truth' if info['global_minimum_is_truth'] else 'WRONG'})  "
                    f"false={info['n_false_minima']}  "
                    f"best_false={info['best_false_minimum_alpha_deg']}deg @ "
                    f"{info['best_false_minimum_value']:.4e}  "
                    f"margin={info['true_to_best_false_margin']:.4e}"
                    + (f"  ({factor:.3f}x)  {info['classification']}" if factor else "")
                )

            with (OUTPUT_DIR / f"holdout_landscape_{truth:g}.csv").open(
                "w", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["alpha_deg", *[f"J_per_scalar_{n}" for n in curves]])
                for index, alpha in enumerate(alphas):
                    writer.writerow([alpha, *[curves[n][index] for n in curves]])

            report["truths"][str(truth)] = summary
            print()

    # ------------------------------------------------------------
    # Aggregate across the three sealed truths
    # ------------------------------------------------------------

    aggregate = {}

    for name in layouts:
        if name == "baseline":
            continue

        factors = np.array(
            [
                report["truths"][str(t)][name]["margin_factor_vs_baseline"]
                for t in HOLDOUT_ALPHAS
            ]
        )
        classifications = [
            report["truths"][str(t)][name]["classification"] for t in HOLDOUT_ALPHAS
        ]

        aggregate[name] = {
            "margin_factors": factors.tolist(),
            "mean_margin_factor": float(np.mean(factors)),
            "median_margin_factor": float(np.median(factors)),
            "worst_margin_factor": float(np.min(factors)),
            "geometric_mean_margin_factor": float(np.exp(np.mean(np.log(factors)))),
            "classifications": classifications,
            "all_improved": bool(np.all(factors > 1.0)),
            "false_minima_by_truth": [
                report["truths"][str(t)][name]["n_false_minima"] for t in HOLDOUT_ALPHAS
            ],
            "baseline_false_minima_by_truth": [
                report["truths"][str(t)]["baseline"]["n_false_minima"]
                for t in HOLDOUT_ALPHAS
            ],
        }

    report["aggregate"] = aggregate

    (OUTPUT_DIR / "holdout_landscape_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("=" * 78)
    print("Aggregate across the three sealed truths")
    print("=" * 78)
    for name, info in aggregate.items():
        print(f"  {name}")
        print(
            f"    margin factors      : {[round(f, 3) for f in info['margin_factors']]}"
        )
        print(
            f"    mean / median / worst: {info['mean_margin_factor']:.3f} / "
            f"{info['median_margin_factor']:.3f} / "
            f"{info['worst_margin_factor']:.3f}"
        )
        print(f"    geometric mean      : {info['geometric_mean_margin_factor']:.3f}")
        print(
            f"    false minima        : "
            f"{info['baseline_false_minima_by_truth']} -> "
            f"{info['false_minima_by_truth']}"
        )
        print(f"    classifications     : {info['classifications']}")
        print(f"    all three improved  : {info['all_improved']}")

    fidelity = max(r["max_absolute_difference"] for r in report["bank_fidelity"])
    print(f"\nholdout persist/reload max |difference|: {fidelity:.3e}")
    print(f"\nWrote {OUTPUT_DIR / 'holdout_landscape_metrics.json'}")


if __name__ == "__main__":
    main()
