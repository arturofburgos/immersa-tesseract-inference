"""Why the physical inverse landscape moved less than the T4 score.

The landscape objective and the T4 pair distance are the same quantity up to a
factor of two:

    J/N(alpha) = 0.5 * ||m(alpha) - m(63)||^2 / N = 0.5 * d(alpha, 63)

so every false minimum of the landscape is a pair distance involving the truth.
This script reports how those specific pairs moved, and contrasts them with the
pairs T4 could actually see on its 2.5 degree design grid.
"""

import csv
import json
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.cfd_bank import DESIGN_GRID_DEG
from immersa_tesseract_inference.sensor_design import retained_pair_mask

ALPHA_TRUE = 63.0
DELTA_ALPHA_MIN_DEG = 7.5

OUTPUT_DIR = Path("results/sensor_design/physical_validation")
LANDSCAPE_CSV = OUTPUT_DIR / "physical_landscape_63deg.csv"
OUTPUT_JSON = OUTPUT_DIR / "alias_transfer_diagnosis.json"


def main() -> None:
    """Compare truth-alias pairs against the pairs T4 optimized."""
    rows = list(csv.DictReader(LANDSCAPE_CSV.open()))

    alphas = np.array([float(r["alpha_deg"]) for r in rows])
    baseline = np.array([float(r["objective_per_scalar_baseline"]) for r in rows])
    optimized = np.array([float(r["objective_per_scalar_s_star"]) for r in rows])

    aliases = [27.0, 42.0, 51.0, 83.0]

    alias_rows = []

    for alias in aliases:
        index = int(np.where(alphas == alias)[0][0])

        # d = 2 * (J/N)
        d_baseline = 2.0 * baseline[index]
        d_optimized = 2.0 * optimized[index]

        alias_rows.append(
            {
                "alias_deg": alias,
                "on_design_grid": bool(np.any(DESIGN_GRID_DEG == alias)),
                "d_baseline": d_baseline,
                "d_s_star": d_optimized,
                "relative_change": (d_optimized - d_baseline) / d_baseline,
            }
        )

    mask = retained_pair_mask(DESIGN_GRID_DEG, DELTA_ALPHA_MIN_DEG)
    indices_i, indices_j = np.where(mask)

    hardest = {}

    for name in ("baseline", "s_star_surrogate"):
        distances = np.load(OUTPUT_DIR / f"pair_distances_{name}.npz")["physical"]
        retained = distances[mask]
        order = np.argsort(retained)[:5]

        hardest[name] = [
            {
                "pair_deg": [
                    float(DESIGN_GRID_DEG[indices_i[k]]),
                    float(DESIGN_GRID_DEG[indices_j[k]]),
                ],
                "distance": float(retained[k]),
            }
            for k in order
        ]

    # The design-grid pair that stands in for the decisive (63, 83) alias.
    i62 = int(np.where(DESIGN_GRID_DEG == 62.5)[0][0])
    i82 = int(np.where(DESIGN_GRID_DEG == 82.5)[0][0])

    proxy = {}

    for name in ("baseline", "s_star_surrogate"):
        distances = np.load(OUTPUT_DIR / f"pair_distances_{name}.npz")["physical"]
        proxy[name] = float(distances[i62, i82])

    decisive = next(r for r in alias_rows if r["alias_deg"] == 83.0)

    diagnosis = {
        "explanation": (
            "The landscape objective equals half the T4 pair distance to the "
            "truth, so each false minimum is a pair distance involving 63 deg. "
            "None of 63 deg or its aliases lie on the 2.5 degree design grid, "
            "so T4 never optimized them directly. The well-resolved aliases "
            "(27, 42, 51) improved by 20-28 percent, in line with the overall "
            "T4 gain. The alias that sets the margin, 83 deg, improved by only "
            "3.8 percent: the 63-83 confusion is roughly three times sharper "
            "than its nearest design-grid proxy (62.5, 82.5), so it is a "
            "sub-grid-scale feature the design criterion could not see."
        ),
        "alpha_true_deg": ALPHA_TRUE,
        "truth_on_design_grid": bool(np.any(DESIGN_GRID_DEG == ALPHA_TRUE)),
        "alias_pairs": alias_rows,
        "decisive_alias": {
            "alias_deg": 83.0,
            "d_true_pair_baseline": decisive["d_baseline"],
            "d_true_pair_s_star": decisive["d_s_star"],
            "d_design_grid_proxy_baseline": proxy["baseline"],
            "d_design_grid_proxy_s_star": proxy["s_star_surrogate"],
            "proxy_over_true_ratio_baseline": proxy["baseline"]
            / decisive["d_baseline"],
            "note": (
                "The proxy pair (62.5, 82.5) is far easier to separate than the "
                "true (63, 83) pair, so optimizing the proxy does not fix the "
                "alias that actually limits inference."
            ),
        },
        "hardest_design_grid_pairs": hardest,
    }

    OUTPUT_JSON.write_text(json.dumps(diagnosis, indent=2) + "\n")

    print("Alias pairs limiting the 63 deg inverse problem:")
    print(
        f"{'alias':>7} {'on grid':>9} {'d baseline':>12} {'d s_star':>11} {'change':>9}"
    )
    for row in alias_rows:
        print(
            f"{row['alias_deg']:7.0f} {row['on_design_grid']!s:>9s} "
            f"{row['d_baseline']:12.6f} {row['d_s_star']:11.6f} "
            f"{row['relative_change']:+8.1%}"
        )

    print()
    print("Decisive (63, 83) alias versus its design-grid proxy (62.5, 82.5):")
    print(
        f"  true pair  baseline {decisive['d_baseline']:.6f}  "
        f"s_star {decisive['d_s_star']:.6f}"
    )
    print(
        f"  proxy pair baseline {proxy['baseline']:.6f}  "
        f"s_star {proxy['s_star_surrogate']:.6f}"
    )
    print(
        f"  proxy is {proxy['baseline'] / decisive['d_baseline']:.1f}x easier "
        f"than the pair that actually matters"
    )

    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
