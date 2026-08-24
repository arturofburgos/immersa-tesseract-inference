"""Select and freeze the surrogate-designed sensor array.

Kept separate from the campaign so the selection is cheap to re-run and can be
audited without repeating the optimization. Reads only surrogate optimization
diagnostics; no physical result may influence it.

Rule
----
1. Take the maximum frozen T4 discrimination over all starts.
2. Any layout within TIE_TOLERANCE of that maximum is numerically tied --
   float32 evaluation noise in WakeSurrogate is of order 1e-7, so smaller
   differences carry no information.
3. Among tied layouts belonging to the same solution cluster, prefer a normally
   converged optimizer termination over an abnormal one.
4. Break any remaining tie by the fewest objective evaluations, then by start
   name, so the outcome is deterministic.
"""

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.sensor_design import (
    ALPHA_GRID_DEG,
    DELTA_ALPHA_MIN_DEG,
    MIN_SENSOR_DISTANCE,
    SensorDesignPipeline,
    canonicalize_layout,
    effective_pair_count,
    retained_pair_mask,
    softmin_weights,
)

TIE_TOLERANCE = 1.0e-6
CLUSTER_RADIUS = 0.05

OPTIMIZATION_DIR = Path("results/sensor_design/optimization")
SUMMARY_CSV = OPTIMIZATION_DIR / "multistart_summary.csv"
CALIBRATION_JSON = OPTIMIZATION_DIR / "calibration.json"
SELECTION_JSON = OPTIMIZATION_DIR / "s_star_surrogate.json"


def git_commit() -> str:
    """Current HEAD, marked dirty when the tree has uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        return f"{head}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    """Apply the selection rule and write the frozen design."""
    rows = list(csv.DictReader(SUMMARY_CSV.open()))
    calibration = json.loads(CALIBRATION_JSON.read_text())

    tau = calibration["tau"]

    scores = np.array([float(r["D_tau_final"]) for r in rows])
    layouts = np.array(
        [[float(r[f"{c}_final"]) for c in ("x1", "y1", "x2", "y2")] for r in rows]
    )

    best_score = scores.max()

    tied = np.where(scores >= best_score - TIE_TOLERANCE)[0]

    # Restrict to the cluster containing the outright maximum.
    argmax = int(np.argmax(scores))
    same_cluster = [
        i
        for i in tied
        if np.linalg.norm(layouts[i] - layouts[argmax]) <= CLUSTER_RADIUS
    ]

    print(f"maximum D_tau           : {best_score:.12f}")
    print(f"tie tolerance           : {TIE_TOLERANCE:g}")
    print(f"numerically tied starts : {[rows[i]['start'] for i in tied]}")
    print(f"in the winning cluster  : {[rows[i]['start'] for i in same_cluster]}")
    print()

    for i in same_cluster:
        print(
            f"  {rows[i]['start']:18s} D={scores[i]:.12f}  "
            f"converged={rows[i]['converged']:5s}  "
            f"evals={rows[i]['objective_gradient_evaluations']:>3s}  "
            f"status={rows[i]['status_message']!r}"
        )

    converged = [i for i in same_cluster if rows[i]["converged"] == "True"]

    pool = converged if converged else same_cluster

    chosen = min(
        pool,
        key=lambda i: (
            int(rows[i]["objective_gradient_evaluations"]),
            rows[i]["start"],
        ),
    )

    row = rows[chosen]

    design = canonicalize_layout(layouts[chosen])

    print()
    print(
        f"selected: {row['start']}  "
        f"({'converged' if converged else 'no converged member; fell back to tied set'})"
    )

    # ------------------------------------------------------------
    # Re-evaluate the chosen layout so the recorded diagnostics come
    # from the frozen coordinates rather than from the campaign row.
    # ------------------------------------------------------------

    with SensorDesignPipeline() as pipeline:
        mask = retained_pair_mask(pipeline.alpha_grid_deg, pipeline.delta_alpha_min_deg)

        scored = pipeline.discrimination(pipeline.measurements(design), tau)

        distances = scored["pair_distances"][mask]

        weights = np.sort(softmin_weights(distances, tau))[::-1]

        n_eff = effective_pair_count(distances, tau)

    frozen = {
        "s_star_surrogate": {
            "x1": design[0],
            "y1": design[1],
            "x2": design[2],
            "y2": design[3],
        },
        "layout_vector": design.tolist(),
        "D_tau": scored["discrimination"],
        "hard_min_distance": scored["min_pair_distance"],
        "n_eff_at_s_star": n_eff,
        "top1_weight_at_s_star": float(weights[0]),
        "top5_weight_at_s_star": float(weights[:5].sum()),
        "top10_weight_at_s_star": float(weights[:10].sum()),
        "sensor_separation": float(
            np.hypot(design[0] - design[2], design[1] - design[3])
        ),
        "source_run": row["start"],
        "source_convergence_status": row["status_message"],
        "source_converged": row["converged"] == "True",
        "selection_rule": (
            "Maximum frozen T4 discrimination at delta_alpha_min = 7.5 deg. "
            "Layouts within the tie tolerance of the maximum are numerically "
            "tied; among tied layouts in the same solution cluster a normally "
            "converged optimizer termination is preferred over an abnormal "
            "one, then the fewest objective evaluations. Selection used only "
            "surrogate optimization diagnostics -- no physical or CFD result "
            "influenced it."
        ),
        "tie_tolerance": TIE_TOLERANCE,
        "cluster_radius": CLUSTER_RADIUS,
        "tied_candidates": [
            {
                "start": rows[i]["start"],
                "D_tau": float(scores[i]),
                "converged": rows[i]["converged"] == "True",
                "status": rows[i]["status_message"],
                "layout": canonicalize_layout(layouts[i]).tolist(),
            }
            for i in same_cluster
        ],
        "tau": tau,
        "tau_calibration": (
            "solved at the baseline layout (1,-0.4),(1,+0.4) so that the "
            "soft-minimum weight perplexity N_eff = 10; frozen thereafter"
        ),
        "delta_alpha_min_deg": DELTA_ALPHA_MIN_DEG,
        "min_sensor_separation": MIN_SENSOR_DISTANCE,
        "alpha_grid_deg": ALPHA_GRID_DEG.tolist(),
        "sealed_truth_excluded": {
            "angle_deg": 63.0,
            "present_in_design_grid": bool(np.any(ALPHA_GRID_DEG == 63.0)),
            "note": (
                "The sealed physical truth is not on the design grid and was "
                "never evaluated during design."
            ),
        },
        "immutability": (
            "Frozen surrogate proposal. Must not be modified in response to "
            "real-CFD performance. Any physically refined array must be "
            "recorded separately as s_star_cfd_refined."
        ),
        "git_commit": git_commit(),
    }

    SELECTION_JSON.write_text(json.dumps(frozen, indent=2) + "\n")

    print()
    print(f"  layout    : {design.tolist()}")
    print(f"  D_tau     : {frozen['D_tau']:.12f}")
    print(f"  hard_min  : {frozen['hard_min_distance']:.12f}")
    print(f"  N_eff     : {n_eff:.6f}")
    print(f"  separation: {frozen['sensor_separation']:.6f}")
    print(f"\nWrote {SELECTION_JSON}")


if __name__ == "__main__":
    main()
