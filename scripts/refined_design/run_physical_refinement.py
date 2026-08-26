"""Mechanistic sensor design: T1 fields -> T2 -> T4, with a real gradient.

Sensors are passive, so the CFD state is independent of where the probes sit.
That makes a genuine mechanistic design loop possible with no new CFD at all:
the persisted ImmersaForward fields are held fixed, WakeObservation supplies the
entire sensor-coordinate dependence, and SensorArrayDesign supplies the global
objective.

    forward   s -> T2[T1(alpha)] -> T4 -> D_CFD
    reverse   T4 VJP -> T2 sensor VJP -> dD_CFD/ds

No derivative passes through T1 with respect to sensor position, because none
exists.

Run with --benchmark to measure one objective+gradient and stop.
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.cfd_bank import (
    BASELINE_LAYOUT,
    FLOW_KEYS,
    LANDSCAPE_GRID_DEG,
    OBSERVATION_TIMES,
    load_flow,
)
from immersa_tesseract_inference.sensor_design import (
    DESIGN_BOUNDS,
    MIN_SENSOR_DISTANCE,
    canonicalize_layout,
    effective_pair_count,
    retained_pair_mask,
    separation_penalty,
    softmin_weights,
    solve_tau_for_effective_pairs,
    unpack_layout,
)
from scipy.optimize import minimize
from tesseract_core import Tesseract

OBSERVATION_IMAGE = "immersa_tesseract_inference_wake_observation"
DESIGN_IMAGE = "immersa_tesseract_inference_sensor_array_design"

DELTA_ALPHA_MIN_DEG = 7.5
TARGET_EFFECTIVE_PAIRS = 10.0

CRITICAL_PAIR = (63.0, 83.0)

MAX_ITERATIONS = 200
F_TOLERANCE = 1.0e-9
G_TOLERANCE = 1.0e-6

TIE_TOLERANCE = 1.0e-6
CLUSTER_RADIUS = 0.05

REFINED_DIR = Path("results/sensor_design/refined_design")
OUTPUT_DIR = REFINED_DIR / "physical_refinement"

PHASE1_SELECTION = Path("results/sensor_design/optimization/s_star_surrogate.json")
PHASE2_SELECTION = REFINED_DIR / "surrogate_v2" / "s_star_surrogate_v2.json"

FIXED_STARTS = {
    "F01_baseline": [1.0, -0.4, 1.0, 0.4],
    "F02_near_wide": [1.0, -0.9, 1.0, 0.9],
    "F03_streamwise": [1.0, 0.0, 2.0, 0.0],
    "F04_mid_vertical": [2.0, -0.5, 2.0, 0.5],
    "F05_far_vertical": [3.0, -0.6, 3.0, 0.6],
    "F06_diagonal": [1.2, -0.8, 2.8, 0.8],
    "F07_antidiagonal": [1.2, 0.8, 2.8, -0.8],
    "F08_near_far_asym": [1.1, 0.2, 2.6, -0.7],
    "F09_far_close": [2.6, 0.15, 3.0, -0.25],
    "F10_centre_offaxis": [1.5, 0.0, 2.2, 0.85],
}

N_RANDOM_STARTS = 10
RANDOM_SEED = 0


def git_commit() -> str:
    """Current HEAD, marked dirty when the tree has uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{head}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class PhysicalDesignPipeline:
    """T1 bank -> T2 -> T4 design objective with a reverse-mode gradient."""

    def __init__(self, alphas: np.ndarray = LANDSCAPE_GRID_DEG) -> None:
        """Load every bank flow once; they never change during optimization."""
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.flows = [load_flow(a) for a in self.alphas]
        self.mask = retained_pair_mask(self.alphas, DELTA_ALPHA_MIN_DEG)

        self._stack = None
        self._observation = None
        self._design = None

        self.n_observation_calls = 0
        self.n_design_calls = 0

    def __enter__(self) -> "PhysicalDesignPipeline":
        """Start the observation and design Tesseracts."""
        from contextlib import ExitStack

        self._stack = ExitStack()
        self._observation = self._stack.enter_context(
            Tesseract.from_image(OBSERVATION_IMAGE)
        )
        self._design = self._stack.enter_context(Tesseract.from_image(DESIGN_IMAGE))
        return self

    def __exit__(self, *_: object) -> None:
        """Stop both Tesseracts."""
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._observation = None
        self._design = None

    def _observation_inputs(
        self,
        flow: dict,
        sensor_x: np.ndarray,
        sensor_y: np.ndarray,
    ) -> dict:
        return {
            **{key: flow[key] for key in FLOW_KEYS},
            "times": flow["times"],
            "sensor_x": sensor_x,
            "sensor_y": sensor_y,
            "sensor_times": OBSERVATION_TIMES,
        }

    def measurements(self, design: np.ndarray) -> np.ndarray:
        """Physical measurements over the AoA grid, (n_alpha, Ns, 5, 2)."""
        sensor_x, sensor_y = unpack_layout(design)

        batch = []

        for flow in self.flows:
            outputs = self._observation.apply(
                self._observation_inputs(flow, sensor_x, sensor_y)
            )
            self.n_observation_calls += 1
            batch.append(np.asarray(outputs["measurements"], dtype=np.float64))

        return np.stack(batch)

    def discrimination(self, measurements: np.ndarray, tau: float) -> dict:
        """Score a physical measurement batch with the frozen T4 functional."""
        outputs = self._design.apply(
            {
                "measurements": measurements.astype(np.float32),
                "alpha_deg": self.alphas.astype(np.float32),
                "delta_alpha_min_deg": float(DELTA_ALPHA_MIN_DEG),
                "tau": float(tau),
            }
        )
        self.n_design_calls += 1

        return {
            "discrimination": float(np.asarray(outputs["discrimination"])),
            "pair_distances": np.asarray(outputs["pair_distances"], dtype=np.float64),
            "min_pair_distance": float(np.asarray(outputs["min_pair_distance"])),
            "n_pairs": int(np.asarray(outputs["n_pairs"])),
        }

    def objective_and_gradient(
        self,
        design: np.ndarray,
        *,
        tau: float,
        lambda_separation: float,
    ) -> tuple[float, np.ndarray]:
        """Minimized physical objective and its gradient in [x1, y1, x2, y2]."""
        design = np.asarray(design, dtype=np.float64)
        sensor_x, sensor_y = unpack_layout(design)

        batch = self.measurements(design)

        scored = self.discrimination(batch, tau)

        cotangent = np.asarray(
            self._design.vector_jacobian_product(
                {
                    "measurements": batch.astype(np.float32),
                    "alpha_deg": self.alphas.astype(np.float32),
                    "delta_alpha_min_deg": float(DELTA_ALPHA_MIN_DEG),
                    "tau": float(tau),
                },
                vjp_inputs=["measurements"],
                vjp_outputs=["discrimination"],
                cotangent_vector={"discrimination": 1.0},
            )["measurements"],
            dtype=np.float64,
        )
        self.n_design_calls += 1

        grad_x = np.zeros(2)
        grad_y = np.zeros(2)

        for index, flow in enumerate(self.flows):
            outputs = self._observation.vector_jacobian_product(
                self._observation_inputs(flow, sensor_x, sensor_y),
                vjp_inputs=["sensor_x", "sensor_y"],
                vjp_outputs=["measurements"],
                cotangent_vector={"measurements": cotangent[index]},
            )
            self.n_observation_calls += 1
            grad_x += np.asarray(outputs["sensor_x"], dtype=np.float64)
            grad_y += np.asarray(outputs["sensor_y"], dtype=np.float64)

        gradient_D = np.array([grad_x[0], grad_y[0], grad_x[1], grad_y[1]])

        penalty, grad_penalty = separation_penalty(
            design, lambda_separation=lambda_separation
        )

        return float(-scored["discrimination"] + penalty), -gradient_D + grad_penalty

    def diagnostics(self, design: np.ndarray, tau: float) -> dict:
        """Full physical diagnostics for one layout."""
        scored = self.discrimination(self.measurements(design), tau)

        retained = scored["pair_distances"][self.mask]

        weights = np.sort(softmin_weights(retained, tau))[::-1]

        masked = np.where(self.mask, scored["pair_distances"], np.inf)
        i, j = np.unravel_index(np.argmin(masked), masked.shape)

        ci = int(np.where(self.alphas == CRITICAL_PAIR[0])[0][0])
        cj = int(np.where(self.alphas == CRITICAL_PAIR[1])[0][0])

        return {
            "layout": np.asarray(design, dtype=np.float64).tolist(),
            "D_tau": scored["discrimination"],
            "hard_min": scored["min_pair_distance"],
            "n_eff": effective_pair_count(retained, tau),
            "top1_weight": float(weights[0]),
            "top10_weight": float(weights[:10].sum()),
            "hardest_pair_deg": [float(self.alphas[i]), float(self.alphas[j])],
            "critical_pair_distance": float(scored["pair_distances"][ci, cj]),
            "separation": float(np.hypot(design[0] - design[2], design[1] - design[3])),
        }


def calibrate_physical_tau(pipeline: PhysicalDesignPipeline) -> dict:
    """Solve for the physical tau giving N_eff = 10 at the original baseline."""
    scored = pipeline.discrimination(pipeline.measurements(BASELINE_LAYOUT), tau=1.0)

    distances = np.sort(scored["pair_distances"][pipeline.mask])

    tau = solve_tau_for_effective_pairs(distances, target=TARGET_EFFECTIVE_PAIRS)

    median = float(np.median(distances))

    return {
        "tau": tau,
        "lambda_separation": 10.0 * median / MIN_SENSOR_DISTANCE**2,
        "effective_pairs": effective_pair_count(distances, tau),
        "median_distance": median,
        "min_distance": float(distances[0]),
        "n_pairs": scored["n_pairs"],
    }


def validate_gradient(
    pipeline: "PhysicalDesignPipeline",
    tau: float,
    lambda_separation: float,
    layout: np.ndarray,
    step: float = 3e-3,
) -> dict:
    """Physical reverse-mode gradient against central finite differences."""
    _, analytic = pipeline.objective_and_gradient(
        layout, tau=tau, lambda_separation=lambda_separation
    )

    def value(design: np.ndarray) -> float:
        scored = pipeline.discrimination(pipeline.measurements(design), tau)
        penalty, _ = separation_penalty(design, lambda_separation=lambda_separation)
        return -scored["discrimination"] + penalty

    finite = np.zeros(4)

    for index in range(4):
        plus, minus = layout.copy(), layout.copy()
        plus[index] += step
        minus[index] -= step
        finite[index] = (value(plus) - value(minus)) / (2.0 * step)

    return {
        "layout": layout.tolist(),
        "step": step,
        "analytic": analytic.tolist(),
        "finite_difference": finite.tolist(),
        "absolute_error": np.abs(analytic - finite).tolist(),
        "relative_l2_error": float(
            np.linalg.norm(analytic - finite) / np.linalg.norm(finite)
        ),
        "cosine_similarity": float(
            analytic @ finite / (np.linalg.norm(analytic) * np.linalg.norm(finite))
        ),
    }


def optimize_physical_start(
    pipeline: "PhysicalDesignPipeline",
    name: str,
    layout: list[float],
    tau: float,
    lambda_separation: float,
) -> dict[str, object]:
    """One physical L-BFGS-B run.

    Kept out of the caller's loop so the objective closure captures plain
    function locals rather than loop variables.
    """
    evaluations = {"count": 0}

    def objective(design: np.ndarray) -> tuple[float, np.ndarray]:
        evaluations["count"] += 1
        return pipeline.objective_and_gradient(
            design, tau=tau, lambda_separation=lambda_separation
        )

    wall_start = time.perf_counter()

    result = minimize(
        objective,
        np.asarray(layout, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=DESIGN_BOUNDS,
        options={
            "maxiter": MAX_ITERATIONS,
            "ftol": F_TOLERANCE,
            "gtol": G_TOLERANCE,
        },
    )

    wall = time.perf_counter() - wall_start

    canonical = canonicalize_layout(np.asarray(result.x))
    info = pipeline.diagnostics(canonical, tau)

    print(
        f"  {name:20s} D={info['D_tau']:.6f}  "
        f"d(63,83)={info['critical_pair_distance']:.6f}  "
        f"evals={evaluations['count']:3d}  {wall / 60:.1f} min  "
        f"{'OK' if result.success else 'STOP'}",
        flush=True,
    )

    return {
        "start": name,
        "x1_final": canonical[0],
        "y1_final": canonical[1],
        "x2_final": canonical[2],
        "y2_final": canonical[3],
        "D_tau_final": info["D_tau"],
        "hard_min_final": info["hard_min"],
        "critical_pair_distance": info["critical_pair_distance"],
        "hardest_pair_deg": str(info["hardest_pair_deg"]),
        "separation_final": info["separation"],
        "iterations": int(result.nit),
        "objective_gradient_evaluations": evaluations["count"],
        "converged": bool(result.success),
        "status_message": str(result.message),
        "wall_time_s": wall,
    }


def sample_random_starts(count: int, seed: int) -> dict[str, list[float]]:
    """Uniform feasible starting layouts."""
    rng = np.random.default_rng(seed)
    starts: dict[str, list[float]] = {}
    while len(starts) < count:
        x = rng.uniform(DESIGN_BOUNDS[0][0], DESIGN_BOUNDS[0][1], 2)
        y = rng.uniform(DESIGN_BOUNDS[1][0], DESIGN_BOUNDS[1][1], 2)
        if np.hypot(x[0] - x[1], y[0] - y[1]) < MIN_SENSOR_DISTANCE:
            continue
        starts[f"R{len(starts) + 1:02d}_random"] = [
            float(x[0]),
            float(y[0]),
            float(x[1]),
            float(y[1]),
        ]
    return starts


def main() -> None:
    """Evaluate the three designs, validate the gradient, then refine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Time one objective+gradient and stop before the campaign.",
    )
    parser.add_argument(
        "--max-starts",
        type=int,
        default=20,
        help="Number of starting layouts for the physical campaign.",
    )
    arguments = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v1 = np.array(
        json.loads(PHASE1_SELECTION.read_text())["layout_vector"], dtype=np.float64
    )
    v2 = np.array(
        json.loads(PHASE2_SELECTION.read_text())["layout_vector"], dtype=np.float64
    )

    print("=" * 78)
    print("Phase-II mechanistic refinement: T1 bank -> T2 -> T4")
    print("=" * 78)

    started = time.perf_counter()

    with PhysicalDesignPipeline() as pipeline:
        print(
            f"loaded {len(pipeline.flows)} bank flows, "
            f"{int(pipeline.mask.sum())} retained pairs",
            flush=True,
        )

        calibration = calibrate_physical_tau(pipeline)
        tau = calibration["tau"]
        lambda_separation = calibration["lambda_separation"]

        print(
            f"physical tau (frozen): {tau:.10f}  "
            f"N_eff={calibration['effective_pairs']:.6f}  "
            f"pairs={calibration['n_pairs']}"
        )
        print()

        # ---- evaluate the three known designs ----
        evaluations = {}
        for name, layout in (
            ("baseline", BASELINE_LAYOUT),
            ("s_star_surrogate_v1", v1),
            ("s_star_surrogate_v2", v2),
        ):
            evaluations[name] = pipeline.diagnostics(layout, tau)
            info = evaluations[name]
            print(
                f"  {name:22s} D={info['D_tau']:.8f}  "
                f"hard_min={info['hard_min']:.8f}  "
                f"d(63,83)={info['critical_pair_distance']:.8f}  "
                f"hardest={info['hardest_pair_deg']}",
                flush=True,
            )

        # ---- benchmark ----
        print()
        # Strictly inside CFD cells. WakeObservation is piecewise-multilinear
        # in sensor position with knots every h = 0.05, so a probe placed
        # exactly on a grid line has different left and right slopes and a
        # central difference there averages the two -- a step-independent
        # mismatch that is a property of the interpolant, not an error.
        interior = np.array([1.327, -0.613, 2.418, 0.694])

        t0 = time.perf_counter()
        pipeline.objective_and_gradient(
            interior, tau=tau, lambda_separation=lambda_separation
        )
        cold = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(3):
            pipeline.objective_and_gradient(
                interior, tau=tau, lambda_separation=lambda_separation
            )
        warm = (time.perf_counter() - t0) / 3

        print(f"physical objective+gradient: cold {cold:.2f} s, warm {warm:.2f} s")
        print(
            f"  -> a 20-start campaign at ~40 evaluations each would take "
            f"~{20 * 40 * warm / 60:.0f} min"
        )

        # ---- gradient validation ----
        print()
        print("Validating the physical gradient against central differences...")
        gradient_check = validate_gradient(
            pipeline, tau, lambda_separation, interior, step=1e-3
        )
        print(f"  analytic : {[round(v, 6) for v in gradient_check['analytic']]}")
        print(
            f"  finite   : {[round(v, 6) for v in gradient_check['finite_difference']]}"
        )
        print(f"  rel L2   : {gradient_check['relative_l2_error']:.6e}")
        print(f"  cosine   : {gradient_check['cosine_similarity']:.9f}")

        report = {
            "phase": "II",
            "calibration": calibration,
            "evaluations": evaluations,
            "benchmark": {
                "cold_seconds": cold,
                "warm_seconds": warm,
                "projected_20_start_minutes": 20 * 40 * warm / 60,
            },
            "gradient_validation": gradient_check,
            "git_commit": git_commit(),
        }

        if arguments.benchmark:
            (OUTPUT_DIR / "physical_benchmark.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(f"\nBenchmark only. Wrote {OUTPUT_DIR / 'physical_benchmark.json'}")
            return

        # ---- physical multistart campaign ----
        starts = dict(FIXED_STARTS)
        starts.update(sample_random_starts(N_RANDOM_STARTS, RANDOM_SEED))
        starts["V2_surrogate"] = v2.tolist()

        selected = dict(list(starts.items())[: arguments.max_starts])
        if "V2_surrogate" not in selected:
            selected["V2_surrogate"] = v2.tolist()

        print()
        print(f"Physical multistart: {len(selected)} starts")

        summaries = []

        for name, layout in selected.items():
            summaries.append(
                optimize_physical_start(pipeline, name, layout, tau, lambda_separation)
            )

        scores = np.array([s["D_tau_final"] for s in summaries])
        layouts = np.array(
            [[s[f"{c}_final"] for c in ("x1", "y1", "x2", "y2")] for s in summaries]
        )

        argmax = int(np.argmax(scores))
        tied = np.where(scores >= scores.max() - TIE_TOLERANCE)[0]
        cluster = [
            i
            for i in tied
            if np.linalg.norm(layouts[i] - layouts[argmax]) <= CLUSTER_RADIUS
        ]
        converged = [i for i in cluster if summaries[i]["converged"]]
        pool = converged if converged else cluster
        chosen = min(
            pool,
            key=lambda i: (
                summaries[i]["objective_gradient_evaluations"],
                summaries[i]["start"],
            ),
        )

        s_star_cfd = canonicalize_layout(layouts[chosen])
        final_info = pipeline.diagnostics(s_star_cfd, tau)

    elapsed = time.perf_counter() - started

    with (OUTPUT_DIR / "physical_multistart_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    frozen = {
        "s_star_cfd_refined": {
            "x1": s_star_cfd[0],
            "y1": s_star_cfd[1],
            "x2": s_star_cfd[2],
            "y2": s_star_cfd[3],
        },
        "layout_vector": s_star_cfd.tolist(),
        "phase": "II-mechanistic",
        "objective": "physical T4 discrimination from T1 bank -> T2 -> T4",
        "source_run": summaries[chosen]["start"],
        "source_converged": summaries[chosen]["converged"],
        **{f"physical_{k}": v for k, v in final_info.items()},
        "distance_to_s_star_surrogate_v2": float(np.linalg.norm(s_star_cfd - v2)),
        "distance_to_s_star_surrogate_v1": float(np.linalg.norm(s_star_cfd - v1)),
        "tau": tau,
        "delta_alpha_min_deg": DELTA_ALPHA_MIN_DEG,
        "immutability": (
            "Mechanistically refined design. Distinct from both surrogate "
            "proposals, which remain unmodified."
        ),
        "git_commit": git_commit(),
    }

    (OUTPUT_DIR / "s_star_cfd_refined.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )

    report["physical_campaign"] = {
        "n_starts": len(summaries),
        "best_D": float(scores.max()),
        "median_D": float(np.median(scores)),
        "worst_D": float(scores.min()),
        "selected": summaries[chosen]["start"],
        "wall_time_s": elapsed,
    }
    report["s_star_cfd_refined"] = frozen

    (OUTPUT_DIR / "physical_refinement_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print()
    print("=" * 78)
    print(f"s_star_cfd_refined : {s_star_cfd.tolist()}")
    print(f"  physical D_tau   : {final_info['D_tau']:.8f}")
    print(f"  d(63,83)         : {final_info['critical_pair_distance']:.8f}")
    print(f"  distance to V2   : {frozen['distance_to_s_star_surrogate_v2']:.6f}")
    print(f"  wall time        : {elapsed / 60.0:.1f} min")
    print(f"\nWrote {OUTPUT_DIR / 's_star_cfd_refined.json'}")


if __name__ == "__main__":
    main()
