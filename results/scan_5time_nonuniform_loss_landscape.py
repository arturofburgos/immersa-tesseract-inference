"""Noise-free AoA loss-landscape scan for sparse temporal observations."""

import csv
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.pipeline import ForwardObservationPipeline

# ============================================================
# Configuration
# ============================================================

ALPHA_TRUE = 63.0

ALPHAS = np.arange(
    35.0,
    80.0 + 1.0,
    1.0,
    dtype=np.float64,
)

MAX_WORKERS = 5

SENSOR_TIMES = np.array(
    [12.0, 13.3, 15.1, 17.4, 20.0],
    dtype=np.float64,
)

SENSOR_CASES = {
    "center": {
        "x": np.array([1.0], dtype=np.float64),
        "y": np.array([0.0], dtype=np.float64),
    },
    "off_center": {
        "x": np.array([1.0], dtype=np.float64),
        "y": np.array([-1.0], dtype=np.float64),
    },
}

# CFD settings
H = 0.05
DT = 0.0025
TF = 20.0
RE = 200.0
SNAPSHOT_FREQ = 40

OUTPUT_DIR = Path("results/5time_nonuniform_landscape")


# ============================================================
# Worker
# ============================================================


def evaluate_chunk(
    worker_id: int,
    alphas: list[float],
    truth_center: np.ndarray,
    truth_off_center: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """Evaluate one chunk of AoAs with one persistent pipeline."""
    rows = []

    truth = {
        "center": truth_center,
        "off_center": truth_off_center,
    }

    log_path = OUTPUT_DIR / f"worker_{worker_id}.log"

    with log_path.open("w", buffering=1) as log:
        print(
            f"Worker {worker_id}: {len(alphas)} AoAs",
            file=log,
            flush=True,
        )

        # No forward states are reused in this scan,
        # so keep the cache tiny.
        with ForwardObservationPipeline(max_cached_flows=1) as pipeline:
            for index, alpha in enumerate(alphas, start=1):
                start = time.perf_counter()

                print(
                    f"[{index}/{len(alphas)}] F(alpha={alpha:.1f} deg)",
                    file=log,
                    flush=True,
                )

                flow = pipeline.run_forward(
                    angle_of_attack_deg=alpha,
                    h=H,
                    dt=DT,
                    tf=TF,
                    Re=RE,
                    snapshot_freq=SNAPSHOT_FREQ,
                )

                for case_name, case in SENSOR_CASES.items():
                    prediction = pipeline.observe(
                        flow,
                        case["x"],
                        case["y"],
                        SENSOR_TIMES,
                    )

                    residual = prediction - truth[case_name]

                    objective = 0.5 * float(np.sum(residual**2))

                    n_scalar = residual.size

                    rows.append(
                        {
                            "alpha_deg": float(alpha),
                            "case": case_name,
                            "x": float(case["x"][0]),
                            "y": float(case["y"][0]),
                            "objective": objective,
                            "objective_per_scalar": (objective / n_scalar),
                            "residual_rms": float(np.sqrt(np.mean(residual**2))),
                            "worker": worker_id,
                        }
                    )

                elapsed = time.perf_counter() - start

                print(
                    f"    done in {elapsed / 60.0:.2f} min",
                    file=log,
                    flush=True,
                )

    return rows


# ============================================================
# Local-minimum helper
# ============================================================


def find_local_minima(
    rows: list[dict[str, float | int | str]],
    case_name: str,
) -> list[dict[str, float | int | str]]:
    """Find grid-local minima for one sensor case."""
    case_rows = sorted(
        [row for row in rows if row["case"] == case_name],
        key=lambda row: float(row["alpha_deg"]),
    )

    minima = []

    for i in range(1, len(case_rows) - 1):
        j_prev = float(case_rows[i - 1]["objective"])

        j_here = float(case_rows[i]["objective"])

        j_next = float(case_rows[i + 1]["objective"])

        if j_here <= j_prev and j_here <= j_next:
            minima.append(case_rows[i])

    return minima


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Run the nonuniform five-time AoA loss-landscape scan."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_start = time.perf_counter()

    print("=" * 76, flush=True)
    print(
        "5-TIME NONUNIFORM NOISE-FREE AoA LOSS LANDSCAPE",
        flush=True,
    )
    print("=" * 76, flush=True)

    print(f"truth AoA      = {ALPHA_TRUE:.1f} deg")
    print(f"AoA scan       = {ALPHAS[0]:.1f} ... {ALPHAS[-1]:.1f} deg")
    print(f"number of AoAs = {len(ALPHAS)}")
    print(f"workers        = {MAX_WORKERS}")
    print(f"sensor times   = {SENSOR_TIMES.tolist()}")

    # ========================================================
    # Compute exact clean truth observations ONCE
    # ========================================================

    print(
        "\nComputing clean truth flow...",
        flush=True,
    )

    with ForwardObservationPipeline(max_cached_flows=1) as pipeline:
        truth_flow = pipeline.run_forward(
            angle_of_attack_deg=ALPHA_TRUE,
            h=H,
            dt=DT,
            tf=TF,
            Re=RE,
            snapshot_freq=SNAPSHOT_FREQ,
        )

        truth_center = pipeline.observe(
            truth_flow,
            SENSOR_CASES["center"]["x"],
            SENSOR_CASES["center"]["y"],
            SENSOR_TIMES,
        )

        truth_off_center = pipeline.observe(
            truth_flow,
            SENSOR_CASES["off_center"]["x"],
            SENSOR_CASES["off_center"]["y"],
            SENSOR_TIMES,
        )

    print(
        "Truth observations complete.",
        flush=True,
    )

    # ========================================================
    # Split AoAs across workers
    # ========================================================

    chunks = [
        chunk.tolist()
        for chunk in np.array_split(
            ALPHAS,
            MAX_WORKERS,
        )
        if len(chunk) > 0
    ]

    print("\nWorker chunks:", flush=True)

    for worker_id, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"  worker {worker_id}: "
            f"{chunk[0]:.1f} -> "
            f"{chunk[-1]:.1f} deg "
            f"({len(chunk)} cases)",
            flush=True,
        )

    print(
        "\nLaunching landscape scan...\n",
        flush=True,
    )

    all_rows = []

    context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=context,
    ) as executor:
        futures = {}

        for worker_id, chunk in enumerate(
            chunks,
            start=1,
        ):
            future = executor.submit(
                evaluate_chunk,
                worker_id,
                chunk,
                truth_center,
                truth_off_center,
            )

            futures[future] = worker_id

        for future in as_completed(futures):
            worker_id = futures[future]

            rows = future.result()

            all_rows.extend(rows)

            elapsed = time.perf_counter() - total_start

            print(
                f"[DONE] worker {worker_id} | elapsed {elapsed / 60.0:.2f} min",
                flush=True,
            )

    # ========================================================
    # Sort + save CSV
    # ========================================================

    case_order = {
        "center": 0,
        "off_center": 1,
    }

    all_rows.sort(
        key=lambda row: (
            float(row["alpha_deg"]),
            case_order[str(row["case"])],
        )
    )

    csv_path = OUTPUT_DIR / "loss_landscape_5times_nonuniform.csv"

    fieldnames = list(all_rows[0].keys())

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_rows)

    # ========================================================
    # Summary
    # ========================================================

    print("\n\n" + "=" * 76)
    print("LOSS-LANDSCAPE SUMMARY")
    print("=" * 76)

    for case_name in [
        "center",
        "off_center",
    ]:
        print(f"\nSensor: {case_name}")

        case_rows = [row for row in all_rows if row["case"] == case_name]

        # Five smallest objective values
        lowest = sorted(
            case_rows,
            key=lambda row: float(row["objective"]),
        )[:8]

        print("\nLowest objective values:")

        print(f"{'alpha':>10} {'J':>18} {'residual RMS':>18}")

        for row in lowest:
            print(
                f"{float(row['alpha_deg']):10.2f} "
                f"{float(row['objective']):18.8e} "
                f"{float(row['residual_rms']):18.8e}"
            )

        minima = find_local_minima(
            all_rows,
            case_name,
        )

        print("\nGrid-local minima:")

        if not minima:
            print("  none found")
        else:
            for row in minima:
                print(
                    f"  alpha = "
                    f"{float(row['alpha_deg']):.2f} deg"
                    f"   J = "
                    f"{float(row['objective']):.8e}"
                )

        # Values near suspected false basin
        print("\nSelected AoAs:")

        for alpha_target in [
            40.0,
            45.0,
            48.0,
            49.0,
            50.0,
            55.0,
            60.0,
            63.0,
            65.0,
            70.0,
        ]:
            match = next(
                (
                    row
                    for row in case_rows
                    if np.isclose(
                        float(row["alpha_deg"]),
                        alpha_target,
                    )
                ),
                None,
            )

            if match is not None:
                print(
                    f"  alpha = "
                    f"{alpha_target:5.1f} deg"
                    f"   J = "
                    f"{float(match['objective']):.8e}"
                )

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 76)

    print(f"CSV saved to: {csv_path}")

    print(f"total wall time = {total_elapsed / 60.0:.2f} min")


if __name__ == "__main__":
    main()
