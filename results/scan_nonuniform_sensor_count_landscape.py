"""Noise-free AoA loss landscapes versus spatial sensor count."""

import csv
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.pipeline import ForwardObservationPipeline

# ============================================================
# Experiment configuration
# ============================================================

ALPHA_TRUE = 63.0

# Expanded scan because the multi-start experiment exposed
# a previously unseen low-AoA basin near 29 degrees.
ALPHAS = np.arange(
    20.0,
    85.0 + 1.0,
    1.0,
    dtype=np.float64,
)

MAX_WORKERS = 10

# Same sparse nonuniform temporal schedule
SENSOR_TIMES = np.array(
    [12.0, 13.3, 15.1, 17.4, 20.0],
    dtype=np.float64,
)

# Same x location for every sensor; vary spatial diversity in y.
SENSOR_CASES = {
    "Ns1": {
        "x": np.array(
            [1.0],
            dtype=np.float64,
        ),
        "y": np.array(
            [0.0],
            dtype=np.float64,
        ),
    },
    "Ns2": {
        "x": np.array(
            [1.0, 1.0],
            dtype=np.float64,
        ),
        "y": np.array(
            [-0.4, 0.4],
            dtype=np.float64,
        ),
    },
    "Ns3": {
        "x": np.array(
            [1.0, 1.0, 1.0],
            dtype=np.float64,
        ),
        "y": np.array(
            [-0.4, 0.0, 0.4],
            dtype=np.float64,
        ),
    },
    "Ns5": {
        "x": np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float64,
        ),
        "y": np.array(
            [-0.8, -0.4, 0.0, 0.4, 0.8],
            dtype=np.float64,
        ),
    },
}


# Developed-wake CFD settings
H = 0.05
DT = 0.0025
TF = 20.0
RE = 200.0
SNAPSHOT_FREQ = 40

OUTPUT_DIR = Path("results/nonuniform_sensor_count_landscape")


# ============================================================
# Local-minimum helper
# ============================================================


def find_local_minima(
    rows: list[dict[str, object]],
    case_name: str,
) -> list[dict[str, object]]:
    """Find grid-local minima for one sensor configuration."""
    case_rows = sorted(
        [row for row in rows if row["case"] == case_name],
        key=lambda row: float(row["alpha_deg"]),
    )

    minima = []

    for i in range(1, len(case_rows) - 1):
        j_prev = float(case_rows[i - 1]["objective_per_scalar"])

        j_here = float(case_rows[i]["objective_per_scalar"])

        j_next = float(case_rows[i + 1]["objective_per_scalar"])

        if j_here <= j_prev and j_here <= j_next:
            minima.append(case_rows[i])

    return minima


# ============================================================
# Worker
# ============================================================


def evaluate_chunk(
    worker_id: int,
    alphas: list[float],
    truth_observations: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """Run a group of AoAs using one persistent Tesseract pipeline."""
    rows: list[dict[str, object]] = []

    log_path = OUTPUT_DIR / f"worker_{worker_id:02d}.log"

    with log_path.open(
        "w",
        buffering=1,
    ) as log:
        print(
            f"Worker {worker_id}: {len(alphas)} AoA cases",
            file=log,
            flush=True,
        )

        print(
            f"AoAs: {alphas}",
            file=log,
            flush=True,
        )

        # Each alpha is unique, so there is little benefit
        # from maintaining a large forward-flow cache here.
        with ForwardObservationPipeline(max_cached_flows=1) as pipeline:
            for index, alpha in enumerate(
                alphas,
                start=1,
            ):
                start = time.perf_counter()

                print(
                    f"\n[{index}/{len(alphas)}] Computing F({alpha:.1f} deg)...",
                    file=log,
                    flush=True,
                )

                # ------------------------------------------------
                # ONE expensive CFD solve for this AoA
                # ------------------------------------------------

                flow = pipeline.run_forward(
                    angle_of_attack_deg=alpha,
                    h=H,
                    dt=DT,
                    tf=TF,
                    Re=RE,
                    snapshot_freq=SNAPSHOT_FREQ,
                )

                # ------------------------------------------------
                # Sample the SAME flow with all sensor arrays
                # ------------------------------------------------

                for case_name, case in SENSOR_CASES.items():
                    prediction = pipeline.observe(
                        flow,
                        case["x"],
                        case["y"],
                        SENSOR_TIMES,
                    )

                    residual = prediction - truth_observations[case_name]

                    objective = 0.5 * float(np.sum(residual**2))

                    n_scalar = int(residual.size)

                    objective_per_scalar = objective / n_scalar

                    residual_rms = float(np.sqrt(np.mean(residual**2)))

                    rows.append(
                        {
                            "alpha_deg": float(alpha),
                            "case": case_name,
                            "n_sensors": len(case["x"]),
                            "n_times": len(SENSOR_TIMES),
                            "n_scalar": n_scalar,
                            "objective": objective,
                            "objective_per_scalar": objective_per_scalar,
                            "residual_rms": residual_rms,
                            "worker": worker_id,
                        }
                    )

                    print(
                        f"    {case_name}: "
                        f"J={objective:.8e}, "
                        f"J/N={objective_per_scalar:.8e}, "
                        f"RMS={residual_rms:.8e}",
                        file=log,
                        flush=True,
                    )

                elapsed = time.perf_counter() - start

                print(
                    f"    AoA {alpha:.1f} complete in {elapsed / 60.0:.2f} min",
                    file=log,
                    flush=True,
                )

    return rows


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Run the full sensor-count landscape comparison."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_start = time.perf_counter()

    print("=" * 88, flush=True)
    print(
        "NONUNIFORM-TIME SENSOR-COUNT AoA LOSS LANDSCAPES",
        flush=True,
    )
    print("=" * 88, flush=True)

    print(f"truth AoA       = {ALPHA_TRUE:.1f} deg")

    print(f"AoA scan        = {ALPHAS[0]:.1f} ... {ALPHAS[-1]:.1f} deg")

    print(f"AoA spacing     = {ALPHAS[1] - ALPHAS[0]:.1f} deg")

    print(f"number of AoAs  = {len(ALPHAS)}")

    print(f"workers         = {MAX_WORKERS}")

    print(f"sensor times    = {SENSOR_TIMES.tolist()}")

    print(f"sensor cases    = {list(SENSOR_CASES)}")

    # ========================================================
    # Compute clean truth observations once
    # ========================================================

    print(
        "\nComputing clean truth flow F(63 deg)...",
        flush=True,
    )

    truth_observations: dict[
        str,
        np.ndarray,
    ] = {}

    with ForwardObservationPipeline(max_cached_flows=1) as pipeline:
        truth_flow = pipeline.run_forward(
            angle_of_attack_deg=ALPHA_TRUE,
            h=H,
            dt=DT,
            tf=TF,
            Re=RE,
            snapshot_freq=SNAPSHOT_FREQ,
        )

        for case_name, case in SENSOR_CASES.items():
            truth_observations[case_name] = pipeline.observe(
                truth_flow,
                case["x"],
                case["y"],
                SENSOR_TIMES,
            )

            print(
                f"  {case_name}: {truth_observations[case_name].shape}",
                flush=True,
            )

    print(
        "Truth observations complete.",
        flush=True,
    )

    # ========================================================
    # Divide 66 AoAs among 16 workers
    # ========================================================

    chunks = [
        chunk.tolist()
        for chunk in np.array_split(
            ALPHAS,
            MAX_WORKERS,
        )
        if len(chunk) > 0
    ]

    print(
        "\nWorker assignments:",
        flush=True,
    )

    for worker_id, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"  worker {worker_id:02d}: "
            f"{chunk[0]:.1f} -> "
            f"{chunk[-1]:.1f} deg "
            f"({len(chunk)} simulations)",
            flush=True,
        )

    print(
        "\nLaunching parallel CFD scan...\n",
        flush=True,
    )

    all_rows: list[dict[str, object]] = []

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
                truth_observations,
            )

            futures[future] = worker_id

        for future in as_completed(futures):
            worker_id = futures[future]

            try:
                rows = future.result()

            except Exception as exc:
                print(
                    f"[FAILED] worker {worker_id}: {exc!r}",
                    flush=True,
                )

                raise

            all_rows.extend(rows)

            elapsed = time.perf_counter() - total_start

            print(
                f"[DONE] worker {worker_id:02d} | elapsed {elapsed / 60.0:.2f} min",
                flush=True,
            )

    # ========================================================
    # Sort
    # ========================================================

    case_order = {
        "Ns1": 1,
        "Ns2": 2,
        "Ns3": 3,
        "Ns5": 5,
    }

    all_rows.sort(
        key=lambda row: (
            float(row["alpha_deg"]),
            case_order[str(row["case"])],
        )
    )

    # ========================================================
    # Save CSV
    # ========================================================

    csv_path = OUTPUT_DIR / "sensor_count_loss_landscapes.csv"

    fieldnames = list(all_rows[0].keys())

    with csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_rows)

    # ========================================================
    # Landscape summaries
    # ========================================================

    print("\n\n" + "=" * 100)
    print("SENSOR-COUNT LANDSCAPE SUMMARY")
    print("=" * 100)

    for case_name in [
        "Ns1",
        "Ns2",
        "Ns3",
        "Ns5",
    ]:
        case_rows = [row for row in all_rows if row["case"] == case_name]

        n_sensors = int(case_rows[0]["n_sensors"])

        print(f"\n{'-' * 76}")

        print(
            f"{case_name}: "
            f"{n_sensors} spatial sensor(s), "
            f"{len(SENSOR_TIMES)} time samples"
        )

        print(f"Scalar measurements = {int(case_rows[0]['n_scalar'])}")

        # ----------------------------------------------------
        # Local minima using normalized objective
        # ----------------------------------------------------

        minima = find_local_minima(
            all_rows,
            case_name,
        )

        print("\nGrid-local minima (using J / number of scalar measurements):")

        if not minima:
            print("  none detected")

        else:
            for row in minima:
                print(
                    f"  alpha = "
                    f"{float(row['alpha_deg']):6.2f} deg"
                    f"   "
                    f"J = "
                    f"{float(row['objective']):.8e}"
                    f"   "
                    f"J/N = "
                    f"{float(row['objective_per_scalar']):.8e}"
                    f"   "
                    f"RMS = "
                    f"{float(row['residual_rms']):.8e}"
                )

        # ----------------------------------------------------
        # Lowest normalized objective values
        # ----------------------------------------------------

        lowest = sorted(
            case_rows,
            key=lambda row: float(row["objective_per_scalar"]),
        )[:10]

        print("\nTen lowest normalized objective values:")

        print(f"{'alpha':>8} {'J':>16} {'J/N':>16} {'RMS':>16}")

        for row in lowest:
            print(
                f"{float(row['alpha_deg']):8.2f} "
                f"{float(row['objective']):16.8e} "
                f"{float(row['objective_per_scalar']):16.8e} "
                f"{float(row['residual_rms']):16.8e}"
            )

        # ----------------------------------------------------
        # Important locations
        # ----------------------------------------------------

        print("\nSelected AoAs:")

        selected_alphas = [
            20.0,
            25.0,
            29.0,
            30.0,
            35.0,
            40.0,
            45.0,
            50.0,
            51.0,
            55.0,
            60.0,
            63.0,
            65.0,
            70.0,
            75.0,
            76.0,
            80.0,
            85.0,
        ]

        for alpha_target in selected_alphas:
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
                    f"   "
                    f"J/N = "
                    f"{float(match['objective_per_scalar']):.8e}"
                    f"   "
                    f"RMS = "
                    f"{float(match['residual_rms']):.8e}"
                )

    # ========================================================
    # Compact comparison of local minima
    # ========================================================

    print("\n\n" + "=" * 100)
    print("COMPACT LOCAL-MINIMUM COMPARISON")
    print("=" * 100)

    for case_name in [
        "Ns1",
        "Ns2",
        "Ns3",
        "Ns5",
    ]:
        minima = find_local_minima(
            all_rows,
            case_name,
        )

        minima_text = ", ".join(f"{float(row['alpha_deg']):.1f}°" for row in minima)

        print(f"{case_name:>4}: {minima_text}")

    # ========================================================
    # Finish
    # ========================================================

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 100)

    print(f"CSV saved to: {csv_path}")

    print(f"total wall time = {total_elapsed / 60.0:.2f} min")


if __name__ == "__main__":
    main()
