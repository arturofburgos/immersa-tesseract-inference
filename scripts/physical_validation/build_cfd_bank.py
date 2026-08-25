"""Build the reusable real-CFD validation bank from ImmersaForward.

Sensors are passive: they do not enter the CFD state. One solve per angle of
attack therefore serves every sensor layout we will ever want to test, so the
bank is built once over the union of the two angle grids we need:

    design grid     20, 22.5, ..., 85   (27 angles, matches the frozen T4 design)
    landscape grid  20, 21,   ..., 85   (66 angles, matches the committed study)
    union                                (79 angles, 14 shared)

Only the five observation-time snapshots are persisted. WakeObservation brackets
in time with ``searchsorted(..., side="right") - 1`` clipped to ``n - 2``, so a
five-entry time grid reproduces all five requested times exactly, including the
final endpoint, which falls back to the last bracket with weight one.

Actual solver time values are stored, never nominal decimals.

The raw bank lives under data/, which is gitignored; only the manifest is
lightweight enough to track.
"""

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.cfd_bank import (
    BANK_DIR,
    DESIGN_GRID_DEG,
    DT,
    LANDSCAPE_GRID_DEG,
    OBSERVATION_TIMES,
    RE,
    SNAPSHOT_FREQ,
    TF,
    H,
    bank_path,
    union_grid,
)
from tesseract_core import Tesseract

# Same concurrency the committed identifiability campaign used.
MAX_WORKERS = 10

FORWARD_IMAGE = "immersa_tesseract_inference_immersa_forward"

IMMERSA_COMMIT = "62810dcff9418d6fface55dd34b5f1b914ffa743"

MANIFEST = Path("results/sensor_design/physical_validation/cfd_bank_manifest.json")


def solve_chunk(alphas: list[float]) -> list[dict]:
    """Solve several angles inside one long-lived T1 container."""
    records = []

    with Tesseract.from_image(FORWARD_IMAGE) as forward:
        for alpha in alphas:
            started = time.perf_counter()

            flow = forward.apply(
                {
                    "angle_of_attack_deg": float(alpha),
                    "h": H,
                    "dt": DT,
                    "tf": TF,
                    "Re": RE,
                    "snapshot_freq": SNAPSHOT_FREQ,
                }
            )

            times = np.asarray(flow["times"], dtype=np.float64)

            # Nearest stored snapshot to each requested observation time. The
            # ACTUAL solver times are what get persisted.
            indices = np.array(
                [int(np.argmin(np.abs(times - t))) for t in OBSERVATION_TIMES]
            )

            if len(np.unique(indices)) != len(indices):
                raise RuntimeError(
                    f"Observation times collapsed onto duplicate snapshots at "
                    f"alpha={alpha}: {indices}"
                )

            ux = np.asarray(flow["ux"], dtype=np.float64)[:, :, indices]
            uy = np.asarray(flow["uy"], dtype=np.float64)[:, :, indices]

            path = bank_path(alpha)

            np.savez_compressed(
                path,
                angle_of_attack_deg=np.float64(alpha),
                times=times[indices],
                requested_times=OBSERVATION_TIMES,
                snapshot_indices=indices,
                ux=ux,
                uy=uy,
                ux_x=np.asarray(flow["ux_x"], dtype=np.float64),
                ux_y=np.asarray(flow["ux_y"], dtype=np.float64),
                uy_x=np.asarray(flow["uy_x"], dtype=np.float64),
                uy_y=np.asarray(flow["uy_y"], dtype=np.float64),
                h=np.float64(H),
                dt=np.float64(DT),
                tf=np.float64(TF),
                Re=np.float64(RE),
                snapshot_freq=np.int64(SNAPSHOT_FREQ),
                n_ib=np.int64(flow["n_ib"]),
                ds=np.float64(flow["ds"]),
                plate_length=np.float64(flow["plate_length"]),
            )

            elapsed = time.perf_counter() - started

            records.append(
                {
                    "angle_of_attack_deg": float(alpha),
                    "file": path.name,
                    "stored_times": times[indices].tolist(),
                    "max_time_offset": float(
                        np.max(np.abs(times[indices] - OBSERVATION_TIMES))
                    ),
                    "bytes": path.stat().st_size,
                    "solve_seconds": elapsed,
                }
            )

            print(f"  alpha={alpha:6.2f}  {elapsed:5.1f} s", flush=True)

    return records


def main() -> None:
    """Solve every missing angle and write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-solve angles whose bank entry already exists.",
    )
    arguments = parser.parse_args()

    BANK_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    alphas = union_grid()

    pending = [float(a) for a in alphas if arguments.force or not bank_path(a).exists()]

    print("=" * 78)
    print("Real-CFD validation bank")
    print("=" * 78)
    print(f"h={H} dt={DT} tf={TF} Re={RE} snapshot_freq={SNAPSHOT_FREQ}")
    print(f"observation times : {OBSERVATION_TIMES.tolist()}")
    print(f"design grid       : {DESIGN_GRID_DEG.size} angles")
    print(f"landscape grid    : {LANDSCAPE_GRID_DEG.size} angles")
    print(f"union             : {alphas.size} angles")
    print(f"already present   : {alphas.size - len(pending)}")
    print(f"to solve          : {len(pending)}")
    print(f"workers           : {MAX_WORKERS}")
    print(flush=True)

    started = time.perf_counter()

    records: list[dict] = []

    if pending:
        chunks = [pending[i::MAX_WORKERS] for i in range(MAX_WORKERS)]
        chunks = [c for c in chunks if c]

        context = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=len(chunks),
            mp_context=context,
        ) as executor:
            futures = [executor.submit(solve_chunk, chunk) for chunk in chunks]

            for future in as_completed(futures):
                records.extend(future.result())

    elapsed = time.perf_counter() - started

    # Rebuild the manifest from what is actually on disk.
    entries = []

    for alpha in alphas:
        path = bank_path(alpha)

        if not path.exists():
            raise RuntimeError(f"Bank entry missing for alpha={alpha}: {path}")

        with np.load(path) as data:
            entries.append(
                {
                    "angle_of_attack_deg": float(data["angle_of_attack_deg"]),
                    "file": path.name,
                    "stored_times": data["times"].tolist(),
                    "max_time_offset": float(
                        np.max(np.abs(data["times"] - OBSERVATION_TIMES))
                    ),
                    "ux_shape": list(data["ux"].shape),
                    "uy_shape": list(data["uy"].shape),
                    "bytes": path.stat().st_size,
                }
            )

    total_bytes = sum(e["bytes"] for e in entries)

    manifest = {
        "description": (
            "Dense ImmersaForward wake fields at the five observation times, "
            "one entry per angle of attack. Sensors are passive, so these "
            "fields serve any sensor layout without new CFD."
        ),
        "configuration": {
            "h": H,
            "dt": DT,
            "tf": TF,
            "Re": RE,
            "snapshot_freq": SNAPSHOT_FREQ,
            "observation_times": OBSERVATION_TIMES.tolist(),
        },
        "provenance": {
            "source": "ImmersaForward Tesseract (T1)",
            "image": FORWARD_IMAGE,
            "immersa_commit": IMMERSA_COMMIT,
        },
        "grids": {
            "design_grid_deg": DESIGN_GRID_DEG.tolist(),
            "landscape_grid_deg": LANDSCAPE_GRID_DEG.tolist(),
            "union_size": int(alphas.size),
        },
        "storage": {
            "directory": str(BANK_DIR),
            "gitignored": True,
            "total_bytes": total_bytes,
            "total_megabytes": total_bytes / 1.0e6,
        },
        "build": {
            "solved_this_run": len(pending),
            "wall_time_s": elapsed,
        },
        "entries": entries,
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    offsets = [e["max_time_offset"] for e in entries]

    print()
    print(f"entries            : {len(entries)}")
    print(f"total size         : {total_bytes / 1.0e6:.1f} MB")
    print(f"max |stored - requested| time offset: {max(offsets):.3e}")
    print(f"wall time          : {elapsed / 60.0:.2f} min")
    print(f"\nWrote {MANIFEST}")


if __name__ == "__main__":
    main()
