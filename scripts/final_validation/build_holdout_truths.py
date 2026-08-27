"""Open the three preregistered unseen physical truths.

33.5, 58.5 and 74.5 degrees were fixed in
results/sensor_design/refined_design/phase2_preregistration.json before any
Phase-II optimization ran, and were verified absent from the design grid, from
T3 training and validation, and from the existing CFD bank.

This script runs ImmersaForward at those three angles for the first time, using
the identical production settings and persistence format as the main bank, and
stores them in a separate directory so it stays obvious which cases were sealed.

Three CFD solves. Nothing else.
"""

import json
import time
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.cfd_bank import (
    DT,
    FLOW_KEYS,
    OBSERVATION_TIMES,
    RE,
    SNAPSHOT_FREQ,
    TF,
    H,
    bank_path,
)
from tesseract_core import Tesseract

FORWARD_IMAGE = "immersa_tesseract_inference_immersa_forward"
IMMERSA_COMMIT = "62810dcff9418d6fface55dd34b5f1b914ffa743"

HOLDOUT_ALPHAS = (33.5, 58.5, 74.5)

HOLDOUT_DIR = Path("data/cfd_holdout_bank")
MANIFEST = Path(
    "results/sensor_design/final_physical_validation/holdout_bank_manifest.json"
)

PREREGISTRATION = Path(
    "results/sensor_design/refined_design/phase2_preregistration.json"
)


def holdout_path(alpha_deg: float) -> Path:
    """Bank filename for one sealed truth, in the holdout directory."""
    return bank_path(alpha_deg, HOLDOUT_DIR)


def load_holdout_flow(alpha_deg: float) -> dict:
    """Reload one sealed truth as a WakeObservation-ready flow mapping."""
    with np.load(holdout_path(alpha_deg)) as data:
        flow = {key: data[key] for key in FLOW_KEYS}
        flow["times"] = data["times"]
    return flow


def main() -> None:
    """Solve the three sealed truths and write the manifest."""
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    preregistered = json.loads(PREREGISTRATION.read_text())["holdout_alphas_deg"]

    if sorted(preregistered) != sorted(HOLDOUT_ALPHAS):
        raise RuntimeError(
            f"Holdout angles disagree with the preregistration: "
            f"{preregistered} vs {list(HOLDOUT_ALPHAS)}"
        )

    print("=" * 78)
    print("Opening the preregistered unseen physical truths")
    print("=" * 78)
    print(f"h={H} dt={DT} tf={TF} Re={RE} snapshot_freq={SNAPSHOT_FREQ}")
    print(f"observation times : {OBSERVATION_TIMES.tolist()}")
    print(f"angles            : {list(HOLDOUT_ALPHAS)}")
    print(flush=True)

    started = time.perf_counter()

    entries = []

    with Tesseract.from_image(FORWARD_IMAGE) as forward:
        for alpha in HOLDOUT_ALPHAS:
            path = holdout_path(alpha)

            if path.exists():
                print(f"  alpha={alpha}: already present, skipping", flush=True)
            else:
                solve_start = time.perf_counter()

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

                indices = np.array(
                    [int(np.argmin(np.abs(times - t))) for t in OBSERVATION_TIMES]
                )

                if len(np.unique(indices)) != len(indices):
                    raise RuntimeError(
                        f"Observation times collapsed onto duplicate snapshots "
                        f"at alpha={alpha}"
                    )

                np.savez_compressed(
                    path,
                    angle_of_attack_deg=np.float64(alpha),
                    times=times[indices],
                    requested_times=OBSERVATION_TIMES,
                    snapshot_indices=indices,
                    ux=np.asarray(flow["ux"], dtype=np.float64)[:, :, indices],
                    uy=np.asarray(flow["uy"], dtype=np.float64)[:, :, indices],
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

                print(
                    f"  alpha={alpha:6.2f}  {time.perf_counter() - solve_start:5.1f} s",
                    flush=True,
                )

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

    elapsed = time.perf_counter() - started

    total_bytes = sum(e["bytes"] for e in entries)

    manifest = {
        "description": (
            "Sealed unseen physical truths. These three angles were "
            "preregistered before Phase-II optimization and evaluated here for "
            "the first time. They are stored separately from the main bank so "
            "their held-out status stays unambiguous."
        ),
        "preregistration": str(PREREGISTRATION),
        "holdout_alphas_deg": list(HOLDOUT_ALPHAS),
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
        "storage": {
            "directory": str(HOLDOUT_DIR),
            "gitignored": True,
            "total_bytes": total_bytes,
            "total_megabytes": total_bytes / 1.0e6,
        },
        "build": {"n_solves": len(HOLDOUT_ALPHAS), "wall_time_s": elapsed},
        "entries": entries,
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print()
    print(f"entries    : {len(entries)}")
    print(f"total size : {total_bytes / 1.0e6:.2f} MB")
    print(
        f"max |stored - requested| time offset: "
        f"{max(e['max_time_offset'] for e in entries):.3e}"
    )
    print(f"wall time  : {elapsed / 60.0:.2f} min")
    print(f"\nWrote {MANIFEST}")


if __name__ == "__main__":
    main()
