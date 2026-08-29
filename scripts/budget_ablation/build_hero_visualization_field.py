"""One presentation-only ImmersaForward render at h = 0.02, full time series.

PRESENTATION ONLY. This field exists solely to give the hero figure a sharper
wake background. It is deliberately kept out of data/cfd_validation_bank/, is
never scored by SensorArrayDesign, and no quantitative result in the repository
is computed at this resolution. Every committed number comes from the frozen
h = 0.05 campaign.

Settings follow the production case, with the grid refined and the time step
reduced to hold the convective CFL number fixed:

    production      h = 0.05   dt = 0.0025   U dt / h = 0.05
    visualization   h = 0.02   dt = 0.0010   U dt / h = 0.05

snapshot_freq = 500 stores a snapshot every 0.5 s, so the run yields 41 frames
over t = 0 .. 20 and the last one lands exactly on t = 20.0 (step 20000). The
hero animates Panel A across the whole sequence and uses the final frame for the
static still.

    python scripts/budget_ablation/build_hero_visualization_field.py
"""

import time
from pathlib import Path

import numpy as np
from immersa_tesseract_inference.pipeline import ForwardObservationPipeline

ALPHA_DEG = 63.0

H = 0.02
DT = 0.001
TF = 20.0
RE = 200.0
SNAPSHOT_FREQ = 500

FINAL_TIME = 20.0

# Alongside the other data assets; data/ is gitignored, so this large
# presentation asset does not ship with the repository.
OUTPUT = Path("data/hero_visualization_alpha063_h002.npz")


def main() -> None:
    """Run the single visualization solve and persist the t = 20 snapshot."""
    print("=" * 78)
    print("Hero visualization field (PRESENTATION ONLY)")
    print("=" * 78)
    print(f"  alpha         : {ALPHA_DEG} deg")
    print(f"  h             : {H}   (production 0.05)")
    print(f"  dt            : {DT}  (production 0.0025; same CFL = {DT / H:g})")
    print(f"  tf            : {TF}")
    print(f"  Re            : {RE}")
    print(f"  snapshot_freq : {SNAPSHOT_FREQ}  -> final snapshot at t = {TF}")
    print(f"  output        : {OUTPUT}")
    print(flush=True)

    if OUTPUT.exists():
        print("Field already present; nothing to do.")
        return

    started = time.perf_counter()

    with ForwardObservationPipeline(max_cached_flows=1) as pipeline:
        flow = pipeline.run_forward(
            angle_of_attack_deg=ALPHA_DEG,
            h=H,
            dt=DT,
            tf=TF,
            Re=RE,
            snapshot_freq=SNAPSHOT_FREQ,
        )

    elapsed = time.perf_counter() - started

    times = np.asarray(flow["times"], dtype=np.float64)

    time_error = float(abs(times[-1] - FINAL_TIME))

    print(f"  solve wall time : {elapsed / 60.0:.2f} min")
    print(f"  stored snapshots: {times.size} over t = {times[0]:g} .. {times[-1]:g}")
    print(f"  |t_final - {FINAL_TIME:g}| : {time_error:.3e}")

    if time_error > 1.0e-9:
        raise SystemExit(f"Last snapshot is not t = {FINAL_TIME}; got {times[-1]}.")

    ux = np.asarray(flow["ux"], dtype=np.float64)
    uy = np.asarray(flow["uy"], dtype=np.float64)

    np.savez_compressed(
        OUTPUT,
        ux=ux,
        uy=uy,
        ux_x=np.asarray(flow["ux_x"], dtype=np.float64),
        ux_y=np.asarray(flow["ux_y"], dtype=np.float64),
        uy_x=np.asarray(flow["uy_x"], dtype=np.float64),
        uy_y=np.asarray(flow["uy_y"], dtype=np.float64),
        times=times,
        angle_of_attack_deg=ALPHA_DEG,
        h=H,
        dt=DT,
        tf=TF,
        Re=RE,
        snapshot_freq=SNAPSHOT_FREQ,
        n_ib=np.asarray(flow["n_ib"]),
        ds=np.asarray(flow["ds"]),
        plate_length=np.asarray(flow["plate_length"]),
        purpose=np.asarray(
            "Presentation-only hero background. Not part of the quantitative "
            "CFD bank; no committed result is computed at this resolution."
        ),
    )

    print(f"  ux shape        : {ux.shape}   (nx, ny, n_time)")
    print(f"  uy shape        : {uy.shape}   (nx, ny, n_time)")
    print(f"  file size       : {OUTPUT.stat().st_size / 1e6:.2f} MB")
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
