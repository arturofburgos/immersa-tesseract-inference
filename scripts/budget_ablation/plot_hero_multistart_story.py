"""Hero telling the whole Ns=5 sensor-design story in one figure.

    (A) the real wake and the region the probes are designed in
    (B1) the conventional rake
    (B2) optimizing from that rake -- a weaker near-wake configuration
    (B3) the winning multistart design -- probes split near and far
    (C)  the surrogate design objective along both optimizations

Every probe position drawn here is a recorded L-BFGS-B iterate from the frozen
campaign; nothing is interpolated or invented. B2 comes from the naive-start
replay, B3 from the random04 winning replay that reproduces the frozen design
bit-exactly. B2 is a diagnostic, never the headline layout.

Two provenance facts the figure states in its own footnote:

    the wake is a representative alpha = 63 deg, t = 20 render at h = 0.02,
    shown for visualization only, while every quantitative value comes from
    the frozen h = 0.05 campaign; and the design was optimized over the AoA
    design ensemble through T3 -> T4, not against this single wake.

Panel C needs the surrogate objective at each stored iterate, which the replays
did not record. It is evaluated once here -- a read-out of the frozen objective
at coordinates that already exist -- and cached, so no optimization is repeated.

Panel A animates the wake across the whole simulation while B2/B3 advance
through optimization iterates. The two are independent quantities sharing one
normalized 0-100% frame index; they are not physically synchronized. The B-panel
backgrounds are fixed at the final snapshot, so only the probes move there.

    python scripts/budget_ablation/plot_hero_multistart_story.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent))
from ablation_core import X_BOUNDS, Y_BOUNDS, AblationPipeline, canonicalize
from plot_readme_figures import plate_endpoints

ALPHA_DEG = 63.0
BUDGET = 5

RESULTS_DIR = Path("results/sensor_budget_ablation")
FIGURE_DIR = RESULTS_DIR / "figures"

FIELD = Path("data/hero_visualization_alpha063_h002.npz")
WINNER_REPLAY = RESULTS_DIR / "ns5_optimization_replay.json"
NAIVE_REPLAY = RESULTS_DIR / "ns5_naive_start_replay.json"
HISTORY_CACHE = RESULTS_DIR / "ns5_objective_histories.json"

STEM = "hero_multistart_story"

BASELINE_COLOR = "#d62828"
NAIVE_COLOR = "#e08214"
WINNER_COLOR = "#0b6e6e"
TEXT_COLOR = "#1d1d1d"
MUTED = "#5a5a5a"

# Panel A shows the wake in context; the B panels crop to the design box.
CONTEXT_X = (-0.9, 4.6)
CONTEXT_Y = (-1.3, 1.3)
DESIGN_X = (0.82, 3.18)
DESIGN_Y = (-0.98, 0.98)

CONTEXT_NX, CONTEXT_NY = 900, 430
DESIGN_NX, DESIGN_NY = 520, 440

MOTION_FRAMES = 40
FRAME_MS = 150
HOLD_MS = 1500
GIF_DPI = 70


def interpolate(data: dict, nx: int, ny: int) -> tuple[np.ndarray, ...]:
    """Ux and vorticity on one regular display grid."""
    ux, uy = data["ux"], data["uy"]
    ux_x, ux_y = data["ux_x"], data["ux_y"]
    uy_x, uy_y = data["uy_x"], data["uy_y"]

    x = np.linspace(max(ux_x[0], uy_x[0]), min(ux_x[-1], uy_x[-1]), nx)
    y = np.linspace(max(ux_y[0], uy_y[0]), min(ux_y[-1], uy_y[-1]), ny)
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    points = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=-1)

    ux_i = RegularGridInterpolator(
        (ux_x, ux_y), ux, bounds_error=False, fill_value=None
    )(points).reshape(mesh_x.shape)
    uy_i = RegularGridInterpolator(
        (uy_x, uy_y), uy, bounds_error=False, fill_value=None
    )(points).reshape(mesh_x.shape)

    omega = np.gradient(uy_i, x, axis=0) - np.gradient(ux_i, y, axis=1)

    return x, y, ux_i, omega


def load_field() -> dict:
    """Panel A over the whole run, plus the fixed design-panel background.

    Panel A animates: one vorticity frame per stored CFD snapshot. The B panels
    are deliberately static and always show the final snapshot, so only probe
    positions move there.
    """
    if not FIELD.exists():
        raise SystemExit(f"{FIELD} is missing; build it before plotting.")

    with np.load(FIELD, allow_pickle=False) as data:
        ux_series = np.asarray(data["ux"], dtype=np.float64)
        uy_series = np.asarray(data["uy"], dtype=np.float64)
        grids = {key: data[key] for key in ("ux_x", "ux_y", "uy_x", "uy_y")}
        times = np.asarray(data["times"], dtype=np.float64)
        spacing = float(data["h"])

    def snapshot(index: int) -> dict:
        return {**grids, "ux": ux_series[:, :, index], "uy": uy_series[:, :, index]}

    omegas = []
    for index in range(times.size):
        cx, cy, _, omega = interpolate(snapshot(index), CONTEXT_NX, CONTEXT_NY)
        omegas.append(omega)

    # One symmetric colour range for the whole sequence, so the animation does
    # not rescale itself frame to frame.
    omega_limit = float(np.percentile(np.abs(np.stack(omegas)), 99.0))

    # The design panels stay on the final snapshot.
    dx, dy, design_ux, _ = interpolate(snapshot(times.size - 1), DESIGN_NX, DESIGN_NY)

    inside_x = (dx >= DESIGN_X[0]) & (dx <= DESIGN_X[1])
    inside_y = (dy >= DESIGN_Y[0]) & (dy <= DESIGN_Y[1])
    crop = design_ux[np.ix_(inside_x, inside_y)]

    return {
        "context_x": cx,
        "context_y": cy,
        "omega_series": omegas,
        "omega_limit": omega_limit,
        "times": times,
        "design_x": dx,
        "design_y": dy,
        "ux": design_ux,
        "ux_low": float(np.percentile(crop, 1.0)),
        "ux_high": float(np.percentile(crop, 99.0)),
        "time": float(times[-1]),
        "h": spacing,
    }


def load_story() -> dict:
    """Validated layouts, trajectories and frozen endpoint metrics."""
    summary = json.loads((RESULTS_DIR / "budget_ablation_summary.json").read_text())
    winner = json.loads(WINNER_REPLAY.read_text())
    naive = json.loads(NAIVE_REPLAY.read_text())

    baseline = np.asarray(summary["designs"][f"Ns{BUDGET}_naive"], dtype=np.float64)
    frozen = np.asarray(summary["designs"][f"Ns{BUDGET}_optimized"], dtype=np.float64)

    naive_path = np.asarray(naive["trajectory"], dtype=np.float64)
    winner_path = np.asarray(winner["trajectory"], dtype=np.float64)

    checks = {
        "B1 equals frozen naive rake": np.array_equal(
            baseline, np.asarray(summary["designs"][f"Ns{BUDGET}_naive"])
        ),
        "B2 starts at the conventional rake": np.array_equal(naive_path[0], baseline),
        "B2 ends at the stored naive-start optimum": np.array_equal(
            canonicalize(naive_path[-1]),
            np.asarray(naive["final_layout"], dtype=np.float64),
        ),
        "B3 ends at the frozen optimized design": np.array_equal(
            winner_path[-1], frozen
        ),
        "B3 is the random04 winner": winner["winning_start"] == "random04",
        "winner replay validated": bool(winner.get("validated")),
    }

    for label, passed in checks.items():
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")
    if not all(checks.values()):
        raise SystemExit("Provenance validation failed; nothing was drawn.")

    physical: dict[str, float] = {}
    with (RESULTS_DIR / "budget_ablation.csv").open() as handle:
        for row in csv.DictReader(handle):
            if int(row["n_sensors"]) == BUDGET:
                physical[row["family"]] = float(row["physical_hard_min"])

    return {
        "baseline": baseline,
        "naive_path": naive_path,
        "winner_path": winner_path,
        "cfd": {
            "baseline": physical["naive"],
            "naive": float(naive["physical_hard_min_final"]),
            "winner": physical["optimized"],
        },
        "surrogate_baseline": float(naive["surrogate_D_naive_baseline"]),
        "tau": float(naive["tau"]),
    }


def objective_histories(story: dict) -> dict:
    """Surrogate D_tau at every stored iterate, computed once and cached."""
    if HISTORY_CACHE.exists():
        cached = json.loads(HISTORY_CACHE.read_text())
        if len(cached["naive"]) == len(story["naive_path"]) and len(
            cached["winner"]
        ) == len(story["winner_path"]):
            print("  objective histories: loaded from cache")
            return cached

    print(
        "  objective histories: evaluating the frozen objective at "
        f"{len(story['naive_path']) + len(story['winner_path'])} stored iterates",
        flush=True,
    )

    tau = story["tau"]
    histories: dict[str, list[float]] = {}

    with AblationPipeline() as pipeline:
        for name, path in (
            ("naive", story["naive_path"]),
            ("winner", story["winner_path"]),
        ):
            values = []
            for index, layout in enumerate(path):
                scored = pipeline.score(pipeline.surrogate_measurements(layout), tau)
                values.append(float(scored["D_tau"]))
                if index % 25 == 0:
                    print(f"    {name} {index}/{len(path)}", flush=True)
            histories[name] = values

    histories["note"] = (
        "Surrogate T3 -> T4 discrimination D_tau evaluated at each recorded "
        "L-BFGS-B iterate of the frozen Ns=5 campaign. Read-out only: no "
        "optimization was repeated and no design was reselected."
    )
    histories["tau"] = tau

    HISTORY_CACHE.write_text(json.dumps(histories, indent=2) + "\n")
    print(f"  wrote {HISTORY_CACHE}")

    return histories


def draw_wake(ax: plt.Axes, field: dict, *, context: bool, frame: int = -1) -> None:
    """Shared wake rendering: vorticity for context, ux for the design panels.

    ``frame`` selects the Panel A snapshot and is ignored by the design panels,
    whose background is fixed at the final snapshot.
    """
    if context:
        ax.pcolormesh(
            field["context_x"],
            field["context_y"],
            field["omega_series"][frame].T,
            cmap="RdBu_r",
            vmin=-field["omega_limit"],
            vmax=field["omega_limit"],
            shading="auto",
            rasterized=True,
        )
        ax.set_xlim(*CONTEXT_X)
        ax.set_ylim(*CONTEXT_Y)
    else:
        ax.pcolormesh(
            field["design_x"],
            field["design_y"],
            field["ux"].T,
            cmap="cividis",
            vmin=field["ux_low"],
            vmax=field["ux_high"],
            shading="auto",
            rasterized=True,
            alpha=0.62,
        )
        ax.set_xlim(*DESIGN_X)
        ax.set_ylim(*DESIGN_Y)

    start, end = plate_endpoints(ALPHA_DEG)
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color="black",
        linewidth=3.4 if context else 2.6,
        solid_capstyle="round",
        zorder=5,
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_title(ax: plt.Axes, tag: str, title: str, subtitle: str, color: str) -> None:
    """Consistent (tag) Title / subtitle stack above a panel."""
    ax.text(
        0.0,
        1.085,
        f"{tag} {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=color,
    )
    ax.text(
        0.0,
        1.012,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.8,
        color=MUTED,
    )


def metric_label(ax: plt.Axes, value: float, gain: float | None, color: str) -> None:
    """Compact frozen real-CFD endpoint metric inside a panel."""
    value_text = f"{value:.4f}"
    if gain is not None:
        value_text += f"   ({gain:+.1%})"

    # Placed beneath the panel so it can carry the full judge-facing name
    # without overflowing the field or hiding a probe.
    ax.text(
        0.5,
        -0.055,
        "Worst-case CFD discrimination / measurement",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.6,
        color=MUTED,
    )
    ax.text(
        0.5,
        -0.145,
        value_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13.0,
        fontweight="bold",
        color=color,
    )


def draw_probes(ax: plt.Axes, layout: np.ndarray, color: str, marker: str) -> None:
    """Probe markers."""
    ax.scatter(
        layout[0::2],
        layout[1::2],
        s=125,
        marker=marker,
        color=color,
        edgecolor="white",
        linewidth=1.7,
        zorder=7,
    )


def draw_traces(ax: plt.Axes, path: np.ndarray, upto: int, color: str) -> None:
    """Faint per-probe path up to the current iterate."""
    if upto <= 0:
        return
    visible = path[: upto + 1]
    for probe in range(BUDGET):
        ax.plot(
            visible[:, 2 * probe],
            visible[:, 2 * probe + 1],
            color=color,
            linewidth=1.3,
            alpha=0.55,
            solid_capstyle="round",
            zorder=6,
        )


def draw_context(ax: plt.Axes, field: dict, frame: int = -1) -> None:
    """Panel A: the wake plus the box the probes are designed in."""
    draw_wake(ax, field, context=True, frame=frame)

    ax.add_patch(
        Rectangle(
            (X_BOUNDS[0], Y_BOUNDS[0]),
            X_BOUNDS[1] - X_BOUNDS[0],
            Y_BOUNDS[1] - Y_BOUNDS[0],
            fill=False,
            edgecolor=WINNER_COLOR,
            linewidth=2.2,
            linestyle=(0, (5, 3)),
            zorder=6,
        )
    )
    ax.text(
        X_BOUNDS[0] + 0.04,
        Y_BOUNDS[1] - 0.13,
        "sensor-design region",
        fontsize=10.5,
        fontweight="bold",
        color=WINNER_COLOR,
        va="top",
        zorder=7,
    )

    panel_title(
        ax,
        "(A)",
        "Representative real-CFD wake",
        rf"$\alpha={ALPHA_DEG:g}^\circ$, $t={field['times'][frame]:.1f}$   "
        rf"(vorticity; probes measure $u_x,u_y$)",
        TEXT_COLOR,
    )


def draw_objective(
    ax: plt.Axes,
    histories: dict,
    story: dict,
    naive_upto: int,
    winner_upto: int,
) -> None:
    """Panel C: the surrogate objective along both optimizations."""
    naive = np.asarray(histories["naive"])
    winner = np.asarray(histories["winner"])

    ax.axhline(
        story["surrogate_baseline"],
        color=BASELINE_COLOR,
        linestyle=(0, (4, 3)),
        linewidth=1.8,
        zorder=2,
    )
    ax.text(
        3,
        story["surrogate_baseline"],
        "conventional rake",
        color=BASELINE_COLOR,
        fontsize=9.6,
        va="bottom",
        ha="left",
    )

    ax.plot(
        np.arange(naive_upto + 1),
        naive[: naive_upto + 1],
        color=NAIVE_COLOR,
        linewidth=2.4,
        zorder=3,
        label="from the rake (B2)",
    )
    ax.plot(
        np.arange(winner_upto + 1),
        winner[: winner_upto + 1],
        color=WINNER_COLOR,
        linewidth=2.6,
        zorder=4,
        label="winning start (B3)",
    )

    for values, upto, color in (
        (naive, naive_upto, NAIVE_COLOR),
        (winner, winner_upto, WINNER_COLOR),
    ):
        ax.plot(
            [upto],
            [values[upto]],
            marker="o",
            markersize=7,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=5,
        )

    ax.set_xlim(-3, len(naive) + 2)
    span = max(naive.max(), winner.max()) - story["surrogate_baseline"]
    ax.set_ylim(
        story["surrogate_baseline"] - 0.12 * span,
        max(naive.max(), winner.max()) + 0.18 * span,
    )
    ax.set_xlabel("optimizer iteration", fontsize=10.5)
    ax.set_ylabel(r"surrogate  $D_\tau$", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="center right")

    panel_title(
        ax,
        "(C)",
        "Design objective during optimization",
        "Surrogate T3 → T4 objective — higher is better",
        TEXT_COLOR,
    )


def link_region_to_panels(
    fig: plt.Figure, ax_context: plt.Axes, axes_b: list[plt.Axes]
) -> None:
    """Tie the design-region box in (A) to the panels that magnify it.

    The B panels all show the same crop, so instead of three connectors they
    repeat the box's dashed frame and a single labelled arrow points down out
    of the box. A fan of straight connectors does not work in this layout: the
    box sits left of centre while the B row spans the full width, so the outer
    lines would run almost horizontally across panel (C) and both titles.
    """
    for ax in axes_b:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(WINNER_COLOR)
            spine.set_linewidth(1.6)
            spine.set_linestyle((0, (5, 3)))
            spine.set_alpha(0.75)

    midpoint = 0.5 * (X_BOUNDS[0] + X_BOUNDS[1])

    ax_context.annotate(
        "",
        xy=(midpoint, Y_BOUNDS[0] - 0.30),
        xytext=(midpoint, Y_BOUNDS[0]),
        xycoords="data",
        textcoords="data",
        annotation_clip=False,
        arrowprops={
            "arrowstyle": "-|>",
            "color": WINNER_COLOR,
            "linewidth": 1.6,
            "linestyle": (0, (4, 2.5)),
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 15,
        },
        zorder=8,
    )
    ax_context.text(
        midpoint + 0.12,
        Y_BOUNDS[0] - 0.20,
        "(B1) to (B3) below show this region",
        fontsize=10.2,
        fontweight="bold",
        color=WINNER_COLOR,
        va="center",
        ha="left",
        clip_on=False,
        zorder=8,
    )


def render(
    fig: plt.Figure,
    field: dict,
    story: dict,
    histories: dict,
    naive_upto: int,
    winner_upto: int,
    context_frame: int = -1,
    animated: bool = False,
) -> None:
    """Draw the whole composition at one point along the timeline."""
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.12, 1.0],
        left=0.03,
        right=0.988,
        top=0.885,
        bottom=0.175,
        wspace=0.13,
        hspace=0.34,
    )

    ax_context = fig.add_subplot(grid[0, 0:2])
    ax_objective = fig.add_subplot(grid[0, 2])
    axes_b = [fig.add_subplot(grid[1, column]) for column in range(3)]

    draw_context(ax_context, field, context_frame)
    draw_objective(ax_objective, histories, story, naive_upto, winner_upto)

    for ax in axes_b:
        draw_wake(ax, field, context=False)

    draw_probes(axes_b[0], story["baseline"], BASELINE_COLOR, "s")
    panel_title(
        axes_b[0],
        "(B1)",
        "Conventional baseline",
        "5 probes at one downstream station",
        BASELINE_COLOR,
    )
    metric_label(axes_b[0], story["cfd"]["baseline"], None, BASELINE_COLOR)

    naive_now = story["naive_path"][naive_upto]
    draw_traces(axes_b[1], story["naive_path"], naive_upto, NAIVE_COLOR)
    draw_probes(axes_b[1], naive_now, NAIVE_COLOR, "o")
    panel_title(
        axes_b[1],
        "(B2)",
        "Optimized from the rake",
        "weaker near-wake local solution",
        NAIVE_COLOR,
    )
    metric_label(
        axes_b[1],
        story["cfd"]["naive"],
        story["cfd"]["naive"] / story["cfd"]["baseline"] - 1.0,
        NAIVE_COLOR,
    )

    winner_now = story["winner_path"][winner_upto]
    draw_traces(axes_b[2], story["winner_path"], winner_upto, WINNER_COLOR)
    draw_probes(axes_b[2], winner_now, WINNER_COLOR, "o")
    panel_title(
        axes_b[2],
        "(B3)",
        "Winning multistart design",
        "best of 10 starts — probes split near and far",
        WINNER_COLOR,
    )
    metric_label(
        axes_b[2],
        story["cfd"]["winner"],
        story["cfd"]["winner"] / story["cfd"]["baseline"] - 1.0,
        WINNER_COLOR,
    )

    link_region_to_panels(fig, ax_context, axes_b)

    fig.suptitle(
        "From a conventional rake to more informative sensing",
        fontsize=21,
        fontweight="bold",
        y=0.975,
        color=TEXT_COLOR,
    )
    # The still shows one instant; the animation sweeps the whole run on an
    # index that is independent of the optimizer iteration.
    wake_note = (
        r"$^\circ$ rendering at $h=0.02$ for visualization only; panel (A) "
        r"spans $t=0$-$20$ on a timeline independent of the optimizer panels."
        if animated
        else r"$^\circ$, $t=20$ rendering at $h=0.02$ for visualization only."
    )
    fig.text(
        0.5,
        0.022,
        "Probe trajectories: actual optimizer iterates from the frozen "
        r"$h=0.05$ T3$\to$T4 campaign over the AoA design set." + "\n"
        "Wake: representative " + f"{ALPHA_DEG:g}" + wake_note,
        ha="center",
        fontsize=10.2,
        linespacing=1.5,
        color=MUTED,
    )


def main() -> None:
    """Validate provenance, then write the static hero and the GIF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Multistart-story hero")
    print("=" * 78)

    story = load_story()
    field = load_field()
    histories = objective_histories(story)

    naive_last = len(story["naive_path"]) - 1
    winner_last = len(story["winner_path"]) - 1

    print(f"  iterates: naive {naive_last + 1}, winner {winner_last + 1}")
    print(f"  ux display range: {field['ux_low']:.3f} .. {field['ux_high']:.3f}")
    print(
        f"  panel A snapshots: {len(field['omega_series'])} over "
        f"t = {field['times'][0]:g} .. {field['times'][-1]:g}"
    )

    written: list[Path] = []

    # ---- static: both trajectories complete ----
    fig = plt.figure(figsize=(15.5, 8.8))
    render(fig, field, story, histories, naive_last, winner_last, context_frame=-1)
    for suffix, kwargs in ((".png", {"dpi": 150}), (".pdf", {})):
        path = FIGURE_DIR / f"{STEM}{suffix}"
        fig.savefig(path, facecolor="white", **kwargs)
        written.append(path)
    plt.close(fig)

    # ---- animation on a normalized 0-100% timeline ----
    # Panel A's simulation time and the optimizer iteration are independent
    # quantities; the shared 0-100% index is a presentation device only.
    context_last = len(field["omega_series"]) - 1
    steps = [index / (MOTION_FRAMES - 1) for index in range(MOTION_FRAMES)]
    sequence = [
        (round(p * naive_last), round(p * winner_last), round(p * context_last))
        for p in steps
    ]

    fig = plt.figure(figsize=(15.5, 8.8), dpi=GIF_DPI)

    captured = []
    for naive_upto, winner_upto, context_frame in sequence:
        fig.clear()
        render(
            fig,
            field,
            story,
            histories,
            naive_upto,
            winner_upto,
            context_frame,
            animated=True,
        )
        fig.canvas.draw()
        captured.append(
            Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        )

    plt.close(fig)

    # One palette for every frame, applied without dithering. Matplotlib already
    # draws the static panels identically, but per-frame adaptive palettes and
    # dithering would re-quantize those pixels differently each time and make
    # the wake shimmer. A shared palette keeps unchanged pixels byte-identical,
    # which also lets the GIF encoder store only the region that actually moves.
    reference = captured[0].copy()
    width, height = reference.size
    montage = Image.new("RGB", (width, height * 3))
    for row, frame in enumerate(
        (captured[0], captured[len(captured) // 2], captured[-1])
    ):
        montage.paste(frame, (0, row * height))

    palette = montage.quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )

    frames = [
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in captured
    ]

    durations = [FRAME_MS] * (len(frames) - 1) + [HOLD_MS]

    path = FIGURE_DIR / f"{STEM}.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    written.append(path)

    print(
        f"  gif {len(frames)} frames, {FRAME_MS} ms each + {HOLD_MS} ms hold "
        f"= {(sum(durations)) / 1000:.1f} s"
    )

    for item in written:
        print(f"Wrote {item}")


if __name__ == "__main__":
    main()
