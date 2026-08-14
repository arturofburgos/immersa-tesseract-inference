module ImmersaSolver

using Immersa
using StaticArrays
using OffsetArrays
using LinearAlgebra

export choose_marker_count
export choose_plate_marker_count

export run_immersa_forward
export run_cylinder_forward
export run_plate_forward


# ================================================================
# Marker-count utilities
# ================================================================

"""
Choose the number of immersed-boundary markers for a circle using

    ds ≈ 2h
"""
function choose_marker_count(
    radius::Float64,
    h::Float64,
)
    radius > 0.0 ||
        throw(ArgumentError("radius must be positive"))

    h > 0.0 ||
        throw(ArgumentError("h must be positive"))

    circumference = 2π * radius

    return round(
        Int,
        circumference / (2h),
    )
end


"""
Choose the number of immersed-boundary markers for a flat plate using

    ds ≈ 2h

Because the plate length is fixed, this marker count is independent
of angle of attack.
"""
function choose_plate_marker_count(
    plate_length::Float64,
    h::Float64,
)
    plate_length > 0.0 ||
        throw(ArgumentError("plate_length must be positive"))

    h > 0.0 ||
        throw(ArgumentError("h must be positive"))

    return round(
        Int,
        plate_length / (2h),
    )
end


# ================================================================
# Shared solver utilities
# ================================================================

"""
Return the physical coordinates of one staggered velocity component.
"""
function velocity_coordinates(
    grid,
    loc,
    arr,
)
    Nx, Ny = size(arr)

    i0, j0 = first.(axes(arr))

    xi = i0:(i0 + Nx - 1)
    yi = j0:(j0 + Ny - 1)

    X, Y = coord(
        grid,
        loc,
        (xi, yi),
        1,
    )

    return (
        x = collect(X[:, 1]),
        y = collect(Y[:, 1]),
    )
end


"""
Apply the deterministic vorticity perturbation used in the original
Immersa example to trigger vortex shedding.
"""
function apply_initial_perturbation!(
    sol,
    grid,
)
    map!(
        sol.ω[1][3],
        CartesianIndices(sol.ω[1][3]),
    ) do I

        x = coord(
            grid,
            Loc_ω(3),
            I,
        )

        p = x - SA[-0.75, 0.0]

        perturbation_radius = 0.25

        0.5 * (
            1 - clamp(
                norm(p) / perturbation_radius,
                0,
                1,
            )
        )
    end

    apply_vorticity!(sol)

    return nothing
end


"""
Run the common Immersa time integration and collect finest-level
staggered ux/uy snapshots entirely in memory.
"""
function solve_forward(
    grid,
    body;
    h::Float64,
    dt::Float64,
    tf::Float64,
    Re::Float64,
    snapshot_freq::Int,
)
    h > 0.0 ||
        throw(ArgumentError("h must be positive"))

    dt > 0.0 ||
        throw(ArgumentError("dt must be positive"))

    tf > 0.0 ||
        throw(ArgumentError("tf must be positive"))

    snapshot_freq > 0 ||
        throw(ArgumentError("snapshot_freq must be positive"))

    # --------------------------------------------------------------
    # Immersed-boundary problem
    # --------------------------------------------------------------

    u0 = UniformFlow(
        t -> SA[1.0, 0.0],
    )

    prob = IBProblem(
        grid,
        body,
        Re,
        u0,
    )

    sol = CNAB(
        prob;
        dt,
        delta = Immersa.DeltaYang3S2(),
    )

    # --------------------------------------------------------------
    # Deterministic perturbation
    # --------------------------------------------------------------

    apply_initial_perturbation!(
        sol,
        grid,
    )

    # --------------------------------------------------------------
    # Snapshot bookkeeping
    # --------------------------------------------------------------

    n_steps = round(
        Int,
        tf / dt,
    )

    i_all = 1:(1 + n_steps)

    i_snapshot =
        i_all[1:snapshot_freq:end]

    n_snapshots =
        length(i_snapshot)

    # --------------------------------------------------------------
    # Velocity arrays
    # --------------------------------------------------------------

    ux_state = sol.u[1][1]
    uy_state = sol.u[1][2]

    ux = Array{Float64}(
        undef,
        size(ux_state)...,
        n_snapshots,
    )

    uy = Array{Float64}(
        undef,
        size(uy_state)...,
        n_snapshots,
    )

    times = Vector{Float64}(
        undef,
        n_snapshots,
    )

    # --------------------------------------------------------------
    # Staggered coordinates
    # --------------------------------------------------------------

    ux_coords = velocity_coordinates(
        grid,
        Loc_u(1),
        ux_state,
    )

    uy_coords = velocity_coordinates(
        grid,
        Loc_u(2),
        uy_state,
    )

    # --------------------------------------------------------------
    # Time integration
    # --------------------------------------------------------------

    for _ in 0:n_steps

        step!(sol)

        if sol.i in i_snapshot

            snapshot_index =
                1 +
                (sol.i - first(i_snapshot)) ÷
                step(i_snapshot)

            times[snapshot_index] =
                sol.t

            ux[:, :, snapshot_index] .=
                OffsetArrays.no_offset_view(
                    sol.u[1][1],
                )

            uy[:, :, snapshot_index] .=
                OffsetArrays.no_offset_view(
                    sol.u[1][2],
                )
        end
    end

    return (
        ux = ux,
        uy = uy,
        times = times,

        ux_x = ux_coords.x,
        ux_y = ux_coords.y,

        uy_x = uy_coords.x,
        uy_y = uy_coords.y,

        h = h,
        dt = dt,
        tf = tf,
        Re = Re,
    )
end


# ================================================================
# Cylinder
# ================================================================

"""
Run flow past a static circular body.

This remains in the codebase as the validated regression case.
"""
function run_cylinder_forward(
    radius::Float64;
    h::Float64 = 0.1,
    dt::Float64 = 0.005,
    tf::Float64 = 1.0,
    Re::Float64 = 200.0,
    snapshot_freq::Int = 20,
    n_ib::Union{Nothing,Int} = nothing,
)

    radius > 0.0 ||
        throw(ArgumentError("radius must be positive"))

    # --------------------------------------------------------------
    # Eulerian grid
    # --------------------------------------------------------------

    gridlims = SA[
        -2.0  10.0
        -3.0   3.0
    ]

    grid = Grid(
        ;
        h,
        n = @.(
            round(
                Int,
                (gridlims[:, 2] -
                 gridlims[:, 1]) / h,
            )
        ),
        x0 = gridlims[:, 1],
        levels = 2,
    )

    # --------------------------------------------------------------
    # Circular body
    # --------------------------------------------------------------

    circumference =
        2π * radius

    n_markers =
        if isnothing(n_ib)
            choose_marker_count(
                radius,
                h,
            )
        else
            n_ib
        end

    n_markers > 0 ||
        throw(ArgumentError("n_ib must be positive"))

    ds =
        circumference / n_markers

    θ_values =
        range(
            0,
            2π,
            n_markers + 1,
        )[1:(end - 1)]

    marker_positions =
        map(θ_values) do θ
            radius * SA[
                cos(θ),
                sin(θ),
            ]
        end

    body = StaticBody(
        marker_positions,
        fill(ds, n_markers),
    )

    result = solve_forward(
        grid,
        body;
        h = h,
        dt = dt,
        tf = tf,
        Re = Re,
        snapshot_freq = snapshot_freq,
    )

    return (
        result...,

        geometry = "cylinder",

        radius = radius,

        n_ib = n_markers,
        ds = ds,
    )
end


# ================================================================
# Flat plate
# ================================================================

"""
Run flow past a fixed flat plate.

The plate:

- has fixed length,
- has fixed mid-chord location,
- has fixed immersed-boundary discretization,
- changes only through angle of attack.

`angle_of_attack_deg` is specified in degrees.

The sign convention matches the native Immersa `plate.jl` example:

    θ = deg2rad(-angle_of_attack_deg)

The plate rotates rigidly about its mid-chord.
"""
function run_plate_forward(
    angle_of_attack_deg::Float64;
    plate_length::Float64 = 1.0,
    h::Float64 = 0.1,
    dt::Float64 = 0.005,
    tf::Float64 = 1.0,
    Re::Float64 = 200.0,
    snapshot_freq::Int = 20,
    n_ib::Union{Nothing,Int} = nothing,
)

    plate_length > 0.0 ||
        throw(
            ArgumentError(
                "plate_length must be positive",
            ),
        )

    # --------------------------------------------------------------
    # Eulerian grid
    #
    # For now use the same domain as the working native plate
    # example. The scientific resolution can be increased later.
    # --------------------------------------------------------------

    gridlims = SA[
        -1.0   6.0
        -1.5   1.5
    ]

    grid = Grid(
        ;
        h,
        n = @.(
            round(
                Int,
                (gridlims[:, 2] -
                 gridlims[:, 1]) / h,
            )
        ),
        x0 = gridlims[:, 1],
        levels = 2,
    )

    # --------------------------------------------------------------
    # Fixed-mid-chord flat plate
    # --------------------------------------------------------------

    mid_chord =
        SA[0.0, 0.0]

    n_markers =
        if isnothing(n_ib)
            choose_plate_marker_count(
                plate_length,
                h,
            )
        else
            n_ib
        end

    n_markers > 0 ||
        throw(
            ArgumentError(
                "n_ib must be positive",
            ),
        )

    # Because L is fixed, ds and n_ib are independent of AoA.
    ds =
        plate_length / n_markers

    angle_rad =
        deg2rad(
            -angle_of_attack_deg,
        )

    # Midpoint discretization:
    #
    # each marker represents one segment of length ds.
    #
    # This also guarantees that changing AoA rotates the same
    # Lagrangian discretization rigidly about the mid-chord.
    s_values =
        range(
            -plate_length / 2 + ds / 2,
             plate_length / 2 - ds / 2;
            length = n_markers,
        )

    marker_positions =
        map(s_values) do s

            mid_chord +
            s * SA[
                cos(angle_rad),
                sin(angle_rad),
            ]
        end

    body = StaticBody(
        marker_positions,
        fill(ds, n_markers),
    )

    # --------------------------------------------------------------
    # Solve
    # --------------------------------------------------------------

    result = solve_forward(
        grid,
        body;
        h = h,
        dt = dt,
        tf = tf,
        Re = Re,
        snapshot_freq = snapshot_freq,
    )

    # --------------------------------------------------------------
    # Marker coordinates are useful for geometry tests.
    #
    # We do not necessarily need to expose them through the final
    # public Tesseract API.
    # --------------------------------------------------------------

    marker_x =
        Float64[
            x[1]
            for x in marker_positions
        ]

    marker_y =
        Float64[
            x[2]
            for x in marker_positions
        ]

    return (
        result...,

        geometry = "plate",

        angle_of_attack_deg =
            angle_of_attack_deg,

        plate_length =
            plate_length,

        mid_chord_x =
            mid_chord[1],

        mid_chord_y =
            mid_chord[2],

        marker_x =
            marker_x,

        marker_y =
            marker_y,

        n_ib =
            n_markers,

        ds =
            ds,
    )
end


# ================================================================
# Backward compatibility
# ================================================================

"""
Compatibility wrapper for the original cylinder API.

For now this keeps all existing cylinder tests and the current
Tesseract interface working while the plate implementation is
validated.

Later the public ImmersaForward Tesseract will be changed to call
`run_plate_forward`.
"""
function run_immersa_forward(
    radius::Float64;
    kwargs...
)
    return run_cylinder_forward(
        radius;
        kwargs...
    )
end


end
