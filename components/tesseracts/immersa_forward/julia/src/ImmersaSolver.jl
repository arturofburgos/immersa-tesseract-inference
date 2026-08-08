module ImmersaSolver

using Immersa
using StaticArrays
using OffsetArrays
using LinearAlgebra

export choose_marker_count
export run_immersa_forward


"""
Choose the number of immersed-boundary markers using the same
spacing rule as the reference cylinder simulation:

    ds ≈ 2h
"""
function choose_marker_count(radius::Float64, h::Float64)
    radius > 0.0 || throw(ArgumentError("radius must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))

    circumference = 2π * radius

    return round(Int, circumference / (2h))
end


"""
Run the Immersa.jl forward problem for flow past a static circular body.

The simulation is deterministic for a given set of inputs.

`n_ib` is optional:
- if `nothing`, the marker count is selected from the current geometry;
- if supplied, that marker count is used explicitly.

The second behavior will later allow the finite-difference VJP to use the
same immersed-boundary discretization for theta+epsilon and theta-epsilon.

Returns selected ux/uy snapshots and their staggered-grid coordinates
entirely in memory.
"""
function run_immersa_forward(
    radius::Float64;
    h::Float64 = 0.1,
    dt::Float64 = 0.005,
    tf::Float64 = 1.0,
    Re::Float64 = 200.0,
    snapshot_freq::Int = 20,
    n_ib::Union{Nothing,Int} = nothing,
)

    radius > 0.0 || throw(ArgumentError("radius must be positive"))
    h > 0.0 || throw(ArgumentError("h must be positive"))
    dt > 0.0 || throw(ArgumentError("dt must be positive"))
    tf > 0.0 || throw(ArgumentError("tf must be positive"))
    snapshot_freq > 0 || throw(ArgumentError("snapshot_freq must be positive"))

    # ------------------------------------------------------------------
    # 1. Eulerian grid
    # ------------------------------------------------------------------

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
                (gridlims[:, 2] - gridlims[:, 1]) / h,
            )
        ),
        x0 = gridlims[:, 1],
        levels = 2,
    )

    # ------------------------------------------------------------------
    # 2. Circular immersed body
    # ------------------------------------------------------------------

    circumference = 2π * radius

    n_markers = if isnothing(n_ib)
        choose_marker_count(radius, h)
    else
        n_ib
    end

    n_markers > 0 || throw(ArgumentError("n_ib must be positive"))

    ds = circumference / n_markers

    body = let
        θ_values = range(0, 2π, n_markers + 1)[1:(end - 1)]

        x = map(θ_values) do θ
            radius * SA[
                cos(θ),
                sin(θ),
            ]
        end

        StaticBody(
            x,
            fill(ds, n_markers),
        )
    end

    # ------------------------------------------------------------------
    # 3. Immersed-boundary problem
    # ------------------------------------------------------------------

    u0 = UniformFlow(t -> SA[1.0, 0.0])

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

    # ------------------------------------------------------------------
    # 4. Deterministic initial perturbation
    #
    # Preserve the exact perturbation used by the reference simulation.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 5. Snapshot bookkeeping
    # ------------------------------------------------------------------

    n_steps = round(Int, tf / dt)

    i_all = 1:(1 + n_steps)

    i_snapshot = i_all[1:snapshot_freq:end]

    n_snapshots = length(i_snapshot)

    # ------------------------------------------------------------------
    # 6. Velocity arrays
    #
    # Level 1 is the finest level used by the reference script.
    # ------------------------------------------------------------------

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

    times = Vector{Float64}(undef, n_snapshots)

    # ------------------------------------------------------------------
    # 7. Staggered velocity coordinates
    #
    # ux and uy do NOT live at the same physical locations.
    # ------------------------------------------------------------------

    function velocity_coordinates(loc, arr)
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

    ux_coords = velocity_coordinates(
        Loc_u(1),
        ux_state,
    )

    uy_coords = velocity_coordinates(
        Loc_u(2),
        uy_state,
    )

    # ------------------------------------------------------------------
    # 8. Time integration
    # ------------------------------------------------------------------

    for _ in 0:n_steps

        step!(sol)

        if sol.i in i_snapshot

            snapshot_index =
                1 +
                (sol.i - first(i_snapshot)) ÷ step(i_snapshot)

            times[snapshot_index] = sol.t

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

    # ------------------------------------------------------------------
    # 9. Return only data useful to the forward-model interface
    # ------------------------------------------------------------------

    return (
        ux = ux,
        uy = uy,
        times = times,

        ux_x = ux_coords.x,
        ux_y = ux_coords.y,

        uy_x = uy_coords.x,
        uy_y = uy_coords.y,

        radius = radius,
        n_ib = n_markers,
        ds = ds,

        h = h,
        dt = dt,
        tf = tf,
        Re = Re,
    )
end


end
