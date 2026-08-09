module ImmersaForward

include(joinpath(@__DIR__, "ImmersaSolver.jl"))

using .ImmersaSolver: run_immersa_forward

export run_forward


function run_forward(
    radius::Float64;
    h::Float64 = 0.1,
    dt::Float64 = 0.005,
    tf::Float64 = 1.0,
    Re::Float64 = 200.0,
    snapshot_freq::Int = 20,
)
    return run_immersa_forward(
        radius;
        h = h,
        dt = dt,
        tf = tf,
        Re = Re,
        snapshot_freq = snapshot_freq,
    )
end


end # module ImmersaForward
