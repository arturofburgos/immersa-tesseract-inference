using Test

include(joinpath(@__DIR__, "..", "src", "ImmersaSolver.jl"))
using .ImmersaSolver


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

function velocity_rms_difference(a, b)
    sum_sq =
        sum(abs2, a.ux .- b.ux) +
        sum(abs2, a.uy .- b.uy)

    n =
        length(a.ux) +
        length(a.uy)

    return sqrt(sum_sq / n)
end


function radius_fd_sensitivity(
    R0,
    epsilon,
    n_ib_base;
    h,
    tf,
    snapshot_freq,
)
    result_minus = run_immersa_forward(
        R0 - epsilon;
        h = h,
        tf = tf,
        snapshot_freq = snapshot_freq,
        n_ib = n_ib_base,
    )

    result_plus = run_immersa_forward(
        R0 + epsilon;
        h = h,
        tf = tf,
        snapshot_freq = snapshot_freq,
        n_ib = n_ib_base,
    )

    dux_dR =
        (result_plus.ux .- result_minus.ux) ./ (2 * epsilon)

    duy_dR =
        (result_plus.uy .- result_minus.uy) ./ (2 * epsilon)

    n =
        length(dux_dR) +
        length(duy_dR)

    sensitivity_rms = sqrt(
        (
            sum(abs2, dux_dR) +
            sum(abs2, duy_dR)
        ) / n
    )

    return (
        dux_dR = dux_dR,
        duy_dR = duy_dR,
        sensitivity_rms = sensitivity_rms,
        result_minus = result_minus,
        result_plus = result_plus,
    )
end


function derivative_relative_difference(a, b)
    numerator =
        sum(abs2, a.dux_dR .- b.dux_dR) +
        sum(abs2, a.duy_dR .- b.duy_dR)

    denominator =
        sum(abs2, b.dux_dR) +
        sum(abs2, b.duy_dR)

    return sqrt(numerator / denominator)
end


# ---------------------------------------------------------------------------
# Shared development configuration
# ---------------------------------------------------------------------------

const H_TEST = 0.1
const TF_TEST = 1.0
const SNAPSHOT_FREQ_TEST = 20


# ---------------------------------------------------------------------------
# Test 1: Basic forward solve
# ---------------------------------------------------------------------------

@testset "Basic forward solve" begin

    result = run_immersa_forward(
        0.50;
        h = H_TEST,
        tf = TF_TEST,
        snapshot_freq = SNAPSHOT_FREQ_TEST,
    )

    @test result.radius == 0.50
    @test result.n_ib == 16

    @test size(result.ux) == (121, 60, 11)
    @test size(result.uy) == (120, 61, 11)

    @test length(result.ux_x) == 121
    @test length(result.ux_y) == 60

    @test length(result.uy_x) == 120
    @test length(result.uy_y) == 61

    @test length(result.times) == 11

    @test all(isfinite, result.ux)
    @test all(isfinite, result.uy)

    @test result.ds > 0.0
end


# ---------------------------------------------------------------------------
# Test 2: Geometry sensitivity and fixed Eulerian output shape
# ---------------------------------------------------------------------------

@testset "Geometry sensitivity" begin

    result_045 = run_immersa_forward(
        0.45;
        h = H_TEST,
        tf = TF_TEST,
        snapshot_freq = SNAPSHOT_FREQ_TEST,
    )

    result_050 = run_immersa_forward(
        0.50;
        h = H_TEST,
        tf = TF_TEST,
        snapshot_freq = SNAPSHOT_FREQ_TEST,
    )

    result_055 = run_immersa_forward(
        0.55;
        h = H_TEST,
        tf = TF_TEST,
        snapshot_freq = SNAPSHOT_FREQ_TEST,
    )

    # Internal immersed-boundary discretization adapts to geometry.
    @test result_045.n_ib == 14
    @test result_050.n_ib == 16
    @test result_055.n_ib == 17

    # External Eulerian output dimensions must remain fixed.
    @test size(result_045.ux) == size(result_050.ux)
    @test size(result_055.ux) == size(result_050.ux)

    @test size(result_045.uy) == size(result_050.uy)
    @test size(result_055.uy) == size(result_050.uy)

    # Changing the radius must change the wake.
    difference_045 = velocity_rms_difference(
        result_045,
        result_050,
    )

    difference_055 = velocity_rms_difference(
        result_055,
        result_050,
    )

    @test difference_045 > 0.0
    @test difference_055 > 0.0

    # Sanity check that the differences are not numerical roundoff.
    @test difference_045 > 1e-4
    @test difference_055 > 1e-4
end


# ---------------------------------------------------------------------------
# Test 3: Finite-difference discretization and convergence
# ---------------------------------------------------------------------------

@testset "Radius finite-difference sensitivity" begin

    R0 = 0.50

    n_ib_base = choose_marker_count(
        R0,
        H_TEST,
    )

    @test n_ib_base == 16

    epsilon_values = (
        0.01,
        0.005,
        0.0025,
    )

    sensitivities = Dict{Float64, Any}()

    for epsilon in epsilon_values

        sensitivity = radius_fd_sensitivity(
            R0,
            epsilon,
            n_ib_base;
            h = H_TEST,
            tf = TF_TEST,
            snapshot_freq = SNAPSHOT_FREQ_TEST,
        )

        sensitivities[epsilon] = sensitivity

        # The +epsilon and -epsilon simulations must use the same
        # immersed-boundary discretization.
        @test sensitivity.result_minus.n_ib == n_ib_base
        @test sensitivity.result_plus.n_ib == n_ib_base

        # Their Eulerian output shapes must remain identical.
        @test size(sensitivity.result_minus.ux) ==
              size(sensitivity.result_plus.ux)

        @test size(sensitivity.result_minus.uy) ==
              size(sensitivity.result_plus.uy)

        # All generated fields and derivatives must be finite.
        @test all(isfinite, sensitivity.result_minus.ux)
        @test all(isfinite, sensitivity.result_minus.uy)

        @test all(isfinite, sensitivity.result_plus.ux)
        @test all(isfinite, sensitivity.result_plus.uy)

        @test all(isfinite, sensitivity.dux_dR)
        @test all(isfinite, sensitivity.duy_dR)

        # Radius must have a measurable influence on the flow.
        @test sensitivity.sensitivity_rms > 0.0
    end


    difference_001_0005 =
        derivative_relative_difference(
            sensitivities[0.01],
            sensitivities[0.005],
        )

    difference_0005_00025 =
        derivative_relative_difference(
            sensitivities[0.005],
            sensitivities[0.0025],
        )

    # Central finite-difference derivative should stabilize
    # as epsilon is reduced.
    @test difference_0005_00025 < difference_001_0005

    # Loose regression bounds based on the validated development case.
    @test difference_001_0005 < 0.05
    @test difference_0005_00025 < 0.02
end
