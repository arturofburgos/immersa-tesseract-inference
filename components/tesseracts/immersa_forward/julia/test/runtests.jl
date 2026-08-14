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
    result_minus = run_cylinder_forward(
        R0 - epsilon;
        h = h,
        tf = tf,
        snapshot_freq = snapshot_freq,
        n_ib = n_ib_base,
    )

    result_plus = run_cylinder_forward(
        R0 + epsilon;
        h = h,
        tf = tf,
        snapshot_freq = snapshot_freq,
        n_ib = n_ib_base,
    )

    dux =
        (result_plus.ux .- result_minus.ux) ./ (2 * epsilon)

    duy =
        (result_plus.uy .- result_minus.uy) ./ (2 * epsilon)

    n =
        length(dux) +
        length(duy)

    sensitivity_rms = sqrt(
        (
            sum(abs2, dux) +
            sum(abs2, duy)
        ) / n
    )

    return (
        dux = dux,
        duy = duy,
        sensitivity_rms = sensitivity_rms,
        result_minus = result_minus,
        result_plus = result_plus,
    )
end


function aoa_fd_sensitivity(
    alpha0,
    epsilon;
    h,
    tf,
    snapshot_freq,
)
    result_minus = run_plate_forward(
        alpha0 - epsilon;
        h = h,
        tf = tf,
        snapshot_freq = snapshot_freq,
    )

    result_plus = run_plate_forward(
        alpha0 + epsilon;
        h = h,
        tf = tf,
        snapshot_freq = snapshot_freq,
    )

    # angle_of_attack is currently measured in degrees,
    # so these sensitivities are velocity / degree.
    dux =
        (result_plus.ux .- result_minus.ux) ./ (2 * epsilon)

    duy =
        (result_plus.uy .- result_minus.uy) ./ (2 * epsilon)

    n =
        length(dux) +
        length(duy)

    sensitivity_rms = sqrt(
        (
            sum(abs2, dux) +
            sum(abs2, duy)
        ) / n
    )

    return (
        dux = dux,
        duy = duy,
        sensitivity_rms = sensitivity_rms,
        result_minus = result_minus,
        result_plus = result_plus,
    )
end


function derivative_relative_difference(a, b)
    numerator =
        sum(abs2, a.dux .- b.dux) +
        sum(abs2, a.duy .- b.duy)

    denominator =
        sum(abs2, b.dux) +
        sum(abs2, b.duy)

    return sqrt(numerator / denominator)
end


# ---------------------------------------------------------------------------
# Shared development configuration
# ---------------------------------------------------------------------------

const H_TEST = 0.1
const TF_TEST = 1.0
const SNAPSHOT_FREQ_TEST = 20


# ===========================================================================
# CYLINDER REGRESSION TESTS
#
# Keep these because the cylinder is our previously validated reference case.
# ===========================================================================


# ---------------------------------------------------------------------------
# Cylinder Test 1: Basic forward solve
# ---------------------------------------------------------------------------

@testset "Cylinder basic forward solve" begin

    result = run_cylinder_forward(
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
# Cylinder Test 2: Geometry sensitivity
# ---------------------------------------------------------------------------

@testset "Cylinder geometry sensitivity" begin

    result_045 = run_cylinder_forward(
        0.45;
        h = H_TEST,
        tf = TF_TEST,
        snapshot_freq = SNAPSHOT_FREQ_TEST,
    )

    result_050 = run_cylinder_forward(
        0.50;
        h = H_TEST,
        tf = TF_TEST,
        snapshot_freq = SNAPSHOT_FREQ_TEST,
    )

    result_055 = run_cylinder_forward(
        0.55;
        h = H_TEST,
        tf = TF_TEST,
        snapshot_freq = SNAPSHOT_FREQ_TEST,
    )

    # Cylinder marker count changes with circumference.
    @test result_045.n_ib == 14
    @test result_050.n_ib == 16
    @test result_055.n_ib == 17

    # Eulerian output dimensions remain fixed.
    @test size(result_045.ux) == size(result_050.ux)
    @test size(result_055.ux) == size(result_050.ux)

    @test size(result_045.uy) == size(result_050.uy)
    @test size(result_055.uy) == size(result_050.uy)

    difference_045 = velocity_rms_difference(
        result_045,
        result_050,
    )

    difference_055 = velocity_rms_difference(
        result_055,
        result_050,
    )

    @test difference_045 > 1e-4
    @test difference_055 > 1e-4
end


# ---------------------------------------------------------------------------
# Cylinder Test 3: Radius finite-difference sensitivity
# ---------------------------------------------------------------------------

@testset "Cylinder radius finite-difference sensitivity" begin

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

        @test sensitivity.result_minus.n_ib == n_ib_base
        @test sensitivity.result_plus.n_ib == n_ib_base

        @test size(sensitivity.result_minus.ux) ==
              size(sensitivity.result_plus.ux)

        @test size(sensitivity.result_minus.uy) ==
              size(sensitivity.result_plus.uy)

        @test all(isfinite, sensitivity.result_minus.ux)
        @test all(isfinite, sensitivity.result_minus.uy)

        @test all(isfinite, sensitivity.result_plus.ux)
        @test all(isfinite, sensitivity.result_plus.uy)

        @test all(isfinite, sensitivity.dux)
        @test all(isfinite, sensitivity.duy)

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

    @test difference_0005_00025 < difference_001_0005

    @test difference_001_0005 < 0.05
    @test difference_0005_00025 < 0.02
end


# ===========================================================================
# PLATE TESTS
#
# These are now the important scientific tests for the hackathon problem.
# ===========================================================================


# ---------------------------------------------------------------------------
# Plate Test 1: Basic forward solve
# ---------------------------------------------------------------------------

@testset "Plate basic forward solve" begin

    result = run_plate_forward(
        60.0;
        h = H_TEST,
        tf = TF_TEST,
        snapshot_freq = SNAPSHOT_FREQ_TEST,
    )

    @test result.geometry == "plate"

    @test result.angle_of_attack_deg == 60.0

    @test result.plate_length == 1.0

    @test result.mid_chord_x ≈ 0.0
    @test result.mid_chord_y ≈ 0.0

    @test result.n_ib == 5
    @test result.ds ≈ 0.2

    @test size(result.ux) == (73, 32, 11)
    @test size(result.uy) == (72, 33, 11)

    @test length(result.ux_x) == 73
    @test length(result.ux_y) == 32

    @test length(result.uy_x) == 72
    @test length(result.uy_y) == 33

    @test length(result.times) == 11

    @test all(isfinite, result.ux)
    @test all(isfinite, result.uy)

    @test all(isfinite, result.marker_x)
    @test all(isfinite, result.marker_y)
end


# ---------------------------------------------------------------------------
# Plate Test 2: Geometry invariance and AoA dependence
# ---------------------------------------------------------------------------

@testset "Plate geometry and AoA dependence" begin

    result_030 = run_plate_forward(
        30.0;
        h = H_TEST,
        tf = TF_TEST,
        snapshot_freq = SNAPSHOT_FREQ_TEST,
    )

    result_060 = run_plate_forward(
        60.0;
        h = H_TEST,
        tf = TF_TEST,
        snapshot_freq = SNAPSHOT_FREQ_TEST,
    )

    result_090 = run_plate_forward(
        90.0;
        h = H_TEST,
        tf = TF_TEST,
        snapshot_freq = SNAPSHOT_FREQ_TEST,
    )

    # ---------------------------------------------------------------
    # Only orientation should change.
    # ---------------------------------------------------------------

    @test result_030.n_ib == 5
    @test result_060.n_ib == 5
    @test result_090.n_ib == 5

    @test result_030.ds ≈ 0.2
    @test result_060.ds ≈ 0.2
    @test result_090.ds ≈ 0.2

    @test result_030.plate_length ≈ 1.0
    @test result_060.plate_length ≈ 1.0
    @test result_090.plate_length ≈ 1.0

    @test result_030.mid_chord_x ≈ 0.0
    @test result_060.mid_chord_x ≈ 0.0
    @test result_090.mid_chord_x ≈ 0.0

    @test result_030.mid_chord_y ≈ 0.0
    @test result_060.mid_chord_y ≈ 0.0
    @test result_090.mid_chord_y ≈ 0.0

    # ---------------------------------------------------------------
    # Eulerian representation must remain identical.
    # ---------------------------------------------------------------

    @test size(result_030.ux) == size(result_060.ux)
    @test size(result_060.ux) == size(result_090.ux)

    @test size(result_030.uy) == size(result_060.uy)
    @test size(result_060.uy) == size(result_090.uy)

    # ---------------------------------------------------------------
    # Different AoAs must produce measurably different wakes.
    # ---------------------------------------------------------------

    difference_030_060 =
        velocity_rms_difference(
            result_030,
            result_060,
        )

    difference_060_090 =
        velocity_rms_difference(
            result_060,
            result_090,
        )

    difference_030_090 =
        velocity_rms_difference(
            result_030,
            result_090,
        )

    @test difference_030_060 > 1e-3
    @test difference_060_090 > 1e-3
    @test difference_030_090 > 1e-3

    # Regression sanity bounds based on the validated development case.
    @test difference_030_060 < 1.0
    @test difference_060_090 < 1.0
    @test difference_030_090 < 1.0
end


# ---------------------------------------------------------------------------
# Plate Test 3: AoA finite-difference sensitivity
# ---------------------------------------------------------------------------

@testset "Plate AoA finite-difference sensitivity" begin

    alpha0 = 60.0

    epsilon_values = (
        2.0,
        1.0,
        0.5,
        0.25,
    )

    sensitivities = Dict{Float64, Any}()

    for epsilon in epsilon_values

        sensitivity = aoa_fd_sensitivity(
            alpha0,
            epsilon;
            h = H_TEST,
            tf = TF_TEST,
            snapshot_freq = SNAPSHOT_FREQ_TEST,
        )

        sensitivities[epsilon] = sensitivity

        # -----------------------------------------------------------
        # Unlike the cylinder radius case, plate discretization
        # should naturally remain fixed for every AoA.
        # -----------------------------------------------------------

        @test sensitivity.result_minus.n_ib == 5
        @test sensitivity.result_plus.n_ib == 5

        @test sensitivity.result_minus.ds ≈ 0.2
        @test sensitivity.result_plus.ds ≈ 0.2

        # -----------------------------------------------------------
        # Eulerian output shape remains fixed.
        # -----------------------------------------------------------

        @test size(sensitivity.result_minus.ux) ==
              size(sensitivity.result_plus.ux)

        @test size(sensitivity.result_minus.uy) ==
              size(sensitivity.result_plus.uy)

        # -----------------------------------------------------------
        # Fields and sensitivities must be finite.
        # -----------------------------------------------------------

        @test all(isfinite, sensitivity.result_minus.ux)
        @test all(isfinite, sensitivity.result_minus.uy)

        @test all(isfinite, sensitivity.result_plus.ux)
        @test all(isfinite, sensitivity.result_plus.uy)

        @test all(isfinite, sensitivity.dux)
        @test all(isfinite, sensitivity.duy)

        # AoA must have a measurable influence on the wake.
        @test sensitivity.sensitivity_rms > 1e-4
    end


    difference_2_1 =
        derivative_relative_difference(
            sensitivities[2.0],
            sensitivities[1.0],
        )

    difference_1_05 =
        derivative_relative_difference(
            sensitivities[1.0],
            sensitivities[0.5],
        )

    difference_05_025 =
        derivative_relative_difference(
            sensitivities[0.5],
            sensitivities[0.25],
        )


    # ---------------------------------------------------------------
    # Central finite-difference estimate should stabilize as epsilon
    # decreases.
    # ---------------------------------------------------------------

    @test difference_1_05 < difference_2_1
    @test difference_05_025 < difference_1_05


    # ---------------------------------------------------------------
    # Loose regression bounds based on the development experiment:
    #
    # 2.0 -> 1.0    ~ 1.76e-2
    # 1.0 -> 0.5    ~ 4.48e-3
    # 0.5 -> 0.25   ~ 1.13e-3
    # ---------------------------------------------------------------

    @test difference_2_1 < 0.03
    @test difference_1_05 < 0.01
    @test difference_05_025 < 0.005


    # ---------------------------------------------------------------
    # Approximate second-order convergence check.
    #
    # For central finite differences, halving epsilon should reduce
    # the truncation error by approximately a factor of four.
    #
    # Keep bounds intentionally loose so the regression is not brittle.
    # ---------------------------------------------------------------

    convergence_ratio_1 =
        difference_2_1 /
        difference_1_05

    convergence_ratio_2 =
        difference_1_05 /
        difference_05_025

    @test 3.9 < convergence_ratio_1 < 4.1
    @test 3.9 < convergence_ratio_2 < 4.1
end
