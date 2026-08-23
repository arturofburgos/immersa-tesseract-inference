BASE_TRAIN_ALPHAS = (
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
    90.0,
)

HIGH_AOA_REFINEMENT = (
    78.75,
    81.25,
    83.75,
    86.25,
)

BROAD_AOA_REFINEMENT = (
    27.5,
    32.5,
    42.5,
    47.5,
    57.5,
    62.5,
    72.5,
    77.5,
    80.625,
    81.875,
    83.125,
    84.375,
)

TRAIN_ALPHAS_BY_VERSION = {
    "v1": BASE_TRAIN_ALPHAS,

    "v2": (
        BASE_TRAIN_ALPHAS
        + HIGH_AOA_REFINEMENT
    ),

    "v3": (
        BASE_TRAIN_ALPHAS
        + HIGH_AOA_REFINEMENT
        + BROAD_AOA_REFINEMENT
    ),
}

VALIDATION_ALPHAS = (
    22.5,
    37.5,
    52.5,
    67.5,
    82.5,
)

TEST_ALPHAS = (
    63.0,
)

SENSOR_TIMES = (
    12.0,
    13.3,
    15.1,
    17.4,
    20.0,
)