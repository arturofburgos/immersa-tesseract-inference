# Surrogate sensor-array design

Frozen output of the differentiable sensor-design campaign. The array below was
selected **entirely from surrogate diagnostics**, before any real-CFD result was
inspected.

## Frozen design — `s_star_surrogate`

| | x | y |
|---|---:|---:|
| sensor 1 | 1.0 | -0.16925565522512323 |
| sensor 2 | 2.588428679688709 | 0.008902068445642432 |

Authoritative copy with full metadata: `optimization/s_star_surrogate.json`.

## How it was selected

1. Twenty L-BFGS-B starts (ten fixed, ten seeded random) against the frozen T4
   discrimination at `delta_alpha_min = 7.5 deg`, with `tau` calibrated once at
   the baseline array so the soft-minimum weight perplexity `N_eff = 10`, then
   held fixed.
2. The largest final `D_tau` wins.
3. **Tie-break.** WakeSurrogate runs in float32, so score differences below
   ~1e-6 carry no information. Layouts within `1e-6` of the maximum are treated
   as numerically tied; among tied layouts in the same solution cluster a
   normally converged optimizer termination is preferred over an abnormal one,
   then the fewest objective evaluations.

Three starts (`F03_streamwise`, `F07_antidiagonal`, `R04_random`) reached the
same optimum within `3.0e-07` in score and `3.5e-05` in layout. `R04_random`
held the bare maximum but terminated `ABNORMAL:`; the tie-break therefore
selected the cleanly converged `F03_streamwise`.

The rule uses optimizer diagnostics only. No physical or CFD quantity entered
the selection.

## Independence from the sealed truth

The design AoA grid is `20, 22.5, ..., 85 deg`. **63 deg is not on it** and was
never evaluated during design, so the array cannot have been tuned to the angle
it will later be validated against.

## Immutability

This layout must not be modified in response to real-CFD performance. If a
physically refined array is produced later it must be recorded separately as
`s_star_cfd_refined`, so the surrogate proposal and any physics refinement stay
scientifically distinguishable.

## Reproducing

```bash
python scripts/sensor_design/run_multistart_campaign.py   # ~6 min, writes the summary
python scripts/sensor_design/select_frozen_design.py      # applies the tie-break rule
python scripts/sensor_design/run_random_baseline.py       # 10,000-layout comparison
python scripts/sensor_design/run_delta_robustness.py      # evaluation only, no reselection
python scripts/sensor_design/plot_sensor_design.py        # figures A-D
```

Selection is a separate step from the campaign so the tie-break can be audited
and re-run without repeating the optimization.
