# Project Journey Summary

## Current Position

- The project has moved from the older monolithic research pipeline toward a modular walk-forward engine built around `main.py` and `src/`.
- The current modeling direction is `GJR-GARCH + EVT + copula`, aligned with the reference paper and the implementation plan.
- Core robustness fixes have already been completed, and the first validation harness is now in place.

## What Has Been Built

- Modular engine under `src/` for:
  - configuration
  - data loading and caching
  - preprocessing
  - walk-forward backtesting
  - volatility modeling
  - EVT tail modeling
  - copula portfolio simulation
  - reporting
  - strategy constraints
- Custom `gjrgarch_fast.py` engine integrated into `src/volatility.py`.
- Regression tests for:
  - allocation constraints
  - volatility fallback handling
  - strict JSON reporting
  - risk-metric calculations

## Key Fixes Completed

- Added fallback handling when volatility fits fail or produce invalid outputs.
- Stopped broken fits from silently flowing through the backtest.
- Replaced the previous allocator with a deterministic dollar-neutral rule that respects net, gross, and position bounds.
- Added Kupiec and Christoffersen exception backtests plus exception-cluster counting.
- Made `summary.json` outputs strict-JSON safe.
- Hardened copula correlation estimation for degenerate residual panels.

## Validation Milestones

### Milestone 1

- Hardened the walk-forward engine.
- Added regression tests.
- Pushed first milestone commit:
  - `2be8b06`
  - `Initial modular GJR-GARCH risk engine baseline`

### Milestone 2

- Added a package-backed plain-GARCH baseline using `arch`.
- Added `compare_walkforward_models.py` for apples-to-apples walk-forward comparison.
- Ran the first real cached-data comparison on `us_stress`.
- Pushed validation milestone commit:
  - `752053f`
  - `Add walk-forward GARCH vs GJR validation harness`

## First Real Comparison Result

Dataset:
- Basket: `us_stress`
- Training window: `500` days
- Out-of-sample slice used for first validation: `60` days

Comparison:
- `garch_baseline`
- `gjr_custom`

Observed result:
- Both models had `1` VaR breach over `60` observations.
- Both passed Kupiec and Christoffersen on this bounded slice.
- Custom GJR was more operationally stable:
  - plain GARCH fallback rate: `0.0194`
  - custom GJR fallback rate: `0.0000`

Artifacts:
- `results_validation/us_stress_2021-10-08_2023-12-29/comparison.md`
- `results_validation/us_stress_2021-10-08_2023-12-29/comparison.json`

## Second Real Comparison Result

Dataset:
- Basket: `india_primary`
- Training window: `500` days
- Out-of-sample slice used for validation: `60` days

Comparison:
- `garch_baseline`
- `gjr_custom`

Observed result:
- Plain GARCH had `5` VaR breaches over `60` observations.
- Custom GJR had `1` VaR breach over `60` observations.
- Plain GARCH failed Kupiec calibration on this slice:
  - Kupiec p-value: `0.00036`
- Custom GJR remained well-calibrated:
  - Kupiec p-value: `0.6357`
- Custom GJR also remained more stable operationally:
  - plain GARCH fallback rate: `0.1367`
  - custom GJR fallback rate: `0.0000`

Interpretation:
- This is the strongest validation result so far.
- On `india_primary`, custom GJR is not only more stable, it is also clearly better calibrated than the plain-GARCH baseline in the bounded out-of-sample test.

Artifacts:
- `results_validation/india_primary_2021-09-24_2023-12-29/comparison.md`
- `results_validation/india_primary_2021-09-24_2023-12-29/comparison.json`

## Where The Project Was Last Left

- The walk-forward GARCH vs GJR validation harness is complete for both baskets on recent bounded out-of-sample slices.
- Current cross-basket picture:
  - `us_stress`: tie on breaches, GJR more stable operationally
  - `india_primary`: GJR clearly better on breaches, calibration, and stability

## Next Steps

- Expand the GARCH vs GJR comparison to a larger out-of-sample window.
- Investigate whether package-baseline convergence warnings are tuning-related or structural.
- Only after stronger validation, decide whether to add regime logic into the new canonical stack.

## Widened 120-Day Comparison

The comparison has now been widened to 120 out-of-sample days for both baskets.

### `us_stress`

- Plain GARCH:
  - `2` VaR breaches over `120` observations
  - fallback rate: `0.0181`
- Custom GJR:
  - `0` VaR breaches over `120` observations
  - fallback rate: `0.0000`

Interpretation:
- On the wider `us_stress` window, GJR moved from “similar breach count but more stable” to clearly better on both stability and breaches.

Artifacts:
- `results_validation/us_stress_2021-07-15_2023-12-29/comparison.md`

### `india_primary`

- Plain GARCH:
  - `11` VaR breaches over `120` observations
  - breach rate: `0.0917`
  - Kupiec p-value: `4.38e-08`
  - fallback rate: `0.1217`
- Custom GJR:
  - `3` VaR breaches over `120` observations
  - breach rate: `0.0250`
  - Kupiec p-value: `0.1653`
  - fallback rate: `0.0000`

Interpretation:
- The widened `india_primary` window strengthens the earlier result.
- Plain GARCH remains badly calibrated, while custom GJR remains far more stable and materially better calibrated.

Artifacts:
- `results_validation/india_primary_2021-06-28_2023-12-29/comparison.md`

## Updated Position

The project is now past the point of “promising direction only.”

Current evidence across the widened validation windows says:
- custom GJR is more stable than the plain-GARCH benchmark on both baskets
- custom GJR is better calibrated on `india_primary`
- custom GJR is now also better on breaches for `us_stress`

## Baseline Investigation Update

The package-backed plain-GARCH baseline was investigated because it was generating too many warnings and fallbacks.

What caused it:
- the wrapper was fitting `arch` on small decimal log returns
- it was using Student-t innovations with `rescale=False`
- that combination was causing many optimizer failures that were not inherent to plain GARCH itself

What fixed it:
- enabling `rescale=True` in the `arch` wrapper

What changed after the fix:
- previously failed windows recovered almost completely under internal rescaling
- the known bad windows for `JPM` and `^NSEI` fit cleanly after the wrapper change

Important consequence:
- some earlier conclusions that favored custom GJR were partly inflated by a wrapper issue in the baseline benchmark
- after fixing the baseline scaling and rerunning the 60-day slices:
  - `us_stress`: rescaled plain GARCH had `0` breaches vs `1` for GJR
  - `india_primary`: rescaled plain GARCH had `0` breaches vs `1` for GJR

So the current honest position is:
- the baseline instability problem is understood and patched
- the earlier comparison story must be revalidated on the wider windows using the corrected baseline

## Revised Next Steps

- Re-run the widened 120-day comparisons using the corrected rescaled baseline.
- Reassess GJR vs plain GARCH only after those corrected wider-window results are available.
- Only after that, decide whether to move into regime logic or additional strategy layers.
