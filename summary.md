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
