# Milestone Status Memo

Date: 2026-04-02

## Current Status

The project now has a working research-grade end-to-end risk engine.

What is currently working:
- rolling walk-forward marginal volatility fitting
- EVT-based left-tail adjustment
- copula-based portfolio risk simulation
- portfolio VaR/CVaR forecasting and backtesting
- basket-level comparison workflows
- direct diagnostic tooling for plain GARCH, package-backed GJR, and custom GJR

Most importantly, the custom GJR engine is no longer the broken component it was during the earlier diagnostic phase.

Current validated position:
- `india_primary` on the widened `252`-day re-audit:
  - plain GARCH: `4` breaches
  - `gjr_arch`: `2` breaches
  - `gjr_custom`: `2` breaches
- `us_stress` on the widened `252`-day re-audit:
  - plain GARCH: `5` breaches
  - `gjr_arch`: `5` breaches
  - `gjr_custom`: `5` breaches

Interpretation:
- the custom engine now matches package-backed GJR on the basket-level backtests that matter
- India still shows a real GJR advantage over plain GARCH
- `us_stress` currently looks like a tie across the tested models on the widened window

## Completed Components

Implemented and working:
- custom GJR engine core in `gjrgarch_fast.py`
- model wrappers in `src/volatility.py`
- walk-forward backtesting in `src/backtest.py`
- EVT fitting in `src/evt.py`
- portfolio risk simulation in `src/copula_model.py`
- reporting and artifact generation in `src/report.py`
- comparison tooling in `compare_walkforward_models.py`
- direct model-diagnostic tooling in `compare_india_diagnostics.py`
- direct engine-control tooling in `compare_gjr_engines.py`

Recent high-value fixes:
- internal rescaling added to the custom GJR wrapper
- warm starts disabled by default for the custom GJR path
- recursion initialization upgraded from a fixed sample proxy to a parameter-aware initial variance
- package-backed GJR added as a standard control in validation comparisons

## What We Understand So Far

The project has crossed an important diagnostic threshold.

What we now know:
- the earlier custom GJR underperformance was mainly caused by numerical conditioning and deployment logic, not just by model-family weakness
- after fixing those issues, custom GJR recovered and matched `arch` GJR on the widened basket-level backtests
- `india_primary` appears to benefit from the GJR family relative to plain GARCH
- `us_stress` does not currently show a basket-level advantage from GJR on the audited widened window
- remaining differences are now narrower and asset-specific rather than broad system failures

The clearest remaining residual mismatch is on `XOM` in `us_stress`, where custom and package-backed GJR still differ at the engine-diagnostic level even though the basket summary is tied.

## Remaining Blockers

The model is not yet “finished production quality.”

Still remaining:
- wider robustness validation across more baskets and time windows
- tighter parameter-level reconciliation against `arch` GJR on the remaining outlier assets
- stronger regression-style tests so the current fixes cannot quietly regress
- a clearer default model-selection policy for when plain GARCH vs GJR should be preferred
- additional reporting and packaging polish if the goal is a formal final deliverable

## How Far Along We Are

If the target is:

Research-grade working system:
- approximately `80-85%` complete

Production-trustworthy validated system:
- approximately `60-70%` complete

Reason:
- the end-to-end engine works
- the major custom-engine failure has been fixed
- the remaining work is narrower validation, policy, and hardening rather than foundational implementation

## Improvement Achieved

The improvement is material, not cosmetic.

Before the audit and fixes:
- custom GJR could underperform the correct GJR control
- that made the model-selection story misleading

After the audit and fixes:
- on the India control slice, custom GJR improved from `3` breaches to `1`
- on the widened India re-audit, custom GJR matched `arch` GJR at `2` breaches while plain GARCH had `4`

That means the custom engine was moved from “not trustworthy” to “credible and competitive with the control.”

## Recommended Next 3 Milestones

### 1. Asset-Level Reconciliation

Goal:
- explain the remaining parameter-level differences between `gjr_custom` and `gjr_arch`

Primary target:
- `XOM` on `us_stress`

Success condition:
- residual forecast and persistence gaps are either reduced further or explained as harmless implementation differences

### 2. Wider Robustness Revalidation

Goal:
- verify that the recovered custom engine continues to track `arch` GJR beyond the currently audited slices

Scope:
- more windows
- both baskets
- possibly additional stress segments

Success condition:
- the current “custom matches control” result holds consistently enough to trust model choice

### 3. Default Model Policy And Hardening

Goal:
- decide what should be the default marginal model in the pipeline and lock in regression checks

Success condition:
- explicit default model policy
- reproducible comparison tests
- no silent return to the earlier custom-engine failure mode

## Bottom Line

The model is now working in the sense that the integrated research engine is operational and the custom GJR path has been rehabilitated.

The remaining work is real, but it is no longer foundational rescue work. It is targeted reconciliation, broader validation, and hardening.
