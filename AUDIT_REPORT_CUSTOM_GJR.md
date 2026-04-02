# Custom GJR Audit Report

Date: 2026-04-02

Scope:
- Custom engine core in `gjrgarch_fast.py`
- Wrapper and deployment path in `src/volatility.py`
- Diagnostic and validation framing in `compare_india_diagnostics.py` and `compare_walkforward_models.py`

Audit objective:
- Determine whether the current `custom_gjr` underperformance is caused by model-family weakness, implementation bias, numerical conditioning, execution choices, or analysis framing.

## Executive Verdict

The current custom GJR path has a confirmed implementation and deployment problem.

The strongest control result is:
- corrected plain GARCH: `3` breaches
- package-backed `arch` GJR: `1` breach
- deployed `custom_gjr`: `3` breaches

That means:
- the GJR family still has signal on the audited `india_primary` slice
- the current custom implementation is not reproducing that signal reliably
- the main issues are numerical conditioning and deployment choices, not just lack of empirical advantage

## Findings

### 1. High: the custom engine is fit on poorly conditioned raw returns while the benchmark is explicitly rescaled

Where:
- `gjrgarch_fast.py`
- `src/volatility.py`
- `src/config.py`

Evidence:
- The default model distribution is Student-t on daily log returns around `1e-2`.
- The `arch` wrapper enables `rescale=True` in [src/volatility.py](/home/sachindb/Documents/garch_evt_copla_project_v2/src/volatility.py#L169), which materially improves optimizer conditioning.
- The custom wrapper in [src/volatility.py](/home/sachindb/Documents/garch_evt_copla_project_v2/src/volatility.py#L90) fits directly on raw returns with no equivalent rescaling step.
- A control experiment fitting the same custom engine on returns scaled by `100` and mapping forecasts back to original units made it much closer to `arch` GJR on the worst assets:
  - `^CNXENERGY`: mean absolute vol-gap vs `arch` fell from `0.000304` to `0.000039`
  - `^CNXPHARMA`: fell from `0.000594` to `0.000016`
  - `^NSEI`: fell from `0.000283` to `0.000057`
- The persistence gaps also compressed materially under scaling:
  - `^CNXENERGY`: `+0.057467` to `-0.008876`
  - `^CNXPHARMA`: `+0.007975` to `-0.001994`
  - `^NSEI`: `-0.003411` to `-0.000522`

Assessment:
- This is the most important confirmed source of bias.
- The custom engine is being judged against a numerically advantaged baseline.
- The raw-scale optimization path is not robust enough for the current data scale.

Recommendation:
- Add an internal rescaling step to `GJRGARCHVolatilityModel.fit_one` so optimization runs on stable magnitudes and all outputs are mapped back to original units.
- Keep the scale factor in the forecast metadata for traceability.

### 2. High: the rolling warm-start logic is prematurely freezing the optimizer and can move forecasts farther from the control engine

Where:
- `src/volatility.py`

Evidence:
- The deployed path caches optimizer states by asset in [src/volatility.py](/home/sachindb/Documents/garch_evt_copla_project_v2/src/volatility.py#L46) and reuses them in [src/volatility.py](/home/sachindb/Documents/garch_evt_copla_project_v2/src/volatility.py#L96).
- On the audited India slice, warm-started rolling fits averaged only about `3` optimizer iterations on the worst assets:
  - `^CNXENERGY`: `3.408`
  - `^CNXPHARMA`: `3.733`
  - `^NSEI`: `3.217`
- Fresh fits of the same raw custom engine on the same windows took far more work:
  - `^CNXENERGY`: `33.042`
  - `^CNXPHARMA`: `25.483`
  - `^NSEI`: `58.325`
- Warm starts also worsened alignment to `arch` GJR on key assets:
  - `^CNXENERGY` mean persistence gap: `+0.091533` warm vs `+0.057467` fresh
  - `^NSEI` mean absolute vol-gap: `0.000384` warm vs `0.000283` fresh

Assessment:
- The current warm-start scheme is not just speeding up optimization.
- In the raw-scale regime it is often causing extremely fast local termination and worse parameter tracking.
- This is a deployment-path issue even if the core engine were otherwise acceptable.

Recommendation:
- Disable warm starts until scaling is fixed and revalidated.
- If warm starts are reintroduced later, gate them with sanity checks:
  - minimum optimizer iterations
  - objective improvement threshold
  - re-run from default start if the solution stops too quickly or deviates sharply from recent fit quality

### 3. Medium: the project’s prior diagnostic framing created a false model-selection conclusion

Where:
- `compare_india_diagnostics.py`
- `compare_walkforward_models.py`

Evidence:
- The prior diagnostic path only compared:
  - `garch_baseline`
  - `gjr_custom`
- It did not include a package-backed `arch` GJR control in [compare_india_diagnostics.py](/home/sachindb/Documents/garch_evt_copla_project_v2/compare_india_diagnostics.py#L108) or [compare_walkforward_models.py](/home/sachindb/Documents/garch_evt_copla_project_v2/compare_walkforward_models.py#L89).
- That framing supported the interim conclusion that custom GJR had no clear edge over corrected plain GARCH.
- A direct control backtest on the same 120-day `india_primary` slice showed:
  - `garch_baseline`: `3` breaches
  - `arch_gjr`: `1` breach
  - `custom_gjr`: `3` breaches

Assessment:
- The earlier conclusion “GJR may have no edge here” was too broad.
- The correct conclusion is narrower:
  - package-backed GJR can outperform plain GARCH here
  - the current custom implementation does not

Recommendation:
- Treat `arch` GJR as the standing control whenever the custom engine is evaluated.
- Do not use `custom_gjr` vs plain GARCH alone to make model-family decisions.

### 4. Medium: the custom recursion fixes the first conditional variance to a sample proxy rather than a parameter-consistent backcast

Where:
- `gjrgarch_fast.py`

Evidence:
- In [gjrgarch_fast.py](/home/sachindb/Documents/garch_evt_copla_project_v2/gjrgarch_fast.py#L139), the recursion initializes `sig2[0]` from a sample variance proxy and keeps it independent of the fitted parameter vector.
- The likelihood therefore optimizes with a hard-wired first state rather than a parameter-consistent backcast or unconditional variance.
- This does not appear to be the dominant source of the observed India mismatch once scaling is corrected, but it does change the objective surface and can bias short-window estimation and parameter comparability against `arch`.

Assessment:
- This is a structural modeling weakness rather than the primary confirmed failure driver.
- It likely contributes residual differences even after conditioning improves.

Recommendation:
- Replace the fixed sample-variance start with a parameter-consistent backcast or unconditional variance initialization.
- Re-run the `arch` GJR comparison after that change.

## Additional Observations

- The custom engine remained operationally stable on the audited windows: no fallbacks and clean convergence flags in the existing diagnostic runs.
- Stability alone is not sufficient here; the engine is stable but still miscalibrated relative to the control path.
- During broader alternative-wrapper testing, SciPy emitted finite-difference warnings on some custom fits. That reinforces the conditioning concern but was not needed to establish the main findings.

## Audit Methods

Methods used in this audit:
- Read the custom engine core and wrapper code.
- Compared deployed `custom_gjr` against `arch` GJR on the same `india_primary` rolling windows.
- Ran focused experiments on the three assets with the largest persistent divergence:
  - `^NSEI`
  - `^CNXPHARMA`
  - `^CNXENERGY`
- Re-fit the same custom engine on scaled returns to isolate numerical conditioning effects.
- Compared rolling warm-started fits against fresh fits to isolate deployment-path bias.

Relevant artifacts:
- [compare_gjr_engines.py](/home/sachindb/Documents/garch_evt_copla_project_v2/compare_gjr_engines.py)
- [engine_summary.md](/home/sachindb/Documents/garch_evt_copla_project_v2/results_diagnostics/india_primary_gjr_engine_compare_2021-06-28_2023-12-29/engine_summary.md)
- [PROJECT_LOG.md](/home/sachindb/Documents/garch_evt_copla_project_v2/PROJECT_LOG.md#L364)

## Recommended Remediation Order

1. Add internal return rescaling to the custom wrapper and map outputs back to original units.
2. Disable warm starts temporarily and re-benchmark the 120-day `india_primary` slice.
3. Add `arch` GJR as a standard control in every diagnostic and validation comparison.
4. Replace the fixed initial conditional variance with a parameter-consistent backcast.
5. Only after those changes, reassess whether the custom engine is suitable as the default marginal model.

## Bottom Line

The audited evidence does not support retiring the GJR family on `india_primary`.

It supports a narrower conclusion:
- the current deployed custom GJR path is biased by poor numerical conditioning and an over-aggressive warm-start execution path
- the analysis layer previously overstated the case against GJR because it lacked an `arch` GJR control
- the custom engine should not be treated as production-valid until it is re-run with internal scaling, stricter warm-start controls, and a parameter-consistent initialization
