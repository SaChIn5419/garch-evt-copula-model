# Project Log

This file is the running handoff log for the project. It records what changed, where the project currently stands, and the next recommended step. It should be updated after each meaningful milestone.

## Logging Rules

- Record each milestone with date, summary, affected files, and current status.
- Keep the latest project state and next step near the top.
- Update this file before wrapping a work session.
- Use this file as the primary resume point for future work.
- Keep the top-level `summary.md` updated as the short journey view for the project.

## Current State

Date: 2026-04-01

Status:
- Project has migrated substantially from the legacy monolithic pipeline toward the modular `src/` walk-forward engine described in `implementation_plan.md`.
- Current active architecture centers on `main.py` plus the `src/` modules.
- Legacy research/forensics flow still exists in `pipeline.py` and `modules/` and has not yet been fully retired.
- First hardening pass on the new engine is complete and the codebase is now in a better state for the first Git milestone.

Implemented:
- Modular `src/` layout for config, data loading, preprocessing, backtest orchestration, risk metrics, reporting, volatility, EVT, copula, and strategy logic.
- Custom `gjrgarch_fast.py` engine wired into `src/volatility.py`.
- Walk-forward backtest flow in `src/backtest.py`.
- EVT left-tail fitting in `src/evt.py`.
- Student-t copula portfolio simulation in `src/copula_model.py`.
- Basic result export and plots in `src/report.py`.

Validated So Far:
- Static review completed against the current repository state.
- Conceptual alignment confirmed between:
  - the PDF paper `1774372236615.pdf`
  - `implementation_plan.md`
  - the current migration direction toward `GJR-GARCH + EVT + copula`
- Local regression tests pass with `python -m unittest discover -s tests -v`.
- Real cached-data walk-forward comparisons completed for:
  - `us_stress` over a recent 60-day out-of-sample slice
  - `india_primary` over a recent 60-day out-of-sample slice
  - `us_stress` over a widened 120-day out-of-sample slice
  - `india_primary` over a widened 120-day out-of-sample slice

Known Gaps / Risks:
- Current repo still has a split between new canonical code and legacy code paths.
- No live market-data end-to-end run was executed in this session because the environment is network-restricted.
- Regime logic from the paper is still not integrated into the new canonical `src/` stack.
- Some previously logged GARCH-vs-GJR conclusions need to be interpreted carefully because the original `arch` baseline wrapper used `rescale=False` on very small decimal log returns.

Recommended Next Step:
- Use the new direct diagnostics to identify which parts of the custom GJR engine are actually hurting performance relative to corrected plain GARCH.

## Milestones

### 2026-04-01 - Initial project-state review and logging setup

Summary:
- Reviewed the current repository as a code review target.
- Compared the modular engine against the project plan and the reference PDF.
- Confirmed the project is directionally aligned with the paper's methodology.
- Added this log file to preserve project state across sessions.

Files Reviewed:
- `README.md`
- `main.py`
- `pipeline.py`
- `implementation_plan.md`
- `gjrgarch_fast.py`
- `src/config.py`
- `src/data_loader.py`
- `src/preprocessing.py`
- `src/backtest.py`
- `src/volatility.py`
- `src/evt.py`
- `src/copula_model.py`
- `src/strategy.py`
- `src/risk_metrics.py`
- `src/report.py`
- `1774372236615.pdf`

Findings Captured:
- Migration direction is correct: `GJR-GARCH + EVT + copula` is a sensible upgrade from plain `GARCH + EVT + copula`.
- Architecture is stronger than the legacy pipeline, but validation and robustness still lag behind the paper's standard.

Last Left At:
- User requested persistent project logging and milestone-based GitHub pushes.
- `PROJECT_LOG.md` created as the repository handoff log.
- Next milestone should focus on statistical backtesting and robustness fixes.

### 2026-04-01 - Walk-forward engine hardening before first Git push

Summary:
- Fixed the main robustness issues identified during review.
- Added local regression tests so the first version pushed to Git has baseline validation.

Files Changed:
- `src/volatility.py`
- `src/strategy.py`
- `src/risk_metrics.py`
- `src/backtest.py`
- `src/report.py`
- `src/copula_model.py`
- `tests/test_strategy.py`
- `tests/test_volatility.py`
- `tests/test_reporting.py`
- `tests/test_risk_metrics.py`

Changes Made:
- Added a deterministic fallback forecast path when GJR-GARCH fitting fails or returns invalid outputs.
- Stopped silently using broken volatility fits downstream by surfacing fallback usage in the backtest outputs.
- Replaced the previous allocator with a deterministic dollar-neutral rule that respects net, gross, and per-position bounds.
- Added Kupiec and Christoffersen exception backtests plus exception-cluster counting.
- Made `summary.json` strict-JSON safe by converting non-finite metrics to `null`.
- Hardened copula correlation estimation for constant residual panels.
- Added regression tests for allocator constraints, volatility fallback, reporting JSON safety, and risk-metric calculations.

Verification:
- `python -m unittest discover -s tests -v` -> passing

Last Left At:
- Hardened first version is ready to be curated into the initial Git milestone.
- Next step is to stage only the intended project files, commit, and push to GitHub.

### 2026-04-01 - First real GARCH vs GJR validation harness

Summary:
- Added a package-backed baseline volatility model using `arch` so the custom GJR engine can be compared against a plain-GARCH benchmark under the same walk-forward backtest.
- Added a bounded comparison runner that uses cached basket data and writes side-by-side validation artifacts.
- Executed the first real comparison on `us_stress` using a 500-day training window plus the most recent 60 out-of-sample days.

Files Changed:
- `src/volatility.py`
- `compare_walkforward_models.py`
- `tests/test_volatility.py`
- `results_validation/us_stress_2021-10-08_2023-12-29/`

Result Snapshot:
- Basket: `us_stress`
- OOS observations: `60`
- Models compared:
  - `garch_baseline`
  - `gjr_custom`
- Both models produced:
  - `1` VaR breach
  - breach rate `0.0167`
  - Kupiec p-value `0.6357`
  - Christoffersen p-value `0.8527`
  - `1` exception cluster
- Stability difference:
  - `garch_baseline` fallback rate: `0.0194`
  - `gjr_custom` fallback rate: `0.0000`
  - `garch_baseline` convergence rate: `0.9806`
  - `gjr_custom` convergence rate: `1.0000`

Interpretation:
- On this initial bounded slice, custom GJR is not yet better on breach counts than plain GARCH, but it is currently more operationally stable in the new engine.
- This is encouraging, but it is not yet strong statistical evidence because the validation window is still small.

Artifacts:
- `results_validation/us_stress_2021-10-08_2023-12-29/comparison.md`
- `results_validation/us_stress_2021-10-08_2023-12-29/comparison.json`
- per-model backtest outputs saved under the same directory

Last Left At:
- First real cached-data comparison is complete.
- Next step is to widen the comparison window and repeat it for `india_primary` before drawing stronger conclusions about model superiority.

### 2026-04-01 - India primary comparison confirms stronger GJR signal

Summary:
- Ran the same bounded walk-forward comparison on `india_primary` using cached data and the existing comparison harness.
- The `india_primary` result shows a much clearer separation between the plain-GARCH baseline and the custom GJR implementation than `us_stress`.

Files Changed:
- `PROJECT_LOG.md`
- `summary.md`
- `results_validation/india_primary_2021-09-24_2023-12-29/`

Result Snapshot:
- Basket: `india_primary`
- OOS observations: `60`
- `garch_baseline`:
  - breaches: `5`
  - breach rate: `0.0833`
  - exception clusters: `4`
  - Kupiec p-value: `0.00036`
  - Christoffersen p-value: `0.2996`
  - convergence rate: `0.8633`
  - fallback rate: `0.1367`
- `gjr_custom`:
  - breaches: `1`
  - breach rate: `0.0167`
  - exception clusters: `1`
  - Kupiec p-value: `0.6357`
  - Christoffersen p-value: `0.8527`
  - convergence rate: `1.0000`
  - fallback rate: `0.0000`

Interpretation:
- This is the first result in the project where the custom GJR path is not just more stable operationally, but also clearly better on risk-backtest behavior than the plain-GARCH baseline.
- On this bounded `india_primary` slice, plain GARCH materially underestimates risk and fails the Kupiec calibration test, while the custom GJR path remains well-calibrated.

Artifacts:
- `results_validation/india_primary_2021-09-24_2023-12-29/comparison.md`
- `results_validation/india_primary_2021-09-24_2023-12-29/comparison.json`

Last Left At:
- The cross-basket picture is now:
  - `us_stress`: tie on breaches, GJR more stable
  - `india_primary`: GJR clearly better on both stability and calibration
- Next step is to widen the out-of-sample window before making a stronger project-level claim about model superiority.

### 2026-04-01 - Widened 120-day comparison strengthens the GJR case

Summary:
- Widened the walk-forward validation window from 60 out-of-sample days to 120 out-of-sample days for both baskets using the existing comparison harness.
- Fixed a comparison-artifact bug where `comparison.json` could fail on `NaN` metrics.

Files Changed:
- `compare_walkforward_models.py`
- `PROJECT_LOG.md`
- `summary.md`
- `results_validation/us_stress_2021-07-15_2023-12-29/`
- `results_validation/india_primary_2021-06-28_2023-12-29/`

Result Snapshot:
- `us_stress`, 120 OOS days:
  - `garch_baseline`: 2 breaches, breach rate `0.0167`, fallback rate `0.0181`
  - `gjr_custom`: 0 breaches, breach rate `0.0000`, fallback rate `0.0000`
- `india_primary`, 120 OOS days:
  - `garch_baseline`: 11 breaches, breach rate `0.0917`, Kupiec p-value `4.38e-08`, fallback rate `0.1217`
  - `gjr_custom`: 3 breaches, breach rate `0.0250`, Kupiec p-value `0.1653`, fallback rate `0.0000`

Interpretation:
- The wider window strengthens the project-level case for the custom GJR path.
- `us_stress`: GJR remains more stable and is now also better on breaches.
- `india_primary`: GJR remains materially better on both calibration and operational stability.
- The package-backed plain-GARCH baseline is still useful as a benchmark, but it is currently not a credible production fallback on these baskets.

Artifacts:
- `results_validation/us_stress_2021-07-15_2023-12-29/comparison.md`
- `results_validation/india_primary_2021-06-28_2023-12-29/comparison.md`

Last Left At:
- The validation story is now stronger and cross-basket:
  - `us_stress` 120-day: GJR better
  - `india_primary` 120-day: GJR clearly better
- Next step is to investigate why the `arch` plain-GARCH baseline is so unstable on these baskets before deciding whether to keep it as anything more than a benchmark.

### 2026-04-01 - Arch baseline instability traced to scaling choice

Summary:
- Investigated why the package-backed plain-GARCH baseline was warning-heavy and fallback-prone.
- Found that the dominant issue was not plain GARCH itself, but the wrapper fitting Student-t `arch` models on small decimal log returns with `rescale=False`.
- Updated `ArchVolatilityModel` to use `rescale=True` by default.

Files Changed:
- `src/volatility.py`
- `PROJECT_LOG.md`
- `summary.md`
- `results_validation_rescaled/us_stress_2021-10-08_2023-12-29/`
- `results_validation_rescaled/india_primary_2021-09-24_2023-12-29/`

Investigation Findings:
- Previously failed windows recovered when either:
  - `arch` was allowed to internally rescale the data, or
  - the same data was manually scaled by `x100`
- Recovery check on previously failed windows:
  - `us_stress`: `13/13` failed windows recovered with rescaled Student-t
  - `india_primary`: `73/73` failed windows recovered with rescaled Student-t
- The known problematic windows for:
  - `JPM` on `2023-11-09`
  - `^NSEI` on `2023-09-07`
  both fit cleanly after the wrapper change.

Corrected Interpretation:
- The earlier fallback-heavy baseline comparisons overstated the advantage of the custom GJR path because the baseline wrapper itself was handicapped by poor scaling.
- On re-run 60-day slices with the rescaled baseline:
  - `us_stress`: plain GARCH had `0` breaches vs `1` for GJR
  - `india_primary`: plain GARCH had `0` breaches vs `1` for GJR
- That means the question is no longer “why is the baseline broken?” but “how does corrected plain GARCH compare to custom GJR on the wider 120-day windows?”

Last Left At:
- The baseline instability root cause is identified and patched.
- Next step is to rerun the wider comparisons with the corrected baseline before making any further model-selection claims.

### 2026-04-01 - Corrected 120-day rerun reverses the earlier model-selection claim

Summary:
- Re-ran the widened 120-day comparisons for both baskets after fixing the `arch` baseline scaling issue.
- The corrected results materially changed the project-level interpretation.

Files Changed:
- `PROJECT_LOG.md`
- `summary.md`
- `results_validation_rescaled/us_stress_2021-07-15_2023-12-29/`
- `results_validation_rescaled/india_primary_2021-06-28_2023-12-29/`

Corrected 120-Day Results:
- `us_stress`:
  - `garch_baseline`: `0` breaches, fallback rate `0.0000`
  - `gjr_custom`: `0` breaches, fallback rate `0.0000`
- `india_primary`:
  - `garch_baseline`: `0` breaches, fallback rate `0.0000`
  - `gjr_custom`: `3` breaches, breach rate `0.0250`, fallback rate `0.0000`

Interpretation:
- After correcting the baseline scaling issue, the earlier evidence favoring custom GJR does not hold on these widened validation windows.
- On `us_stress`, the models are tied on breaches and both are operationally stable.
- On `india_primary`, corrected plain GARCH is now strictly better on breaches than the custom GJR path.
- The honest current conclusion is that the custom GJR engine is not yet validated as superior to a correctly specified plain-GARCH benchmark.

Artifacts:
- `results_validation_rescaled/us_stress_2021-07-15_2023-12-29/comparison.md`
- `results_validation_rescaled/india_primary_2021-06-28_2023-12-29/comparison.md`

Last Left At:
- The project now has a corrected benchmark, and that benchmark currently performs at least as well as, and on `india_primary` better than, the custom GJR path on the tested windows.
- Next step is to diagnose why the custom GJR engine is underperforming relative to corrected plain GARCH before adding any new modeling layer.

### 2026-04-01 - Direct India diagnostics added for model understanding

Summary:
- Added a dedicated diagnostic comparison for `india_primary` so plain GARCH and custom GJR can be inspected visually on the same windows.
- Fixed another baseline-wrapper issue: when `arch` internally rescales data, the wrapper now maps forecasts and conditional variances back to original return units.

Files Changed:
- `src/volatility.py`
- `compare_india_diagnostics.py`
- `PROJECT_LOG.md`
- `summary.md`
- `results_diagnostics/india_primary_diagnostics_2021-06-28_2023-12-29/`

Diagnostic Artifacts:
- `volatility_forecast_panel.png`
- `persistence_panel.png`
- `residual_variance_panel.png`
- `residual_sq_acf_panel.png`
- `diagnostic_summary.md`

Diagnostic Takeaways:
- Forecast volatility levels are now directly comparable in original units and are fairly close between models.
- Persistence differences are asset-specific rather than uniformly in favor of GJR.
- Standardized residual variance is near `1` for both models across assets.
- Squared standardized residual lag-1 autocorrelation is mixed by asset; GJR is not showing a clear blanket improvement.
- Both models converged cleanly with zero fallbacks in the corrected diagnostic run.

Interpretation:
- The new visual and diagnostic layer is now sufficient for model comparison.
- At this point, custom GJR does not have a clear diagnostic edge over corrected plain GARCH on `india_primary`.
- The remaining work should focus on identifying whether the custom GJR implementation has a specification issue, a forecasting issue, or simply no empirical advantage on these baskets.

Artifacts:
- `results_diagnostics/india_primary_diagnostics_2021-06-28_2023-12-29/diagnostic_summary.md`
- `results_diagnostics/india_primary_diagnostics_2021-06-28_2023-12-29/volatility_forecast_panel.png`
- `results_diagnostics/india_primary_diagnostics_2021-06-28_2023-12-29/persistence_panel.png`
- `results_diagnostics/india_primary_diagnostics_2021-06-28_2023-12-29/residual_variance_panel.png`
- `results_diagnostics/india_primary_diagnostics_2021-06-28_2023-12-29/residual_sq_acf_panel.png`

Last Left At:
- The project now has backtest comparisons plus direct model-diagnostic visuals.
- Next step is to decide whether to keep investing in the custom GJR engine or revert the default marginal model back to corrected plain GARCH.

### 2026-04-02 - GJR engine-level diagnostic isolated the current issue

Summary:
- Added a dedicated engine comparison for `india_primary` that measures `custom_gjr` directly against package-backed `arch` GJR on the same rolling windows.
- Confirmed that the asymmetry specification still has signal on this basket, but the current custom implementation is not reproducing the stronger `arch` GJR walk-forward behavior.

Files Changed:
- `compare_gjr_engines.py`
- `PROJECT_LOG.md`
- `summary.md`
- `results_diagnostics/india_primary_gjr_engine_compare_2021-06-28_2023-12-29/`

Diagnostic Artifacts:
- `engine_summary.md`
- `engine_diagnostics.csv`
- `vol_forecast_gap_panel.png`
- `persistence_gap_panel.png`
- `residual_sq_acf_gap_panel.png`

Diagnostic Takeaways:
- `custom_gjr` and `arch` GJR are operationally stable on the same India windows: both converged cleanly with zero fallbacks.
- The two engines are close on some assets, but they are not equivalent:
  - `^NSEI`: custom volatility forecast averages about `5.4%` above `arch` GJR.
  - `^CNXPHARMA`: custom volatility forecast averages about `2.4%` above `arch` GJR.
  - `^CNXENERGY`: custom persistence averages about `0.0915` above `arch` GJR and leaves materially higher squared-residual autocorrelation.
- A direct 120-day control backtest on the same `india_primary` slice gave:
  - `garch_baseline`: `3` breaches
  - `arch_gjr`: `1` breach
  - `custom_gjr`: `3` breaches

Interpretation:
- The earlier question is now sharper: this is not just “GJR may have no empirical edge.”
- On this India segment, package-backed GJR does outperform corrected plain GARCH, while the current custom GJR does not.
- That points to an implementation-quality gap in the custom engine or wrapper, not just a model-family tie.

Artifacts:
- `results_diagnostics/india_primary_gjr_engine_compare_2021-06-28_2023-12-29/engine_summary.md`
- `results_diagnostics/india_primary_gjr_engine_compare_2021-06-28_2023-12-29/vol_forecast_gap_panel.png`
- `results_diagnostics/india_primary_gjr_engine_compare_2021-06-28_2023-12-29/persistence_gap_panel.png`
- `results_diagnostics/india_primary_gjr_engine_compare_2021-06-28_2023-12-29/residual_sq_acf_gap_panel.png`

Last Left At:
- The next step should focus on reconciling the custom recursion and forecast path against `arch` GJR, starting with the assets showing the largest stable divergence: `^NSEI`, `^CNXPHARMA`, and `^CNXENERGY`.

### 2026-04-02 - First remediation pass closed the India control gap

Summary:
- Updated the custom GJR wrapper to fit on internally rescaled returns and map forecasts back to original units.
- Disabled custom warm starts by default so rolling fits no longer terminate after only a few local optimizer steps.

Files Changed:
- `src/volatility.py`
- `PROJECT_LOG.md`

Validation:
- Re-ran the 120-day `india_primary` control comparison after the wrapper change.
- Results:
  - `garch_baseline`: `3` breaches
  - `arch_gjr`: `1` breach
  - `custom_gjr`: `1` breach
- Mean optimizer iterations for `custom_gjr` rose to about `37.04`, versus the previously suspicious near-`3` warm-start regime.

Interpretation:
- The audit findings were actionable and correct.
- Numerical conditioning plus the warm-start deployment path were the main reasons the custom engine was underperforming on the India control slice.
- After the first remediation pass, the custom wrapper now tracks the package-backed GJR path much more closely at the backtest level.

Last Left At:
- The next step should be a narrower engine-core reconciliation against `arch` GJR, especially around the custom recursion initialization and any remaining parameter-level differences.

### 2026-04-02 - Core initialization updated after wrapper stabilization

Summary:
- Replaced the custom recursion's fixed sample-variance start with a parameter-aware initial variance that blends unconditional variance with a residual backcast.
- Applied the same initialization helper to the simulation path so the engine uses a consistent variance start rule.

Files Changed:
- `gjrgarch_fast.py`
- `PROJECT_LOG.md`

Validation:
- Re-ran the 120-day `india_primary` control comparison after the core change.
- Results stayed aligned:
  - `arch_gjr`: `1` breach
  - `custom_gjr`: `1` breach
- `custom_gjr` remained fully operational with zero fallbacks and mean optimizer iterations around `35.37`.

Interpretation:
- The wrapper fix was the main driver of the earlier correction, and the core initialization change did not break the recovered behavior.
- The custom engine now has a cleaner likelihood setup than the old sample-proxy start while preserving the improved backtest outcome.

Last Left At:
- The next step should focus on tighter parameter-level reconciliation against `arch` GJR and wider revalidation beyond the audited India control slice.

### 2026-04-02 - Wider revalidation confirms custom GJR recovery

Summary:
- Updated the walk-forward comparison workflow so package-backed GJR is now a default control model alongside plain GARCH and custom GJR.
- Re-ran wider 252-day walk-forward re-audits on both `india_primary` and `us_stress`.
- Added a direct `us_stress` engine comparison between `arch` GJR and custom GJR.

Files Changed:
- `compare_walkforward_models.py`
- `compare_gjr_engines.py`
- `PROJECT_LOG.md`

Validation:
- `india_primary` re-audit (`252` days):
  - `garch_baseline`: `4` breaches
  - `gjr_arch`: `2` breaches
  - `gjr_custom`: `2` breaches
- `us_stress` re-audit (`252` days):
  - all three models recorded `5` breaches
- `us_stress` engine comparison showed zero fallbacks and full convergence for both GJR implementations, with generally small remaining gaps; the largest stable residual mismatch is on `XOM`.

Interpretation:
- The wrapper and initialization fixes generalized beyond the narrow India control slice.
- The custom GJR path now matches package-backed GJR on the widened basket-level backtests that matter for model selection.
- Current evidence supports keeping the custom engine in contention rather than reverting to plain GARCH by default.
- Remaining work is now parameter-level and asset-specific, not a broad deployment failure.

Last Left At:
- The next step should focus on the remaining asset-level reconciliation against `arch` GJR, with `XOM` the clearest residual gap on `us_stress`.

### 2026-04-02 - Residual XOM gap traced to negative-gamma windows

Summary:
- Investigated the remaining `XOM` mismatch between package-backed GJR and custom GJR on `us_stress`.
- Added negative-gamma tracking to the engine comparison workflow so this behavior is explicit in future diagnostics.

Files Changed:
- `compare_gjr_engines.py`
- `PROJECT_LOG.md`

Diagnostic Takeaways:
- About `15%` of the audited `XOM` windows had negative `gamma` in the package-backed `arch` GJR fit.
- The custom engine enforces `gamma >= 0`, so it cannot reproduce those windows exactly.
- In the negative-gamma subset, the residual gap is much larger:
  - mean volatility-gap about `0.0020`
  - mean persistence-gap about `-0.0908`
- Outside those windows, the residual mismatch is much smaller and close to the current “acceptable implementation difference” range.

Interpretation:
- The remaining `XOM` gap is now best understood as a parameter-space / model-specification difference rather than another general execution bug.
- The current custom engine is still behaving credibly at the basket level, but exact match to `arch` GJR is limited when `arch` chooses negative asymmetry.

Last Left At:
- The next decision is conceptual rather than purely mechanical:
  - either keep the current nonnegative-`gamma` custom specification and document the difference
  - or relax the custom asymmetry constraint if exact `arch`-style matching is the goal
