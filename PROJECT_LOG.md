# Project Log

This file is the running handoff log for the project. It records what changed, where the project currently stands, and the next recommended step. It should be updated after each meaningful milestone.

## Logging Rules

- Record each milestone with date, summary, affected files, and current status.
- Keep the latest project state and next step near the top.
- Update this file before wrapping a work session.
- Use this file as the primary resume point for future work.

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

Known Gaps / Risks:
- Current repo still has a split between new canonical code and legacy code paths.
- No live market-data end-to-end run was executed in this session because the environment is network-restricted.
- Regime logic from the paper is still not integrated into the new canonical `src/` stack.

Recommended Next Step:
- Run the first real basket backtest through the hardened `src/` path and compare:
  - old plain-GARCH workflow vs current GJR workflow
  - breach rate
  - Kupiec p-value
  - Christoffersen p-value
  - exception clustering

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
