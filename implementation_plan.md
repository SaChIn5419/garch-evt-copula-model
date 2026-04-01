# Implementation Plan: Walk-Forward Risk Engine Upgrade

## 1. Objective

This project will be upgraded from a monolithic, one-shot research pipeline into a modular walk-forward risk engine that can:

1. forecast risk at the individual asset level,
2. forecast and backtest risk at the portfolio level,
3. support dollar-neutral portfolio construction,
4. provide clear model diagnostics and explainable outputs,
5. evolve from a research-grade stack into a production-capable engine.

The immediate goal is not to build every advanced feature at once. The first production milestone is a credible walk-forward VaR/CVaR engine with strict anti-leakage controls, reproducible backtests, and portfolio constraints that reflect a realistic long/short workflow.

## 2. Guiding Principles

- No look-ahead bias. Any value used at time `t` must be estimated only from data available up to `t-1`.
- Dual modeling scope. The system must support both per-asset forecasts and portfolio-level forecasts.
- Modular interfaces. Each model layer must expose clean inputs/outputs so components can be swapped without rewriting the backtest harness.
- Research explainability first. Custom implementations are preferred where they materially improve transparency, control, or future commercial value.
- Production discipline. Every phase must end with deterministic outputs, validation artifacts, and a clear definition of done.

## 3. Target System Behavior

### 3.1 Forecasting Scope

The engine will operate at two levels:

- Asset level:
  Forecast conditional volatility, tail risk, and risk metrics for each asset independently.
- Portfolio level:
  Construct a dollar-neutral portfolio from the modeled assets, estimate next-period portfolio risk, and backtest realized outcomes.

### 3.2 Portfolio Construction Scope

The portfolio layer will be designed for long/short risk allocation rather than equal-weight allocation.

Initial portfolio constraints for implementation:

- Net exposure must equal `0`.
- Gross exposure must equal `1.0` by default.
- Position bounds must be configurable per asset, with an initial default of `[-0.30, 0.30]`.
- Turnover controls are optional in the first implementation and can be added later.

If optimization is unavailable or unstable for a given date, the system must fall back to a deterministic feasible portfolio rule rather than silently failing.

### 3.3 Data Universe Scope

The project currently targets two baskets:

- `india_primary`: NIFTY 50, Bank, IT, Energy, Pharma
- `us_stress`: DAL, XOM, JPM, BA, AAL, CCL

These baskets should be treated as separate backtest universes in the first implementation. We will not combine India and US assets into a single tradable portfolio until calendar alignment, market hours, and missing-data policy are explicitly designed and tested.

## 4. Execution Strategy

The implementation will proceed in phase-gated order. Later phases are not allowed to block delivery of the first credible risk engine.

### Phase Gate Logic

- Phase 1 must deliver a reliable walk-forward backtest skeleton and baseline risk metrics.
- Phase 2 must deliver volatility modeling behind a stable interface.
- Phase 3 must deliver EVT-based tail risk on top of Phase 2 outputs.
- Phase 4 and beyond extend the engine with dependence, regime, and sizing logic.

This means the custom GJR-GARCH engine is important, but it is not allowed to prevent the rest of the risk framework from being built and validated.

## 5. Repository Architecture

The repository will be migrated to a `src/`-based layout.

### Core Modules

- `src/config.py`
  Global hyperparameters, ticker baskets, rolling windows, optimizer limits, and portfolio constraints.
- `src/data_loader.py`
  Downloading, validation, caching, and market-specific data alignment.
- `src/preprocessing.py`
  Log returns, realized volatility inputs, drawdown features, standardized residual helpers, and anti-leakage transforms.
- `src/backtest.py`
  Walk-forward engine that coordinates rolling windows, refits, forecasts, realized outcomes, and metric collection.
- `src/risk_metrics.py`
  VaR, CVaR, breach counting, Kupiec, Christoffersen, drawdown, and summary statistics.
- `src/report.py`
  Diagnostic plots, backtest summaries, and structured tables for review.

### Model Modules

- `src/volatility.py`
  Volatility model interface and implementations.
  This will include:
  - a baseline package-backed model for validation and fallback,
  - a custom GJR-GARCH implementation under active development,
  - deterministic fitting and optimizer diagnostics.
- `src/evt.py`
  Peaks-over-threshold fitting, GPD parameter estimation, threshold diagnostics, and tail-based VaR/CVaR forecasts.
- `src/copula_model.py`
  Pseudo-observations, dependence estimation, copula fitting, and portfolio simulation logic.
- `src/regime.py`
  Regime-state feature generation and causal filtering logic.
- `src/strategy.py`
  Dollar-neutral optimization, portfolio sizing rules, and fail-safe fallback allocation.

### Entrypoints

- `main.py`
  Main execution entrypoint for configured backtests.

### Legacy Components

Legacy files will not be deleted immediately. They should remain available until the new stack reproduces or clearly supersedes their relevant functionality.

- `pipeline.py`
- `compare_var_models.py`
- `modules/quant_engine.py`
- `modules/forensics.py`

Deletion should happen only after replacement behavior is implemented and reviewed.

## 6. Backtest Contract

This section defines the minimum protocol required before implementation starts.

### 6.1 Frequency

- Data frequency: daily
- Forecast horizon: 1 day ahead
- Rebalancing frequency: daily in the first implementation
- Metric evaluation frequency: daily

### 6.2 Training and Refit Rules

Initial defaults:

- Minimum training window: `500` trading days
- Volatility model refit frequency: every day in the first baseline implementation
- Copula refit frequency: every `20` trading days in the first dynamic dependence implementation
- Regime model refit frequency: every `20` trading days unless stability issues require a longer cadence

These values must be configurable through `src/config.py`.

### 6.3 Walk-Forward Sequence

For each backtest date `t`:

1. collect the historical window ending at `t-1`,
2. fit or update required models using only that historical data,
3. generate one-step-ahead forecasts for each asset,
4. construct portfolio weights subject to dollar-neutral constraints,
5. compute forecast portfolio risk,
6. observe realized asset and portfolio returns at `t`,
7. record breaches, forecast errors, and diagnostic metadata.

### 6.4 Output Objects

The backtest engine should produce:

- asset-level forecast table,
- portfolio weight history,
- portfolio-level forecast table,
- realized return series,
- breach flags,
- model diagnostics,
- summary statistics.

## 7. Phase Plan

### Phase 1: Data Layer, Features, and Backtest Skeleton

Goal:
Build the anti-leakage foundation and baseline walk-forward engine.

Deliverables:

- stable `data_loader.py` with cache support and explicit validation,
- `preprocessing.py` with log returns and leakage-safe transforms,
- `backtest.py` with rolling walk-forward execution,
- baseline historical or parametric VaR/CVaR calculation,
- asset-level and portfolio-level breach counting,
- saved backtest result tables for later analysis.

Required implementation details:

- Separate handling for India and US baskets.
- Explicit missing-data policy.
- No forward-fill across economically invalid gaps without documentation.
- Portfolio engine must support dollar-neutral constraints from day one.

Definition of done:

- The engine runs end-to-end for at least one basket.
- Forecasts are produced for each asset and the portfolio.
- Breach statistics and summary outputs are reproducible.
- A targeted leakage test passes.

### Phase 2: Volatility Model Interface and Baseline GJR/GARCH Work

Goal:
Replace naive risk estimation with a modular conditional volatility layer.

Deliverables:

- `volatility.py` model interface,
- package-backed baseline GARCH/GJR-GARCH implementation,
- standardized residual output,
- conditional variance forecasts,
- model-fit diagnostics and fallback signaling.

Custom engine track:

- Implement a custom GJR-GARCH likelihood function.
- Support parameters `omega`, `alpha`, `gamma`, `beta`.
- Enforce positivity and stationarity-related constraints explicitly.
- Add deterministic initialization, warm starts, and bounded retry logic.
- Persist fit diagnostics for each date and asset.

Definition of done:

- Baseline volatility model is integrated into walk-forward backtests.
- Standardized residuals are available for EVT fitting.
- Custom engine can be benchmarked against the baseline on controlled samples, even if it is not yet the default production path.

### Phase 3: EVT Tail Modeling

Goal:
Model tail risk using standardized residuals rather than Gaussian assumptions.

Deliverables:

- `evt.py` with POT workflow,
- threshold selection utilities,
- GPD parameter estimation,
- rolling tail-based VaR and CVaR forecasts,
- threshold stability and mean excess diagnostics.

Implementation notes:

- Threshold selection should begin with configurable quantiles, not fully automated heuristics.
- If EVT fitting is unstable for a window, the engine must emit a visible fallback path and diagnostic record.

Definition of done:

- EVT-based asset-level risk forecasts are produced in walk-forward mode.
- Portfolio-level tail risk can be computed from constituent forecasts or simulated aggregation logic.
- The report layer can render threshold diagnostics.

### Phase 4: Dependence Modeling and Portfolio Simulation

Goal:
Move from marginal risk estimation to portfolio-aware joint loss modeling.

Deliverables:

- `copula_model.py` with pseudo-observation mapping,
- baseline Gaussian and Student-t copula support,
- rolling dependence estimation,
- tail-dependence diagnostics,
- portfolio simulation from fitted marginals plus dependence structure.

Scope control:

- Gaussian and Student-t copulas are phase-critical.
- Clayton and Gumbel can be added after the first working dynamic dependence version.

Definition of done:

- Joint scenarios can be simulated for the portfolio.
- Portfolio VaR/CVaR is no longer based only on naive aggregation.
- Dependence outputs are stable enough to compare against simpler baselines.

### Phase 5: Regime Detection and Adaptive Sizing

Goal:
Condition exposure rules on market state without introducing look-ahead bias.

Deliverables:

- `regime.py` with causal feature computation,
- 3-state Gaussian HMM or equivalent regime model,
- filtered state probabilities using only past data,
- `strategy.py` sizing logic based on regime, volatility, and breach history.

Implementation notes:

- No Viterbi smoothing on the test path.
- Regime labels should be interpreted post hoc for reporting, not used in a way that leaks future information.

Definition of done:

- Portfolio sizing changes as a function of filtered regime state.
- State probabilities and realized drawdowns can be compared in reports.

### Phase 6: Reporting, Stress Tests, and Packaging

Goal:
Turn the engine into a usable research product with clear outputs.

Deliverables:

- `report.py` summary tables and plots,
- stress test slices for selected crisis periods,
- model comparison reports,
- output directory structure for reproducible experiments,
- documentation updates for setup and execution.

Stress scenarios:

- COVID crash,
- sector-specific contagion windows,
- user-configured event windows.

Definition of done:

- A full backtest run produces a coherent artifact set.
- Users can inspect model outputs without reading raw intermediate arrays.

## 8. Validation and Testing Plan

Testing must be wider than two unit tests. The engine has enough moving parts that we need layered validation.

### 8.1 Automated Tests

- `tests/test_preprocessing.py`
  Validate log-return calculations and leakage-safe expanding transforms.
- `tests/test_backtest.py`
  Detect look-ahead bias in rolling window execution.
- `tests/test_volatility.py`
  Compare baseline and custom volatility outputs on fixed samples where feasible.
- `tests/test_evt.py`
  Validate POT transformations, parameter constraints, and monotonic tail-risk outputs.
- `tests/test_strategy.py`
  Verify dollar-neutral constraints, gross-exposure constraints, and fallback allocation behavior.
- `tests/test_risk_metrics.py`
  Validate VaR/CVaR, breach counting, and statistical backtest calculations.

### 8.2 Diagnostic Validation

- residual whiteness diagnostics where relevant,
- convergence logging for every volatility fit,
- threshold stability plots for EVT,
- rolling breach plots,
- regime-state probability plots,
- portfolio exposure history plots.

### 8.3 Acceptance Metrics

At minimum, each implemented phase should be reviewed against:

- reproducibility,
- absence of leakage,
- forecast availability rate,
- optimizer failure rate,
- stability of risk forecasts,
- interpretability of outputs.

## 9. Risks and Mitigations

### Risk 1: Custom GJR-GARCH instability

Mitigation:

- keep a baseline package-backed implementation behind the same interface,
- log every optimization failure,
- use deterministic initialization and bounded retry strategies,
- do not make downstream phases depend on the custom engine being perfect.

### Risk 2: Mixed-market calendar distortions

Mitigation:

- backtest India and US baskets separately at first,
- define explicit calendar alignment rules before any cross-market portfolio is attempted.

### Risk 3: Silent fallback behavior

Mitigation:

- every fallback path must leave a diagnostic flag in the output tables,
- reports must include counts of fallbacks and failed fits.

### Risk 4: Over-expansion of scope

Mitigation:

- finish baseline risk forecasting before advanced copula families,
- finish dependence modeling before regime-aware sizing,
- keep legacy code until the new stack is validated.

## 10. Immediate Build Order

This is the order we should actually implement in the repository:

1. stabilize `src/config.py`, `src/data_loader.py`, and `src/preprocessing.py`,
2. build `src/backtest.py` and `src/risk_metrics.py`,
3. wire a baseline volatility interface in `src/volatility.py`,
4. integrate EVT in `src/evt.py`,
5. add dollar-neutral strategy logic in `src/strategy.py`,
6. add copula-based dependence modeling,
7. add regime modeling,
8. build reporting and stress-test outputs,
9. retire legacy modules only after replacement is proven.

## 11. Immediate Next Milestone

The first implementation milestone is:

"Run a leakage-safe walk-forward backtest on one basket, produce asset-level and dollar-neutral portfolio-level VaR/CVaR forecasts, record breaches, and save reproducible outputs."

Until that milestone is complete, advanced dependence families, regime overlays, and broad report polish are secondary.
