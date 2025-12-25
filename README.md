# GARCH-EVT-Copula Portfolio Optimization

A "Senior Quant" level implementation of a robust portfolio optimization model that handles:
*   **Fat Tails** (via GARCH-EVT)
*   **Asymmetric Tail Dependence** (via Student-t Copula)
*   **Regime Switching** (Normal vs. Extreme volatility)
*   **STARR Ratio Optimization** (Return-Adjusted Risk)

## Features
*   **GARCH(1,1)** with Student-t innovations to filter volatility clustering.
*   **Ljung-Box Test** to validate "White Noise" assumption of residuals.
*   **Student-t Copula** (df=4) to model joint crashes (Tail Dependence).
*   **Optimization**: Maximizes **STARR Ratio** (Excess Return / CVaR) for tail-risk adjusted returns.
*   **Validation**: Includes **Ljung-Box Tests** for GARCH residual adequacy.
*   **Diagnosis**: **Robust Vine Copula** analysis (via Bootstrap) to detect structural breaks (C-Vine vs D-Vine).
*   **Backtesting**: "Crisis vs Now" feature to compare current portfolio resilience against historical crashes (e.g., COVID-19).
*   **Pipeline**: Automated `pipeline.py` orchestrator to run the full analysis suite in one go.

## Usage
1.  Run `run_pipeline.bat` (Windows) to execute the full workflow.
2.  Or run components individually:
    *   `run_model.bat`: Portfolio Optimization.
    *   `run_vine.bat`: Structural Stress Testing.

## Logic
1.  **Marginal Models**: Fits GARCH to strip volatility.
2.  **Copula**: Calibrates a t-Copula to the residuals (focusing on extreme/tail events if configured).
3.  **Simulation**: Generates 10,000 future scenarios preserving the crash-correlation structure.
4.  **Optimization**: Finds the asset weights that maximize the STARR Ratio.

---
*Created by **Sachin D B***
*MSc Economics Student, Dr B R Ambedkar School of Economics University, Bengaluru*

