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
*   **Min-CVaR** and **Max-STARR** optimization objectives.

## How to Run
1.  **Install Requirements:**
    ```bash
    pip install numpy pandas yfinance arch scipy statsmodels
    ```
2.  **Run Model:**
    Double-click `run_model.bat` or run:
    ```bash
    python quant_model.py
    ```

## Logic
1.  **Marginal Models**: Fits GARCH to strip volatility.
2.  **Copula**: Calibrates a t-Copula to the residuals (focusing on extreme/tail events if configured).
3.  **Simulation**: Generates 10,000 future scenarios preserving the crash-correlation structure.
4.  **Optimization**: Finds the asset weights that maximize the STARR Ratio.

---
*Created by [Your Name/Handle]*
