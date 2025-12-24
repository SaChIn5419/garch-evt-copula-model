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

## Conceptual Difference: Student-t vs. Vine Copula
*   **The Main Model (`quant_model.py`)** uses a **Student-t Copula**. This is a "Symmetric" approach. It assumes that if the market crashes, all assets correlate in a way described by a single global structure (the degrees of freedom). It is robust, fast, and standard for **Portfolio Optimization**.
*   **The Stress Test (`vine_stress_test.py`)** uses a **Vine Copula**. This is a "Structural" approach. It decomposes the market into a tree of specific pair-wise relationships (e.g., "Gold depends on USD", "Tech depends on Nasdaq"). It is more complex but better for **Diagnosing Causality** (e.g., Is this a D-Vine chain reaction or a C-Vine market collapse?).

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
*Created by **Sachin D B***
*MSc Economics Student, Dr B R Ambedkar School of Economics University, Bengaluru*

