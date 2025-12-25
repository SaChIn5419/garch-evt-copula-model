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
*   **Optimization**: Maximizes **STARR Ratio** (Excess Return / CVaR)## Architecture: The `QuantRiskPipeline` Class
The project has been refactored into a single, professional-grade Python class (`QuantRiskPipeline`) that handles the entire lifecycle:
1.  **Data Ingestion**: Automatic fetching and scaling.
2.  **GARCH Filtering**: Volatility clustering removal.
3.  **Vine Diagnosis**: Robust Bootstrapped MST for structure detection.
4.  **Copula Optimization**: Student-t Simulations for Max-STARR allocation.

## Usage
1.  **Install Requirements:**
    ```bash
    pip install numpy pandas yfinance arch scipy statsmodels networkx matplotlib
    ```
2.  **Run the Pipeline:**
    Double-click `run_pipeline.bat` (Windows).
    *   **Interactive Menu**: You will be asked to select a specific crisis cluster (e.g., "1" for Yen Carry Unwind) or type "all" to run the full battery.
    *   **Visualization**: Network graphs will popup automatically for each cluster.

## Logic
1.  **Marginal Models**: Fits GARCH to strip volatility.
2.  **Copula**: Calibrates a t-Copula to the residuals (focusing on extreme/tail events if configured).
3.  **Simulation**: Generates 10,000 future scenarios preserving the crash-correlation structure.
4.  **Optimization**: Finds the asset weights that maximize the STARR Ratio.

---
*Created by **Sachin D B***
*MSc Economics Student, Dr B R Ambedkar School of Economics University, Bengaluru*

