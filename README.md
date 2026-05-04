# GARCH-EVT-Copula Portfolio Optimization

A robust, production-grade walk-forward risk engine for portfolio optimization that handles:
*   **Fat Tails** (via custom fast GJR-GARCH and EVT)
*   **Asymmetric Tail Dependence** (via Student-t Copula)
*   **Dollar-Neutral Optimization** (Risk and exposure constrained allocation)
*   **Extensive Model Comparison & Direct Diagnostics**

## Features
*   **Custom GJR-GARCH(1,1)** with Student-t innovations and relaxed boundaries to filter volatility clustering robustly and accurately. Vindicated against standard package-backed controls via 252-day out-of-sample backtests on stress segments.
*   **EVT (Extreme Value Theory) Left-Tail Adjustments** to explicitly handle severe drawdowns and tail risk.
*   **Student-t Copula** to model joint crashes and dependence structure reliably even under degenerate correlation assumptions.
*   **Strict Anti-Leakage Controls** with a rigorously tested rolling walk-forward execution engine to compute unbiased conditional VAR and CVAR.
*   **Optimization**: Deterministic dollar-neutral allocation respecting net, gross, and per-position exposure bounds.

## Architecture: Modular Walk-Forward Engine
The project has been upgraded into a modular architecture under `src/` to prioritize explainability, robustness, and flexibility:
1.  **Data Ingestion & Config**: Controlled caching and predefined evaluation baskets (`src/data_loader.py`, `src/config.py`).
2.  **GJR-GARCH Filtering**: Custom fast GJR-GARCH execution (`gjrgarch_fast.py`, `src/volatility.py`) dynamically mapping standardized residuals and conditional variance.
3.  **EVT Tail Adjustment**: Peaks-over-threshold bounds fitted on filtered residuals (`src/evt.py`).
4.  **Copula Portfolio Simulation**: Scalable multi-variate modeling using pseudo-observations (`src/copula_model.py`).
5.  **Backtest Engine & Diagnostics**: Reproducible performance validation outputs spanning Kupiec and Christoffersen exception testing.

## Usage
1.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Main Pipeline:**
    Execute the core risk engine across predefined baskets:
    ```bash
    python main.py
    ```
3.  **Run Diagnostic Tooling:**
    Compare model architectures directly on evaluation slices:
    ```bash
    python compare_walkforward_models.py
    python compare_gjr_engines.py
    ```

---
*Created by **Sachin D B***
*MSc Economics Student, Dr B R Ambedkar School of Economics University, Bengaluru*

