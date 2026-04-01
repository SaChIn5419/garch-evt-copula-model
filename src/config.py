"""
config.py
Global hyperparameter configuration and basket definitions
for the Quantitative Risk Pipeline.
"""

BASKETS = {
    "india_primary": [
         "^NSEI",       # NIFTY 50
         "^NSEBANK",    # NIFTY Bank
         "^CNXIT",      # NIFTY IT
         "^CNXENERGY",  # NIFTY Energy
         "^CNXPHARMA"   # NIFTY Pharma
    ],
    "us_stress": [
         "DAL",         # Delta Airlines
         "XOM",         # Exxon Mobil
         "JPM",         # JPMorgan Chase
         "BA",          # Boeing
         "AAL",         # American Airlines
         "CCL"          # Carnival Corp
    ]
}

# Default Date Ranges for the rolling backtest
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2024-01-01"
CACHE_DIR = ".cache"
RESULTS_DIR = "results"

# Walk-forward / Engine configs
ROLLING_WINDOW = 500
FORECAST_ALPHA = 0.01
MODEL_DIST = "studentt"
MODEL_P = 1
MODEL_Q = 1
MODEL_MAXITER = 500
MODEL_TOL = 1e-8
EVT_THRESHOLD_QUANTILE = 0.90
EVT_MIN_EXCEEDANCES = 25
COPULA_DF = 6
COPULA_SIMULATIONS = 5000

# Portfolio construction defaults
NET_EXPOSURE = 0.0
GROSS_EXPOSURE = 1.0
POSITION_BOUND = 0.30
