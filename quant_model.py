import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import multivariate_t
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.diagnostic import acorr_ljungbox

# ==========================================
# CONFIGURATION
# ==========================================
# 1. ASSET SELECTION
# You can change these tickers to any valid Yahoo Finance symbol.
# Examples: '^NSEI' (Nifty 50), 'GC=F' (Gold), 'BTC-USD' (Bitcoin), 'AAPL' (Apple)
TICKERS = ['^NSEI', 'GC=F']

# 2. REGIME SELECTION
# Options: 'All', 'Extreme', 'Normal'
# 'All': Use all historical data to model correlation.
# 'Extreme': Use only the most volatile 10% of days to model correlation (Stress Testing).
# 'Normal': Use only the least volatile 90% of days (Business as Usual).
REGIME = 'Extreme' 

# Threshold for defining "Extreme" (e.g., top 10% of joint shocks)
EXTREME_THRESHOLD = 0.10 

# Copula Degrees of Freedom (Lower = Fatter Tails/More Crash Dependence)
COPULA_DF = 4

# Risk Free Rate for STARR Ratio (Daily)
RISK_FREE_RATE = 0.05 / 252 

# ==========================================
# MODEL IMPLEMENTATION
# ==========================================

def get_data(tickers, start="2020-01-01", end="2024-01-01"):
    """Downloads adjusted close prices and calculates returns."""
    print(f"Fetching data for {tickers}...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    # Scale returns by 100 to avoid optimizer "small number" issues
    returns = data.pct_change().dropna() * 100
    return returns

def fit_garch(returns):
    """
    Fits a GARCH(1,1) model to each asset to standardized residuals.
    Returns:
        - std_residuals: DataFrame of (return / volatility)
        - latest_vol: The forecasted volatility for the next time step
    """
    residuals_map = {}
    latest_vol_map = {}
    
    print("\nFitting GARCH(1,1) models...")
    for asset in returns.columns:
        # Using Student's t-distribution for better fat-tail handling in the marginals
        # rescale=False because we manually scaled by 100 above
        am = arch_model(returns[asset], vol='Garch', p=1, q=1, dist='t', rescale=False)
        res = am.fit(disp='off')
        
        print(f" - {asset}: Optimization Success = {res.convergence_flag == 0}")
        
        # Standardize residuals: z_t = r_t / sigma_t
        std_resid = res.resid / res.conditional_volatility
        residuals_map[asset] = std_resid
        
        # Forecast volatility for tomorrow
        # horizon=1 gives a forecast for the next step
        forecast = res.forecast(horizon=1)
        # Extract the variance forecast and take square root for volatility
        # The forecast index corresponds to the last date in data
        latest_vol_map[asset] = np.sqrt(forecast.variance.iloc[-1].values[0])
        
        print(f" - {asset}: Current Volatility = {latest_vol_map[asset]:.4f}")

    return pd.DataFrame(residuals_map), latest_vol_map

def test_residuals(std_residuals):
    """
    Validates if GARCH residuals are truly i.i.d (white noise) using Ljung-Box test.
    We test the squared residuals to check for remaining volatility clustering.
    """
    print("\n[VALIDATION] Running Ljung-Box Test on Residuals...")
    all_pass = True
    for asset in std_residuals.columns:
        # Test up to 10 lags
        lb_test = acorr_ljungbox(std_residuals[asset]**2, lags=[10], return_df=True)
        p_value = lb_test.iloc[0]['lb_pvalue']
        
        status = "PASS" if p_value > 0.05 else "FAIL"
        if status == "FAIL": all_pass = False
        
        print(f" - {asset}: p-value={p_value:.4f} [{status}] (Null Hypothesis: No Autocorrelation)")
    
    if not all_pass:
        print("  [WARNING] Some residuals show autocorrelation. GARCH model might need tuning (e.g., different p,q).")
    else:
        print("  [SUCCESS] Residuals look like white noise. EVT assumption holds.")

def filter_regime(std_residuals, regime='All', threshold=0.10):
    """
    Filters the residuals based on the selected market regime.
    We use the Euclidean norm of the standardized residuals vector to estimate 'magnitude' of shock.
    """
    if regime == 'All':
        return std_residuals
    
    # Calculate magnitude of shock for each day: sqrt(z1^2 + z2^2 + ...)
    magnitudes = np.linalg.norm(std_residuals, axis=1)
    
    # Determine cutoff percentile
    qt = np.quantile(magnitudes, 1 - threshold)
    
    if regime == 'Extreme':
        # Keep days where shock magnitude is greater than threshold
        mask = magnitudes > qt
        print(f"\nFiltering for EXTREME regime (Top {threshold*100}% of volatility shocks).")
        print(f"Data points selected: {sum(mask)} / {len(mask)}")
    elif regime == 'Normal':
        # Keep days where shock magnitude is lower than threshold
        mask = magnitudes <= qt
        print(f"\nFiltering for NORMAL regime (Bottom {(1-threshold)*100}% of volatility shocks).")
        print(f"Data points selected: {sum(mask)} / {len(mask)}")
    else:
        raise ValueError(f"Unknown regime: {regime}")
        
    return std_residuals[mask]

def fit_copula_and_simulate(filtered_residuals, n_sims=10000):
    """
    1. Transforms residuals to Uniform [0,1] via Empirical CDF.
    2. Calculates Spearman Rank Correlation on these Uniforms.
    3. Simulates new random scenarios using this correlation structure.
    """
    # 1. Transform to Uniform (PIT - Probability Integral Transform)
    u_matrix = []
    
    # We must compute ECDF based on the inputs provided (the regime subset)
    # Note: For a stricter Semi-Parametric approach, one might stick the 
    # regime-filtered correlation onto the 'All' marginals, but typically
    # if we stress test, we care about the dependence structure IN stress.
    for col in filtered_residuals.columns:
        ecdf = ECDF(filtered_residuals[col])
        u_data = ecdf(filtered_residuals[col])
        # Clip to avoid infs in inverse transform if we were using parametric, 
        # but helpful for stability generally.
        u_data = np.clip(u_data, 1e-6, 1-1e-6)
        u_matrix.append(u_data)
    
    u_df = pd.DataFrame(np.array(u_matrix).T, columns=filtered_residuals.columns)
    
    # 2. Spearman Correlation (Dependence Structure)
    # This captures the non-linear relationship
    copula_corr = u_df.corr(method='spearman')
    print("\nCalibrated Dependence Structure (Correlation Matrix):")
    print(copula_corr)
    


    # 3. Simulate Scenarios (Student-t Copula)
    # We generate variables from a multivariate t-distribution
    # This captures tail dependence better than Gaussian.
    mean = np.zeros(len(copula_corr))
    # Correlation matrix acts as the shape matrix for the t-distribution
    shape_matrix = copula_corr.values
    
    # Generate correlated t-distributed variables
    # df=4 implies heavy tails (crashes likely happen together)
    sim_t = multivariate_t.rvs(loc=mean, shape=shape_matrix, df=COPULA_DF, size=n_sims)
    
    # Transform t-distributed margins back to Uniform, then could map to whatever margin we want.
    # But since we want "standardized shocks" for our returns, and our GARCH residuals were 
    # approx t-distributed anyway, we can largely use these directly or transform them 
    # to match the exact ECDF of residuals.
    # "Canonical Vine" approach often maps U -> Empirical Quantile.
    
    # Simplified Robust Approach: 
    # Use the rank structure from the t-copula, but map to the Empirical distribution of residuals.
    # This preserves the specific shape of YOUR asset's history.
    
    sim_shocks = np.zeros_like(sim_t)
    for i, col in enumerate(filtered_residuals.columns):
        # 1. Convert t-simulations to Uniform [0,1] using t CDF
        u_sim = multivariate_t(loc=[0], shape=[1], df=COPULA_DF).cdf(sim_t[:, i])
        
        # 2. Map Uniforms to Empirical Residuals (Inverse ECDF / Quantile)
        # We use interpolation to map [0,1] back to the actual residual values
        # Sort residuals to create a lookup
        sorted_res = np.sort(filtered_residuals[col].values)
        sim_shocks[:, i] = np.quantile(sorted_res, u_sim)
        
    return sim_shocks

def calculate_portfolio_cvar(weights, returns_scenarios, alpha=0.05):
    """Calculates Conditional Value at Risk (Expected Shortfall)"""
    # Portfolio return for each simulated scenario
    port_returns = np.dot(returns_scenarios, weights)
    losses = -port_returns
    
    # VaR is the (1-alpha) percentile of loss
    var = np.percentile(losses, (1-alpha)*100)
    
    # CVaR is average of losses exceeding VaR
    cvar = losses[losses > var].mean()
    return cvar

def calculate_starr_ratio(weights, returns_scenarios, risk_free_rate=0.0):
    """
    Calculates STARR Ratio: (Expected Return - Rf) / CVaR.
    We want to MAXIMIZE this, so we return negative for minimization.
    """
    port_returns = np.dot(returns_scenarios, weights)
    expected_return = port_returns.mean()
    cvar = calculate_portfolio_cvar(weights, returns_scenarios)
    
    # Avoid division by zero
    if cvar == 0: return 1e6
    
    starr = (expected_return - risk_free_rate) / cvar
    return -starr # Return negative for minimization

def main():
    # 1. Data
    returns = get_data(TICKERS)
    
    # 2. GARCH (Marginals)
    std_residuals, latest_vol_map = fit_garch(returns)
    
    # 3. Regime Filtering
    regime_residuals = filter_regime(std_residuals, regime=REGIME, threshold=EXTREME_THRESHOLD)
    
    # 4. Copula Simulation
    # Generate correlated standard shocks
    sim_shocks = fit_copula_and_simulate(regime_residuals)
    
    # 5. Transform Shocks to Returns
    # Return = Forecast_Vol * Simulated_Shock
    sim_returns = np.zeros_like(sim_shocks)
    
    asset_names = returns.columns
    current_vols = np.array([latest_vol_map[name] for name in asset_names])
    
    for i in range(len(asset_names)):
        sim_returns[:, i] = current_vols[i] * sim_shocks[:, i]
        
    # 6. Optimization
    # Verify basics first
    test_residuals(regime_residuals)

    print("\nOptimizing Portfolio for Maximum STARR Ratio (Return / CVaR)...")
    n_assets = len(asset_names)
    initial_weights = np.ones(n_assets) / n_assets
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    result = minimize(
        calculate_starr_ratio, 
        initial_weights, 
        args=(sim_returns, RISK_FREE_RATE), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    print("-" * 50)
    print(f"RESULTS (Regime: {REGIME})")
    print("-" * 50)
    for i, asset in enumerate(asset_names):
        print(f"Weight {asset:<10}: {result.x[i]:.4f}")
    
    # Calculate final metrics for the optimal weights
    opt_cvar = calculate_portfolio_cvar(result.x, sim_returns)
    opt_return = np.dot(sim_returns, result.x).mean()
    print(f"\nPredicted 1-Day CVaR (95%): {opt_cvar:.5f}%")
    print(f"Expected 1-Day Return     : {opt_return:.5f}%")
    print(f"STARR Ratio               : {(opt_return - RISK_FREE_RATE)/opt_cvar:.5f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
