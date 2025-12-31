import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy.stats import norm
import os

# Output Directory
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup Plotting Style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

def fetch_data(ticker="^NSEI", start="2020-01-01", end="2024-01-01"):
    print(f"\n[1] Fetching Data for {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    
    # Handle multi-index columns if they exist (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            adj_close = df[('Adj Close', ticker)]
        except KeyError:
            adj_close = df['Adj Close']  # Fallback
            if isinstance(adj_close, pd.DataFrame):
                 adj_close = adj_close.iloc[:, 0]
    else:
        adj_close = df['Adj Close']

    # Calculate Returns
    returns = adj_close.pct_change().dropna() * 100 # In percentage
    
    # Ensure Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()
        
    print(f"    Loaded {len(returns)} days of data. Shape: {returns.shape}")
    print(f"    Head: {returns.head()}")
    return returns

def calculate_normal_var(returns, window=252, confidence=0.05):
    """
    Calculates Normal VaR using Rolling Mean and Standard Deviation.
    VaR = Mean - Z * StdDev
    """
    print(f"\n[2] Calculating Normal VaR (Rolling Window: {window})...")
    
    try:
        # Rolling Statistics
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        
        # Z-Score for Confidence (e.g., 1.645 for 95%)
        z_score = norm.ppf(1 - confidence)
        
        # VaR
        vars_normal = rolling_mean - z_score * rolling_std
        vars_normal = vars_normal.dropna()
        print(f"    Normal VaR Shape: {vars_normal.shape}")
        return vars_normal
    except Exception as e:
        print(f"    [Error in Normal VaR] {e}")
        return pd.Series()

def calculate_garch_evt_var(returns, confidence=0.05):
    """
    Calculates VaR using GARCH(1,1) Volatility + Empirical Quantile (FHS).
    This captures 'Fat Tails' (EVT proxy) and 'Volatility Clustering' (GARCH).
    """
    print(f"\n[3] Calculating GARCH-EVT VaR...")
    
    # 1. Fit GARCH(1,1)
    # We use the entire history to estimate the model parameters (In-Sample) 
    # to demonstrate the structural difference. A rolling fit would be similar but slower.
    am = arch_model(returns, vol='Garch', p=1, q=1, dist='t', rescale=False)
    res = am.fit(disp='off')
    
    # 2. Get Conditional Volatility (Sigma_t)
    cond_vol = res.conditional_volatility
    
    # 3. Get Standardized Residuals
    std_resid = res.resid / cond_vol
    
    # 4. Calculate Quantile of Residuals (EVT / Historical approach)
    # Instead of assuming -1.645 (Normal), we take the ACTUAL 5th percentile of the residuals
    # This accounts for Skewness and Kurtosis (Fat Tails)
    empirical_quantile = np.percentile(std_resid, confidence * 100)
    
    print(f"    Normal Z-Score (5%): {norm.ppf(confidence):.4f}")
    print(f"    GARCH-EVT Resid Quantile (5%): {empirical_quantile:.4f} (Broader Tail!)")
    
    # 5. Calculate Dynamic VaR
    # VaR_t = Mean_Forecast + Sigma_t * Quantile_Resid
    # Assuming Mean ~ 0 for simplicity or use GARCH mean model
    
    # Re-align indices
    vars_garch_evt = cond_vol * empirical_quantile
    
    return vars_garch_evt, res

def analyze_breaches(returns, var_series, name="Model"):
    """
    Identifies days where Actual Return < VaR Limit
    """
    # Align dates
    common_idx = returns.index.intersection(var_series.index)
    pf_ret = returns.loc[common_idx]
    pf_var = var_series.loc[common_idx]
    
    breaches = pf_ret < pf_var
    n_breaches = breaches.sum()
    pct_breaches = n_breaches / len(common_idx)
    
    print(f"    [{name}] Breaches: {n_breaches} / {len(common_idx)} ({pct_breaches:.2%})")
    return pf_ret[breaches]


def simulate_risk_managed_portfolio(scenario_name, returns, var_normal, var_garch, initial_capital=100000, target_var_pct=0.01):
    """
    Simulates a Risk-Managed Portfolio.
    Rule: Allocate Capital such that expected 1-day VaR = 1% of Equity.
    Position_Value = (Equity * Target_VaR_PCT) / Model_VaR_PCT
    
    If Model_VaR is low (safer), we leverage up.
    If Model_VaR is high (risky), we de-leverage.
    """
    print(f"\n[5] Simulating Risk-Managed Capital Allocation (Target VaR: {target_var_pct:.1%})...")
    
    # Align Data
    common_idx = returns.index.intersection(var_normal.index).intersection(var_garch.index)
    
    # Storage
    equity_norm = [initial_capital]
    equity_garch = [initial_capital]
    
    pos_norm_trace = []
    pos_garch_trace = []
    
    ret_series = returns.loc[common_idx] / 100 # Convert back to decimal
    
    # Absolute VaR (positive number for division)
    v_norm_series = var_normal.loc[common_idx].abs() / 100
    v_garch_series = var_garch.loc[common_idx].abs() / 100
    
    for date in common_idx:
        # 1. Determine Position Size based on YESTERDAY'S Forecast (Conceptually)
        # We use current day row for simplicity as 'var' is ex-ante in this context (or we assume we rebalance at open)
        
        # Risk Budget: $1,000 (1% of 100k)
        # Position = $1,000 / 0.02 (2% VaR) = $50,000 Exposure
        
        # Normal Strategy
        cap_n = equity_norm[-1]
        risk_budget_n = cap_n * target_var_pct
        # Clamp VaR to avoid division by zero or extreme leverage
        est_var_n = max(v_norm_series.loc[date], 0.001) 
        pos_n = risk_budget_n / est_var_n
        
        # GARCH Strategy
        cap_g = equity_garch[-1]
        risk_budget_g = cap_g * target_var_pct
        est_var_g = max(v_garch_series.loc[date], 0.001)
        pos_g = risk_budget_g / est_var_g
        
        # 2. Calculate PnL
        # PnL = Position * Realized_Return
        day_ret = ret_series.loc[date]
        
        new_cap_n = cap_n + (pos_n * day_ret)
        new_cap_g = cap_g + (pos_g * day_ret)
        
        # Prevent Bankruptcy
        equity_norm.append(max(new_cap_n, 0))
        equity_garch.append(max(new_cap_g, 0))
        
        pos_norm_trace.append(pos_n)
        pos_garch_trace.append(pos_g)
        
    # Convert to Series
    equity_norm = pd.Series(equity_norm[1:], index=common_idx)
    equity_garch = pd.Series(equity_garch[1:], index=common_idx)
    pos_norm = pd.Series(pos_norm_trace, index=common_idx)
    pos_garch = pd.Series(pos_garch_trace, index=common_idx)
    
    # Stats
    dd_norm = (equity_norm.cummax() - equity_norm).max()
    dd_garch = (equity_garch.cummax() - equity_garch).max()
    
    print(f"    [Normal VaR] Final Equity: ${equity_norm.iloc[-1]:,.0f} | Max Margin Used: ${pos_norm.max():,.0f}")
    print(f"    [GARCH-EVT]  Final Equity: ${equity_garch.iloc[-1]:,.0f} | Max Margin Used: ${pos_garch.max():,.0f}")
    
    # --- Graph 4: Capital Preservation ---
    plt.figure(figsize=(15, 8))
    
    # Subplot 1: Equity Curve
    plt.subplot(2, 1, 1)
    plt.plot(equity_norm.index, equity_norm, color='red', label='Strategy with Normal VaR (Blind to Risk)', linewidth=1.5)
    plt.plot(equity_garch.index, equity_garch, color='green', label='Strategy with GARCH-EVT (Responsive)', linewidth=1.5)
    plt.title("Impact of Risk Model on PnL (Target Risk: 1% of Equity)", fontsize=14)
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    
    # Subplot 2: Exposure (Leverage)
    plt.subplot(2, 1, 2)
    plt.plot(pos_norm.index, pos_norm, color='red', alpha=0.6, label='Normal VaR Exposure (Constant/Slow)')
    plt.plot(pos_garch.index, pos_garch, color='blue', alpha=0.6, label='GARCH-EVT Exposure (Dynamic)')
    plt.title("Responsiveness: Position Sizing during Volatility", fontsize=14)
    plt.ylabel("Position Size ($)")
    plt.legend()
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, f"{scenario_name}_capital_preservation.png")
    plt.savefig(filename)
    print(f"    Saved: {filename}")
    plt.close()

def plot_comparisons(returns, var_normal, var_garch, breaches_normal, breaches_garch):
    print("\n[4] Generating Graphs...")
    
    # --- Graph 1: Time Series Comparison ---
    plt.figure(figsize=(15, 7))
    plt.plot(returns.index, returns, color='gray', alpha=0.3, label='Daily Returns')
    plt.plot(var_normal.index, var_normal, color='red', linestyle='--', label='Normal VaR (95%)', linewidth=1.5)
    plt.plot(var_garch.index, var_garch, color='blue', label='GARCH-EVT VaR (95%)', linewidth=1.5)
    
    # Highlight Breaches
    plt.scatter(breaches_normal.index, breaches_normal, color='red', s=20, marker='x', label='Normal VaR Failure', zorder=5)
    
    plt.title("Normal VaR vs. GARCH-EVT VaR: Capturing Volatility Clusters", fontsize=16)
    plt.ylabel("Return (%)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "var_comparison_timeseries.png"))
    print("    Saved: var_comparison_timeseries.png")
    plt.close()

    # --- Graph 2: Zoom in on Crises (e.g., COVID March 2020) ---
    max_vol_date = returns.abs().idxmax()
    start_zoom = max_vol_date - pd.Timedelta(days=60)
    end_zoom = max_vol_date + pd.Timedelta(days=60)
    
    mask_ret = (returns.index >= start_zoom) & (returns.index <= end_zoom)
    ret_zoom = returns[mask_ret]
    
    # Calculate mask specific to var_normal (which is shorter due to rolling window)
    mask_norm = (var_normal.index >= start_zoom) & (var_normal.index <= end_zoom)
    norm_zoom = var_normal[mask_norm]
    
    # Calculate mask specific to var_garch
    mask_garch = (var_garch.index >= start_zoom) & (var_garch.index <= end_zoom)
    garch_zoom = var_garch[mask_garch]
    
    plt.figure(figsize=(15, 7))
    plt.plot(ret_zoom.index, ret_zoom, color='gray', alpha=0.5, marker='o', label='Daily Returns')
    plt.plot(norm_zoom.index, norm_zoom, color='red', linestyle='--', label='Normal VaR (Slow to React)')
    plt.plot(garch_zoom.index, garch_zoom, color='blue', label='GARCH-EVT VaR (Dynamic)')
    
    plt.fill_between(garch_zoom.index, garch_zoom, min(ret_zoom.min(), garch_zoom.min())-1, color='blue', alpha=0.1)
    
    plt.title(f"Why Normal VaR Fails During Crisis ({max_vol_date.strftime('%Y')})", fontsize=16)
    plt.ylabel("Return (%)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "var_comparison_zoom.png"))
    print("    Saved: var_comparison_zoom.png")
    plt.close()
    
    # --- Graph 3: Distribution Analysis (QQ Plot / Density) ---
    plt.figure(figsize=(12, 6))
    sns.kdeplot(returns, fill=True, label='Actual Returns (Fat Tails)', color='green')
    
    # Generate Normal Distribution with same Mean/Std
    mu, std = norm.fit(returns)
    x = np.linspace(returns.min(), returns.max(), 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=2, label='Normal Assumption')
    
    # Mark VaR points
    plt.axvline(np.percentile(returns, 5), color='green', linestyle=':', label='Actual 5% Quantile')
    plt.axvline(norm.ppf(0.05, mu, std), color='red', linestyle=':', label='Normal 5% Quantile')
    
    plt.title("Distribution mismatch: The 'Fat Tail' Problem", fontsize=16)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "var_distribution.png"))
    print("    Saved: var_distribution.png")
    plt.close()

def make_plots_for_scenario(scenario, returns, var_normal, var_garch):
    name = scenario['name']
    print(f"\n[Graphing] {name}...")
    
    # 1. Zoom Plot (The Breach)
    mask_zoom = (returns.index >= scenario['zoom_start']) & (returns.index <= scenario['zoom_end'])
    
    r_zoom = returns[mask_zoom]
    if r_zoom.empty:
        print(f"   [WARNING] No data for Zoom period {scenario['zoom_start']}-{scenario['zoom_end']}")
        return
        
    vn_zoom = var_normal.reindex(r_zoom.index)
    vg_zoom = var_garch.reindex(r_zoom.index)
    
    plt.figure(figsize=(14, 7))
    plt.plot(r_zoom.index, r_zoom, color='gray', alpha=0.5, marker='o', label='Daily Returns')
    plt.plot(vn_zoom.index, vn_zoom, color='red', linestyle='--', linewidth=2, label='Normal VaR (Fails)')
    plt.plot(vg_zoom.index, vg_zoom, color='blue', linewidth=2, label='GARCH-EVT (Reacts)')
    
    # Fill the gap
    plt.fill_between(vg_zoom.index, vg_zoom, min(r_zoom.min(), vg_zoom.min())-1, color='blue', alpha=0.1)
    
    plt.title(f"Forensic Analysis: {scenario['desc']} ({scenario['ticker']})", fontsize=16)
    plt.ylabel("Returns (%)")
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    filename = os.path.join(OUTPUT_DIR, f"{name}_zoom.png")
    plt.savefig(filename)
    print(f"   Saved: {filename}")
    plt.close()



def main():
    print(">>> STARTING ROBUST VAR ANALYSIS (^NSEI) <<<")
    
    try:
        # 1. Fetch Data (Nifty 50 - Covers COVID Crash)
        ticker = "^NSEI"
        start_date = "2018-01-01"
        end_date = "2024-01-01" # Post-COVID recovery included
        
        returns = fetch_data(ticker=ticker, start=start_date, end=end_date)
        if returns.empty:
            print("   [ERROR] No data found.")
            return

        # 2. Model
        var_norm = calculate_normal_var(returns)
        var_garch, _ = calculate_garch_evt_var(returns)
        
        # 3. Analyze
        print("\n[Analysis] Checking for Failures (Returns < VaR)...")
        breaches_normal = analyze_breaches(returns, var_norm, name="Normal VaR")
        breaches_garch = analyze_breaches(returns, var_garch, name="GARCH-EVT VaR")
        
        # 4. Plot Qualitative Comparisons
        # Scenario: COVID-19 Crash (March 2020)
        scenario = {
            "name": "Nifty_Covid_Crash",
            "ticker": ticker,
            "desc": "Pandemic Volatility (2020)",
            "zoom_start": "2020-02-15",
            "zoom_end": "2020-04-15"
        }
        
        # A. General Plots (Time Series, Dist)
        plot_comparisons(returns, var_norm, var_garch, breaches_normal, breaches_garch)
        
        # B. Forensic Zoom Plot
        make_plots_for_scenario(scenario, returns, var_norm, var_garch)
        
        # 5. Simulation (Capital Preservation)
        simulate_risk_managed_portfolio("Nifty_Covid_Simulation", returns, var_norm, var_garch, target_var_pct=0.01)

        print("\n>>> ANALYSIS COMPLETE <<<")
        
    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
