import os
import sys
# Add current directory to path so modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.quant_engine import QuantEngine
import modules.forensics as forensics

# Configuration
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The "Toxic" COVID Liquidity Portfolio (March 2020 Focus)
TOXIC_TICKERS = ['DAL', 'XOM', 'JPM', 'SPG', 'GLD', 'ZM']
START_DATE = "2019-06-01" # Post-ZM IPO (April 2019) + Stabilization
END_DATE = "2020-04-30"   # Through the crash

def main():
    print("="*60)
    print("      QUANT RISK PIPELINE: LIQUIDITY CRASH FORENSICS")
    print("="*60)
    
    # 1. Initialize Engine
    engine = QuantEngine(TOXIC_TICKERS, START_DATE, END_DATE)
    
    # 2. Data Ingestion
    engine.fetch_data()
    if engine.returns is None or engine.returns.empty:
        print("[ERROR] No data. Exiting.")
        return

    # 3. GARCH Filtering (Volatility Clustering)
    engine.fit_garch()
    
    # 3b. Structure Diagnosis (Bootstrapped MST)
    engine.diagnose_vine_structure()
    
    # 4. VaR Comparison (The Forensic Evidence)
    # 4. VaR Comparison (The Forensic Evidence)
    # 4a. Standard Normal VaR (Benchmark)
    # Use window=60 (3 months) because we only have ~230 days of data and require a lag
    var_normal, eqw_ret = engine.calculate_normal_var(confidence=0.01, window=60) # 99% VaR
    
    # 4b. GARCH-EVT VaR (Challenger)
    var_evt, _ = engine.calculate_garch_evt_var(confidence=0.01) # 99% VaR
    
    # 5. Portfolio Optimization (STARR Ratio)
    # We do this to see how the model WOULD have allocated capital
    opt_weights, sim_returns = engine.optimize_portfolio_starr()
    
    # 5b. Risk Attribution (Component CVaR)
    cvar_dict = engine.calculate_component_cvar(opt_weights, sim_returns)
    
    # 6. Forensic Visualization (The 5 Plots)
    print("\n[6] Generating Forensic Analysis Plots...")
    
    # Plot 1: Gold Betrayal
    forensics.plot_gold_betrayal(engine.returns, OUTPUT_DIR)
    
    # Plot 2: Volatility Cone (Ghost in the Machine)
    forensics.plot_volatility_cone(engine.returns, OUTPUT_DIR)
    
    # Plot 3: Tail Dependence (DAL vs SPG)
    forensics.plot_tail_dependence(engine.returns, OUTPUT_DIR)
    
    # Plot 4: VaR Breach (Money Shot)
    # Note: engine.returns is multi-asset. var_normal/evt are portfolio level (equal weight)
    # forensics.plot_var_breach handles this.
    forensics.plot_var_breach(engine.returns, var_normal, var_evt, OUTPUT_DIR)
    
    # Plot 5: Topology (Systemic Risk)
    forensics.plot_topology(engine.returns, OUTPUT_DIR)
    
    # Plot 6: Risk Attribution (New!)
    forensics.plot_risk_attribution(cvar_dict, OUTPUT_DIR)
    
    print(f"\n[DONE] Pipeline execution complete. Check '{OUTPUT_DIR}' for forensic evidence.")

if __name__ == "__main__":
    main()
