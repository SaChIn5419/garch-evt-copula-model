import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import rankdata

# Check for pyvinecopulib
try:
    import pyvinecopulib as pv
except ImportError:
    print("[ERROR] pyvinecopulib is not installed.")
    print("Please install it using: pip install pyvinecopulib")
    print("Note: This library requires a C++ compiler. If on Windows, it might be tricky.")
    # We define a dummy class so the script compiles, but it will fail at runtime if called
    class pv:
        class Vinecop: pass
        class FitControlsVinecop: pass
        class BicopFamily: 
            all = 'all'
        class RVineStructure:
            @staticmethod
            def select(data): return None

class AutomatedVineManager:
    def __init__(self, uniform_data, asset_names):
        """
        uniform_data: DataFrame of residuals transformed to [0,1]
        asset_names: List of strings
        """
        self.u = uniform_data
        self.names = asset_names
        self.model = None
        
    def diagnose_structure_logic(self):
        """
        Heuristic to tell the user IF this looks like a C-Vine or D-Vine.
        """
        # Calculate Correlation Matrix (Spearman for non-linear dependence)
        corr_matrix = self.u.corr(method='spearman').abs()
        
        # Calculate 'Centrality' (Sum of correlations for each asset)
        # Subtract 1 to remove self-correlation
        centrality_scores = corr_matrix.sum() - 1
        
        max_score = centrality_scores.max()
        mean_score = centrality_scores.mean()
        std_score = centrality_scores.std()
        
        dominant_asset = centrality_scores.idxmax()
        
        print("\n--- [AUTOMATED STRUCTURE DIAGNOSIS] ---")
        print("Asset Centrality Scores (Higher = More Connected):")
        print(centrality_scores.round(2).sort_values(ascending=False))
        
        # LOGIC: 
        # If one asset has vastly higher correlation sum than others -> C-Vine (Star)
        if max_score > (mean_score + 1.5 * std_score):
            print(f"\nDIAGNOSIS: >> C-VINE DETECTED (Star Structure) <<")
            print(f"Reason: '{dominant_asset}' is the central hub (Market Leader).")
            print("Risk Implication: If this asset crashes, the whole portfolio follows.")
        else:
            print(f"\nDIAGNOSIS: >> D-VINE / R-VINE DETECTED (Chain/Complex) <<")
            print(f"Reason: No single dominant leader. Connectivity is distributed.")
            print("Risk Implication: Contagion likely spreads primarily through specific sectors.")
            
    def fit_optimal_vine(self):
        """
        Uses pyvinecopulib to LEARN the best structure automatically.
        """
        print("\n[INFO] Learning Optimal Vine Structure from Data...")
        
        try:
            # 1. Select Controls
            # Prune tree at level 2 to keep it interpretable and fast
            controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.all], trunc_lvl=2)
            
            # 2. Fit the Vine
            # Structure.Select = The algo finds the Maximum Spanning Tree automatically
            self.model = pv.Vinecop(data=self.u.values, structure=pv.RVineStructure.select(self.u.values))
            
            print("[SUCCESS] Vine Copula Fitted.")
            print(f"Log-Likelihood: {self.model.loglik:.2f}")
            print(f"Model Parameters: {self.model.n_pars}")
            print(f"AIC: {self.model.aic:.2f}")
        except Exception as e:
            print(f"[ERROR] Fitting Vine Copula failed: {e}")
            
    def visualize_first_tree(self):
        """
        Visualizes the first layer of the tree to show dependencies.
        """
        if self.model is None:
            return

        print("\n--- PRIMARY DEPENDENCIES (Tree 1 Description) ---")
        # In pyvinecopulib, structure matrix is accessed via self.model.structure
        # But for simple visualization, the string representation is best
        print(self.model.str())
        
        # NOTE: Full graphical visualization of R-Vines requires parsing the matrix
        # which is complex. For this demo, the text structure and Diagnosis is strictly sufficient.

# ==========================================
# PORTFOLIO STRESS TEST CONFIGURATION
# ==========================================

# A Solid "Global Macro" Portfolio for Stress Testing
STRESS_PORTFOLIO = [
    '^NSEI',       # India Equity (Emerging Mkt)
    '^GSPC',       # US Equity (Developed Mkt)
    'GC=F',        # Gold (Safe Haven)
    'CL=F',        # Oil (Energy/Inflation)
    'HDFCBANK.NS', # Financials (Rate Sensitive)
    'INFY.NS',     # Tech (Export Dnepedent)
    'BTC-USD'      # Crypto (Speculative Liquidity)
]

def get_data(tickers):
    print(f"Fetching data for {len(tickers)} assets...")
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01", auto_adjust=False)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

def run_stress_test():
    print("="*60)
    print("      ADVANCED VINE COPULA STRESS TEST")
    print("="*60)
    
    # 1. Get Real Market Data
    returns = get_data(STRESS_PORTFOLIO)
    if returns.empty:
        print("No data fetched.")
        return

    # 2. Transform to Uniform (Empirical Copula approach for simplicity here)
    # In full model we used GARCH residuals, here we use raw returns ranks 
    # to test the "Raw Dependence Structure" quickly.
    print(f"\nTransforming {len(returns)} days of data to Uniform Ranking...")
    u_data = returns.apply(lambda x: rankdata(x) / (len(x) + 1))
    
    # 3. Initialize Manager
    vm = AutomatedVineManager(u_data, u_data.columns)
    
    # 4. Diagnose
    vm.diagnose_structure_logic()
    
    # 5. Fit
    if 'pyvinecopulib' in sys.modules:
         vm.fit_optimal_vine()
         vm.visualize_first_tree()
    else:
        print("\n[NOTE] Skipping Fit/Visualize because pyvinecopulib is missing.")

import sys
if __name__ == "__main__":
    run_stress_test()
