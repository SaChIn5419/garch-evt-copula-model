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
            # We try to use defaults if specific attributes fail
            try:
                # Some versions of python-pyvinecopulib expose families differently
                controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.all], trunc_lvl=2)
            except AttributeError:
                # Fallback: Just use defaults which usually allow all families
                print("[INFO] 'BicopFamily.all' not found. Using default internal family set.")
                controls = pv.FitControlsVinecop(trunc_lvl=2)
            
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
        
    def visualize_proxy_tree(self):
        """
        ROBUST VISUALIZATION: Uses Bootstrapping + Hierarchical Logic
        to prevent 'Spurious Bridges' (like Bitcoin connecting to Indian Banks).
        """
        print("\n--- RUNNING ROBUST STRUCTURE LEARNING (BOOTSTRAP MST) ---")
        
        n_bootstraps = 100
        threshold_freq = 0.50  # Edge must appear in 50% of simulations to be real
        min_corr_floor = 0.15  # Noise floor
        
        n_assets = len(self.names)
        edge_counts = {} # Key: (u, v), Value: count
        
        # ---------------------------------------------------------
        # 1. BOOTSTRAPPING LOOP
        # ---------------------------------------------------------
        print(f"[INFO] Running {n_bootstraps} bootstrapped topology simulations...")
        for _ in range(n_bootstraps):
            # Resample data with replacement (Bagging)
            resampled_u = self.u.sample(frac=1.0, replace=True)
            corr_matrix = resampled_u.corr(method='spearman').abs()
            
            # Build a temporary MST for this sample
            G_temp = nx.Graph()
            G_temp.add_nodes_from(self.names)
            edges_temp = []
            
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    u, v = self.names[i], self.names[j]
                    w = corr_matrix.loc[u, v]
                    
                    # FIX #3: NOISE FLOOR
                    if w < min_corr_floor:
                        continue
                        
                    # FIX #2: HIERARCHY CHECK (Heuristic)
                    # If one is a Stock (ends in .NS) and other is Crypto/Global
                    # penalize the weight so it prefers local index
                    is_stock_u = ".NS" in u
                    is_stock_v = ".NS" in v
                    is_global_u = u in ['BTC-USD', 'GC=F', 'CL=F', '^GSPC']
                    is_global_v = v in ['BTC-USD', 'GC=F', 'CL=F', '^GSPC']
                    
                    if (is_stock_u and is_global_v) or (is_stock_v and is_global_u):
                        w = w * 0.5 # Penalize 'Stock-Global' direct links by 50%
                        
                    edges_temp.append((u, v, w))
            
            # Sort and build MST (Kruskal's logic)
            edges_temp.sort(key=lambda x: x[2], reverse=True)
            uf = nx.utils.UnionFind(self.names)
            
            for u, v, w in edges_temp:
                if uf[u] != uf[v]:
                    uf.union(u, v)
                    # Count this edge
                    pair = tuple(sorted((u, v)))
                    edge_counts[pair] = edge_counts.get(pair, 0) + 1
                    
        # ---------------------------------------------------------
        # 2. FILTERING & BUILD FINAL GRAPH
        # ---------------------------------------------------------
        G_final = nx.Graph()
        G_final.add_nodes_from(self.names)
        
        print("\n--- ROBUST DEPENDENCIES (Stable > 50% of time) ---")
        for (u, v), count in edge_counts.items():
            freq = count / n_bootstraps
            if freq > threshold_freq:
                # Get the average correlation from original data for label
                avg_w = self.u[[u,v]].corr(method='spearman').iloc[0,1]
                G_final.add_edge(u, v, weight=abs(avg_w))
                print(f"  {u} <--> {v} | Stability: {freq:.0%} | Corr: {avg_w:.2f}")

        # ---------------------------------------------------------
        # 3. PLOT
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G_final, k=0.6, seed=42) # k controls spacing
        
        # Color nodes by type (Visual Hierarchy)
        node_colors = []
        for node in G_final.nodes():
            if node in ['^NSEI', '^GSPC']: color = '#ff9999' # Indices (Red)
            elif '.NS' in node: color = '#99ff99' # Stocks (Green)
            else: color = '#99ccff' # Global/Crypto (Blue)
            node_colors.append(color)

        nx.draw_networkx_nodes(G_final, pos, node_size=2000, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G_final, pos, width=2, alpha=0.6, edge_color='gray')
        nx.draw_networkx_labels(G_final, pos, font_size=9, font_weight='bold')
        
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G_final.edges(data=True)}
        nx.draw_networkx_edge_labels(G_final, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Robust Vine Structure (Bootstrapped {n_bootstraps} times)")
        plt.axis('off')
        plt.show()

# ==========================================
# PORTFOLIO STRESS TEST CONFIGURATION
# ==========================================

# EXPANDED STRESS TEST DICTIONARY
NEW_CLUSTERS = {
    "Yen_Carry_Unwind": ['JPY=X', '^NSEI', '^GSPC'],
    "China_Industrial": ['HG=F', 'TATASTEEL.NS', '^HSI'],
    "Real_Estate_Rates": ['^TNX', 'DLF.NS', 'VNQ'],
    "AI_Supply_Chain":  ['NVDA', 'TSM', 'INFY.NS'],
    "Defensive_Rotation": ['^NSEI', 'HINDUNILVR.NS', 'ITC.NS']
}

def get_data(tickers):
    print(f"Fetching data for {len(tickers)} assets...")
    try:
        data = yf.download(tickers, start="2023-01-01", end="2024-01-01", auto_adjust=False)['Adj Close']
        returns = data.pct_change(fill_method=None).dropna()
        return returns
    except Exception as e:
        print(f"[ERROR] Data fetch failed: {e}")
        return pd.DataFrame()

def run_batch_stress_test():
    for name, tickers in NEW_CLUSTERS.items():
        print(f"\n{'='*60}")
        print(f"  TESTING CLUSTER: {name}")
        print(f"{'='*60}")
        
        # 1. Get Data
        try:
            returns = get_data(tickers)
            if returns.empty:
                print(f"[SKIP] No data for {name}")
                continue
                
            # 2. Transform to Uniform (Empirical)
            u_data = returns.apply(lambda x: rankdata(x) / (len(x) + 1))
            
            # 3. Run Robust Manager
            vm = AutomatedVineManager(u_data, u_data.columns)
            vm.diagnose_structure_logic()
            vm.visualize_proxy_tree() # This uses your new Bootstrap Logic
            
        except Exception as e:
            print(f"[ERROR] Failed on {name}: {e}")

if __name__ == "__main__":
    run_batch_stress_test()
