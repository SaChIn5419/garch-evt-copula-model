import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import rankdata, multivariate_t
from scipy.optimize import minimize
from arch import arch_model
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

# Suppress harmless warnings for production output
warnings.filterwarnings("ignore")

class QuantRiskPipeline:
    def __init__(self, tickers, start_date="2020-01-01", end_date="2024-01-01", regime='Extreme'):
        """
        Initializes the pipeline for a specific basket of assets.
        """
        self.tickers = tickers
        self.start = start_date
        self.end = end_date
        self.regime = regime
        
        # Internal State Storage
        self.data = None
        self.returns = None
        self.residuals = None
        self.volatility_map = {}
        self.optimal_weights = None
        
        print(f"\n[INIT] Pipeline initialized for: {tickers}")

    def run(self):
        """
        Master execution flow.
        """
        print("="*60)
        print("        STARTING GARCH-EVT-VINE PIPELINE")
        print("="*60)
        
        # 1. ETL
        self._step_1_fetch_data()
        
        # 2. GARCH Filter (Volatility Clustering Removal)
        self._step_2_fit_garch()
        
        # 3. Structure Diagnosis (Vine Copula / Network Topology)
        # We pass the residuals because Vine structure must be independent of marginal volatility
        self._step_3_diagnose_vine_structure()
        
        # 4. Simulation & Optimization (The Copula Engine)
        self._step_4_optimize_portfolio()
        
        print("\n[DONE] Pipeline execution complete.")

    def _step_1_fetch_data(self):
        print("\n--- STEP 1: DATA INGESTION ---")
        try:
            print(f"   Fetching data for {len(self.tickers)} assets...")
            raw_data = yf.download(self.tickers, start=self.start, end=self.end, auto_adjust=False, progress=False)['Adj Close']
            # Scale returns by 100 for numerical stability in optimizers
            # Handle case where only one ticker is downloaded (Series vs DataFrame)
            if isinstance(raw_data, pd.Series):
                raw_data = raw_data.to_frame()
                
            self.returns = raw_data.pct_change(fill_method=None).dropna() * 100
            print(f"   [SUCCESS] Loaded {len(self.returns)} days of data.")
        except Exception as e:
            print(f"   [ERROR] Data fetch failed: {e}")

    def _step_2_fit_garch(self):
        print("\n--- STEP 2: GARCH(1,1) FILTERING ---")
        resid_dict = {}
        
        if self.returns is None or self.returns.empty:
            print("   [SKIP] No returns data.")
            return

        for asset in self.returns.columns:
            # Fit GARCH(1,1) with Student-t distribution
            am = arch_model(self.returns[asset], vol='Garch', p=1, q=1, dist='t', rescale=False)
            res = am.fit(disp='off')
            
            # Store Conditional Volatility (for later reconstruction)
            latest_vol = np.sqrt(res.forecast(horizon=1).variance.iloc[-1].values[0])
            self.volatility_map[asset] = latest_vol
            
            # Standardize Residuals (z = r / sigma)
            std_resid = res.resid / res.conditional_volatility
            resid_dict[asset] = std_resid
            
            # Validation: Ljung-Box Test
            lb_pvalue = acorr_ljungbox(std_resid**2, lags=[10]).iloc[0]['lb_pvalue']
            status = "PASS" if lb_pvalue > 0.05 else "FAIL"
            print(f"   Asset: {asset:<15} | Vol: {latest_vol:.2f}% | White Noise Test: {status} (p={lb_pvalue:.2f})")
            
        self.residuals = pd.DataFrame(resid_dict)

    def _step_3_diagnose_vine_structure(self):
        print("\n--- STEP 3: ROBUST VINE TOPOLOGY (BOOTSTRAP) ---")
        """
        Uses the Robust Bootstrapped MST logic to detect 'Spurious Bridges'.
        """
        if self.residuals is None or self.residuals.empty:
            print("   [SKIP] No residuals data.")
            return

        # Transform residuals to Uniform Rank [0,1]
        u_data = self.residuals.apply(lambda x: rankdata(x) / (len(x) + 1))
        
        n_bootstraps = 100
        threshold_freq = 0.50
        min_corr_floor = 0.15
        edge_counts = {}
        
        print(f"   Running {n_bootstraps} bootstrapped simulations to map systemic risk...")
        
        for _ in range(n_bootstraps):
            # Resample (Bagging)
            resampled_u = u_data.sample(frac=1.0, replace=True)
            corr_matrix = resampled_u.corr(method='spearman').abs()
            
            # Build MST Candidates
            edges_temp = []
            assets = u_data.columns
            for i in range(len(assets)):
                for j in range(i+1, len(assets)):
                    u, v = assets[i], assets[j]
                    
                    # Safety check for missing columns
                    if u not in corr_matrix.index or v not in corr_matrix.index:
                        continue
                        
                    w = corr_matrix.loc[u, v]
                    
                    if w < min_corr_floor: continue
                    
                    # Hierarchical Penalty (Global -> Local -> Stock)
                    # Prevents "Stock connecting to Global" if "Stock connecting to Local" is an option
                    is_stock_u, is_stock_v = ".NS" in u, ".NS" in v
                    is_global_u = u in ['BTC-USD', 'GC=F', 'CL=F', '^GSPC', 'JPY=X', '^TNX', 'HG=F'] # Expanded Global List
                    is_global_v = v in ['BTC-USD', 'GC=F', 'CL=F', '^GSPC', 'JPY=X', '^TNX', 'HG=F']
                    
                    if (is_stock_u and is_global_v) or (is_stock_v and is_global_u):
                        w *= 0.5 # Penalty
                        
                    edges_temp.append((u, v, w))
            
            edges_temp.sort(key=lambda x: x[2], reverse=True)
            
            # Kruskal's Algorithm for MST
            uf = nx.utils.UnionFind(assets)
            for u, v, w in edges_temp:
                if uf[u] != uf[v]:
                    uf.union(u, v)
                    pair = tuple(sorted((u, v)))
                    edge_counts[pair] = edge_counts.get(pair, 0) + 1

        # Print Robust Structure
        print(f"   [STRUCTURAL DIAGNOSIS] Significant Connections (>50% Stability):")
        G = nx.Graph()
        for (u, v), count in edge_counts.items():
            freq = count / n_bootstraps
            if freq > threshold_freq:
                avg_corr = u_data[[u,v]].corr(method='spearman').iloc[0,1]
                G.add_edge(u, v, weight=avg_corr)
                print(f"   -> {u} <==> {v} (Stability: {freq:.0%} | Corr: {avg_corr:.2f})")
        
        # Centrality Check (Star vs Chain)
        if len(G.nodes) > 0:
            degrees = dict(G.degree())
            max_degree = max(degrees.values())
            if max_degree >= (len(assets) - 1) * 0.7 and len(assets) > 2: # Heuristic
                print("   >> SYSTEM TYPE: C-VINE (Star/Centralized Risk)")
            else:
                print("   >> SYSTEM TYPE: D-VINE (Chain/Decentralized Risk)")

            # --- PLOTTING LOGIC RESTORED ---
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, k=0.6, seed=42) 
            
            # Color nodes
            node_colors = []
            for node in G.nodes():
                if node in ['^NSEI', '^GSPC', '^HSI', '^TNX']: color = '#ff9999' # Indices (Red)
                elif '.NS' in node: color = '#99ff99' # Stocks (Green)
                else: color = '#99ccff' # Global/Crypto (Blue)
                node_colors.append(color)

            nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.9)
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
            
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
            plt.title(f"Robust Vine Structure (Regime: {self.regime})\nAssets: {self.tickers}")
            plt.axis('off')
            plt.show()

    def _step_4_optimize_portfolio(self):
        print("\n--- STEP 4: COPULA SIMULATION & OPTIMIZATION ---")
        
        if self.residuals is None or self.residuals.empty:
            print("   [SKIP] No residuals data.")
            return

        # 1. Filter Regime (Extreme Volatility)
        # We calculate the L2 norm of the residual vector to find 'shock days'
        magnitudes = np.linalg.norm(self.residuals, axis=1)
        # Handle cases with little data
        if len(magnitudes) < 10:
             threshold_val = 0
             extreme_indices = [True] * len(magnitudes)
        else:
            threshold_val = np.quantile(magnitudes, 0.90) # Top 10%
            extreme_indices = magnitudes > threshold_val
        
        regime_resid = self.residuals[extreme_indices]
        print(f"   Calibrated on {len(regime_resid)} Extreme Stress Days.")
        
        # 2. Fit Student-t Copula (Simplified via Correlation)
        # Transform to Uniform
        u_matrix = []
        for col in regime_resid.columns:
            ecdf = ECDF(regime_resid[col])
            # Clip to avoid inf
            u_data = np.clip(ecdf(regime_resid[col]), 1e-6, 1-1e-6) 
            u_matrix.append(u_data)
        
        u_df = pd.DataFrame(np.array(u_matrix).T, columns=regime_resid.columns)
        copula_corr = u_df.corr(method='spearman')
        
        # 3. Simulate 10,000 Scenarios
        n_sims = 10000
        # df=4 for heavy tails
        try:
             sim_t = multivariate_t.rvs(loc=np.zeros(len(copula_corr)), shape=copula_corr.values, df=4, size=n_sims)
        except Exception:
             # Fallback if matrix singular
             print("   [WARNING] Correlation matrix singular. Adding noise.")
             copula_corr += np.eye(len(copula_corr)) * 1e-5
             sim_t = multivariate_t.rvs(loc=np.zeros(len(copula_corr)), shape=copula_corr.values, df=4, size=n_sims)

        # Convert to Return Shocks
        sim_returns = np.zeros_like(sim_t)
        current_vols = np.array([self.volatility_map[name] for name in self.residuals.columns])
        
        # Map t-shocks back to Empirical Residuals (Semi-Parametric) then to Returns
        for i, asset in enumerate(self.residuals.columns):
            u_sim = multivariate_t(loc=[0], shape=[1], df=4).cdf(sim_t[:, i])
            # Inverse ECDF of original residuals
            sorted_res = np.sort(self.residuals[asset].values)
            sim_shocks = np.quantile(sorted_res, u_sim)
            # Re-scale by current GARCH volatility
            sim_returns[:, i] = current_vols[i] * sim_shocks

        # 4. Optimize (Maximize STARR Ratio)
        def objective(weights):
            port_ret = np.dot(sim_returns, weights)
            exp_ret = port_ret.mean()
            # CVaR (95%)
            losses = -port_ret
            var = np.percentile(losses, 95)
            cvar = losses[losses > var].mean()
            
            if cvar == 0: return 1e6
            starr = exp_ret / cvar
            return -starr # Minimize negative STARR

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.tickers)))
        init_w = np.ones(len(self.tickers)) / len(self.tickers)
        
        res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        
        print("\n   [OPTIMAL ALLOCATION - STARR RATIO]")
        for i, asset in enumerate(self.tickers):
            print(f"   {asset:<15}: {res.x[i]*100:.1f}%")

        # Check Final Expected Return
        final_ret = np.dot(sim_returns, res.x).mean()
        if final_ret < 0:
            print("   [NOTE] Expected Return is Negative. Model suggests Capital Preservation.")


# ==========================================
# EXECUTION BLOCK (User Interface)
# ==========================================
if __name__ == "__main__":
    import sys
    
    # Define the 5 Strategic Stress Clusters
    STRATEGIC_CLUSTERS = {
        "1": {"name": "Yen Carry Unwind", "tickers": ['JPY=X', '^NSEI', '^GSPC']},
        "2": {"name": "China Industrial", "tickers": ['HG=F', 'TATASTEEL.NS', '^HSI']},
        "3": {"name": "Real Estate Rates", "tickers": ['^TNX', 'DLF.NS', 'VNQ']},
        "4": {"name": "AI Arms Race", "tickers":  ['NVDA', 'TSM', 'INFY.NS']},
        "5": {"name": "Defensive Rotation", "tickers": ['^NSEI', 'HINDUNILVR.NS', 'ITC.NS']}
    }

    print(">>> INITIALIZING SENIOR QUANT PIPELINE")
    print(f"    Available Clusters:")
    for key, val in STRATEGIC_CLUSTERS.items():
        print(f"    [{key}] {val['name']}")
    
    # Check for CLI argument first
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        # Fallback to manual input if run directly without batch
        choice = input("\n    Select Cluster (1-5) or 'all': ").strip().lower()

    if choice == 'all':
        selected_keys = STRATEGIC_CLUSTERS.keys()
    elif choice in STRATEGIC_CLUSTERS:
        selected_keys = [choice]
    else:
        print("[ERROR] Invalid selection. Running Default (1).")
        selected_keys = ["1"]
    
    for key in selected_keys:
        cluster = STRATEGIC_CLUSTERS[key]
        print(f"\n\n{'#'*80}")
        print(f"   ANALYZING CLUSTER: {cluster['name']}")
        print(f"{'#'*80}")
        
        pipeline = QuantRiskPipeline(tickers=cluster['tickers'])
        pipeline.run()

    input("\nAnalysis Complete. Press Enter to exit...")
