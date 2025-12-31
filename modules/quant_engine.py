import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy.stats import rankdata, multivariate_t, norm
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.diagnostic import acorr_ljungbox
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

class QuantEngine:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start = start_date
        self.end = end_date
        
        # State
        self.returns = None
        self.residuals = None
        self.volatility_map = {}
        self.garch_models = {}
        self.copula_corr = None
        
    def fetch_data(self):
        print(f"\n[1] Fetching Data for {len(self.tickers)} assets ({self.start} to {self.end})...")
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Attempt download
                raw = yf.download(self.tickers, start=self.start, end=self.end, progress=False, auto_adjust=False)['Adj Close']
                
                if isinstance(raw, pd.Series):
                    raw = raw.to_frame()
                
                # Check if we actually got data for all tickers
                missing_tickers = [t for t in self.tickers if t not in raw.columns]
                if missing_tickers:
                    print(f"    [WARNING] Attempt {attempt+1}: Missing data for {missing_tickers}. Retrying...")
                    time.sleep(2)
                    continue

                # Forward fill then backwards fill to handle slight exchange holidays mismatches
                raw = raw.ffill().bfill()
                
                self.returns = raw.pct_change().dropna() * 100
                
                if self.returns.empty:
                     print(f"    [WARNING] Attempt {attempt+1}: Data loaded but resulted in empty returns (likely alignment issue). Retrying...")
                     time.sleep(2)
                     continue
                     
                print(f"    Loaded {len(self.returns)} days. Shape: {self.returns.shape}")
                return self.returns

            except Exception as e:
                print(f"    [WARNING] Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        
        print("    [ERROR] All data fetch attempts failed.")
        return pd.DataFrame()

    def fit_garch(self):
        print("\n[2] GARCH(1,1) Filtering (Removing Volatility Clusters)...")
        resid_dict = {}
        
        if self.returns is None: return

        for asset in self.returns.columns:
            # Fit GARCH(1,1) Student-t
            am = arch_model(self.returns[asset], vol='Garch', p=1, q=1, dist='t', rescale=False)
            res = am.fit(disp='off')
            self.garch_models[asset] = res
            
            # Store Volatility
            latest_vol = np.sqrt(res.forecast(horizon=1).variance.iloc[-1].values[0])
            self.volatility_map[asset] = latest_vol
            
            # Standardize Residuals
            std_resid = res.resid / res.conditional_volatility
            resid_dict[asset] = std_resid
            
            # Validation: Ljung-Box Test (White Noise Check)
            lb_pvalue = acorr_ljungbox(std_resid**2, lags=[10]).iloc[0]['lb_pvalue']
            status = "PASS" if lb_pvalue > 0.05 else "FAIL"
            print(f"    Asset: {asset:<6} | Vol: {latest_vol:.2f}% | White Noise: {status} (p={lb_pvalue:.2f})")
            
        self.residuals = pd.DataFrame(resid_dict)

    def diagnose_vine_structure(self):
        print("\n[3] Diagnosing Systemic Topology (Robust Bootstrapped MST)...")
        if self.residuals is None: return

        # Transform to Uniform Rank [0,1]
        u_data = self.residuals.apply(lambda x: rankdata(x) / (len(x) + 1))
        
        n_bootstraps = 100
        threshold_freq = 0.50
        min_corr_floor = 0.15
        edge_counts = {}
        
        print(f"    Running {n_bootstraps} bootstrapped simulations to map systemic risk...")
        
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
                    
                    if u not in corr_matrix.index or v not in corr_matrix.index: continue
                        
                    w = corr_matrix.loc[u, v]
                    if w < min_corr_floor: continue
                    
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
        print(f"    [STRUCTURAL DIAGNOSIS] Significant Connections (>50% Stability):")
        G = nx.Graph()
        for (u, v), count in edge_counts.items():
            freq = count / n_bootstraps
            if freq > threshold_freq:
                avg_corr = u_data[[u,v]].corr(method='spearman').iloc[0,1]
                G.add_edge(u, v, weight=avg_corr)
                print(f"    -> {u} <==> {v} (Stability: {freq:.0%} | Corr: {avg_corr:.2f})")
        
        # Centrality Check
        if len(G.nodes) > 0:
            degrees = dict(G.degree())
            max_degree = max(degrees.values())
            if max_degree >= (len(assets) - 1) * 0.7: 
                print("    >> SYSTEM TYPE: C-VINE (Star/Centralized Risk)")
            else:
                print("    >> SYSTEM TYPE: D-VINE (Chain/Decentralized Risk)")

    def calculate_normal_var(self, confidence=0.05, window=252):
        print(f"\n[4a] Benchmark: Normal VaR (Rolling {window}d)...")
        if self.returns is None: return pd.Series()
        
        # Portfolio Level? Or Asset Level? 
        # The user wants "Portfolio Level" for the comparison chart.
        # Let's assume Equal Weight Portfolio for the "Money Shot"
        
        # If returns is multi-asset, we can either do this per asset or for an EQW portfolio
        # For the "Univariate" Forensic Analysis (like SPY or Portfolio), we usually pass a Series
        # But here we have a DataFrame.
        
        # Let's create an EQUAL WEIGHT INDEX for the VaR check
        eqw_returns = self.returns.mean(axis=1)
        
        roll_mean = eqw_returns.rolling(window).mean()
        roll_std = eqw_returns.rolling(window).std()
        z_score = norm.ppf(1 - confidence)
        var_normal = roll_mean - z_score * roll_std
        return var_normal.dropna(), eqw_returns

    def calculate_garch_evt_var(self, confidence=0.05):
        print(f"\n[4b] Challengers: GARCH-EVT VaR (Dynamic)...")
        # Again, for the portfolio level comparison
        eqw_returns = self.returns.mean(axis=1)
        
        # Fit GARCH on the Portfolio itself (Univariate GARCH on Portfolio Returns)
        am = arch_model(eqw_returns, vol='Garch', p=1, q=1, dist='t', rescale=False)
        res = am.fit(disp='off')
        
        cond_vol = res.conditional_volatility
        
        # EVT / t-dist quantile
        # We use the fitted Degrees of Freedom
        df = res.params['nu']
        from scipy.stats import t
        t_q = t.ppf(confidence, df=df) # 5% quantile is negative, e.g. -1.64 or -2.something
        
        var_evt = cond_vol * t_q # Vol is positive, Quantile is negative -> VaR is negative return
        
        return var_evt, res

    def optimize_portfolio_starr(self, n_sims=1000):
        print(f"\n[5] Optimizing Portfolio (Max STARR Ratio) via Copula Simulation...")
        if self.residuals is None: return
        
        # 1. Copula Fit (Correlation of Uniforms)
        u_matrix = []
        for col in self.residuals.columns:
            ecdf = ECDF(self.residuals[col])
            u_data = np.clip(ecdf(self.residuals[col]), 1e-6, 1-1e-6)
            u_matrix.append(u_data)
        
        u_df = pd.DataFrame(np.array(u_matrix).T, columns=self.residuals.columns)
        self.copula_corr = u_df.corr(method='spearman')
        
        # 2. Simulate
        try:
            # Multivariate t (df=4 for fat tails)
            sim_t = multivariate_t.rvs(loc=np.zeros(len(self.tickers)), 
                                       shape=self.copula_corr.values, 
                                       df=4, size=n_sims)
        except:
             # Fix Singularity
             self.copula_corr += np.eye(len(self.tickers)) * 1e-4
             sim_t = multivariate_t.rvs(loc=np.zeros(len(self.tickers)), 
                                       shape=self.copula_corr.values, 
                                       df=4, size=n_sims)

        # 3. Transform to Returns
        sim_returns = np.zeros_like(sim_t)
        current_vols = np.array([self.volatility_map[t] for t in self.tickers])
        
        for i, asset in enumerate(self.tickers):
             # t-dist CDF -> Uniform
             from scipy.stats import t
             u_sim = t.cdf(sim_t[:, i], df=4)
             
             # Inverse ECDF (Empirical Quantile)
             sorted_res = np.sort(self.residuals[asset].values)
             # Basic quantile lookup
             sim_shocks = np.quantile(sorted_res, u_sim)
             
             sim_returns[:, i] = current_vols[i] * sim_shocks
             
        # 4. Optimize
        def objective(weights):
            port_ret = np.dot(sim_returns, weights)
            exp_ret = port_ret.mean()
            losses = -port_ret
            var = np.percentile(losses, 95)
            cvar = losses[losses > var].mean()
            if cvar == 0: return 1e6
            return -(exp_ret / cvar) # Min neg STARR

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.tickers)))
        init = np.ones(len(self.tickers))/len(self.tickers)
        
        res = minimize(objective, init, bounds=bounds, constraints=constraints, method='SLSQP')
        
        print(f"    [OPTIMAL WEIGHTS] STARR Ratio")
        for i, t in enumerate(self.tickers):
            print(f"    {t:<6}: {res.x[i]*100:.1f}%")
            
        return res.x, sim_returns

    def calculate_component_cvar(self, weights, sim_returns):
        print(f"\n[5b] Calculating Risk Attribution (Component CVaR)...")
        # 1. Calculate Portfolio Returns
        port_ret = np.dot(sim_returns, weights)
        losses = -port_ret
        
        # 2. Identify Tail Scenarios (Loss > VaR 95%)
        var_95 = np.percentile(losses, 95)
        tail_indices = losses > var_95
        
        # 3. Component CVaR = w_i * E[Loss_i | Tail]
        # This simplifies to: w_i * mean(-returns_i[tail_indices])
        cvar_contributions = {}
        total_cvar = 0
        
        for i, asset in enumerate(self.tickers):
            # Asset return in the tail scenarios
            tail_asset_rets = sim_returns[tail_indices, i]
            # Expected shortfall of asset i in these scenarios
            es_asset = -tail_asset_rets.mean()
            
            # Contribution
            contrib = weights[i] * es_asset
            cvar_contributions[asset] = contrib
            total_cvar += contrib
            
        print(f"    Portfolio CVaR (95%): {total_cvar:.2f}%")
        for asset, contrib in cvar_contributions.items():
            pct = (contrib / total_cvar) * 100 if total_cvar > 0 else 0
            print(f"    {asset:<6}: {contrib:.2f}% (Risk Share: {pct:.1f}%)")
            
        return cvar_contributions
