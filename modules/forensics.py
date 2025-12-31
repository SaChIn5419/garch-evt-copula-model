import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from arch import arch_model
import os

def plot_gold_betrayal(returns, output_dir):
    print("\n[Plot 1] Generating 'Gold Betrayal' Rolling Correlation...")
    
    # Rolling Correlation (GLD vs JPM)
    window = 22
    if 'GLD' not in returns.columns or 'JPM' not in returns.columns:
        print("    [SKIP] Missing GLD or JPM for Gold Betrayal plot.")
        return

    rolling_corr = returns['GLD'].rolling(window=window).corr(returns['JPM'])
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=rolling_corr, color='crimson', linewidth=2.5, label='GLD-JPM Correlation')
    
    # Highlight Crash
    plt.axvspan('2020-03-01', '2020-03-31', color='grey', alpha=0.3, label='Liquidity Black Hole')
    
    plt.title("The 'Gold Betrayal': Safe Haven Failure (March 2020)", fontsize=16, fontweight='bold')
    plt.ylabel("Correlation (22-Day Rolling)")
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    
    filename = os.path.join(output_dir, "1_gold_betrayal.png")
    plt.savefig(filename)
    print(f"    Saved: {filename}")
    plt.close()

def plot_volatility_cone(returns, output_dir):
    asset = 'JPM' # Epicenter of banking stress
    if asset not in returns.columns: return

    print(f"\n[Plot 2] Generating Volatility Cone for {asset}...")
    
    r_asset = returns[asset]
    
    # 1. Historical Vol (30-day rolling std dev)
    hist_vol = r_asset.rolling(window=30).std()
    
    # 2. GARCH Vol
    am = arch_model(r_asset, vol='Garch', p=1, q=1, dist='t', rescale=False)
    res = am.fit(disp='off')
    garch_vol = res.conditional_volatility
    
    # Plot from Feb 2020
    subset = r_asset.loc["2020-01-01":]
    h_vol_sub = hist_vol.loc["2020-01-01":]
    g_vol_sub = garch_vol.loc["2020-01-01":]
    
    plt.figure(figsize=(14, 7))
    plt.plot(subset.index, subset, color='gray', alpha=0.3, label=f'{asset} Daily Returns')
    
    # Risk Cones (2 Sigma)
    plt.plot(h_vol_sub.index, 2*h_vol_sub, color='blue', linestyle='--', label='Standard Hist Vol (2σ)')
    plt.plot(h_vol_sub.index, -2*h_vol_sub, color='blue', linestyle='--')
    
    plt.plot(g_vol_sub.index, 2*g_vol_sub, color='red', linewidth=2, label='GARCH Predicted Vol (2σ)')
    plt.plot(g_vol_sub.index, -2*g_vol_sub, color='red', linewidth=2)
    
    plt.title(f"The 'Ghost in the Machine': GARCH vs. Historical Vol ({asset})", fontsize=16)
    plt.legend()
    
    filename = os.path.join(output_dir, "2_volatility_cone.png")
    plt.savefig(filename)
    print(f"    Saved: {filename}")
    plt.close()

def plot_tail_dependence(returns, output_dir):
    print(f"\n[Plot 3] Generating Tail Dependence Scatter (DAL vs SPG)...")
    if 'DAL' not in returns.columns or 'SPG' not in returns.columns: return
    
    x = returns['DAL']
    y = returns['SPG']
    
    plt.figure(figsize=(10, 10))
    
    # 1. Scatter Points
    mask_crash = (returns.index >= '2020-03-01') & (returns.index <= '2020-03-31')
    
    plt.scatter(x[~mask_crash], y[~mask_crash], alpha=0.3, color='gray', label='Normal Regime')
    plt.scatter(x[mask_crash], y[mask_crash], color='red', s=100, marker='x', label='March 2020 (The Crash)', zorder=5)
    
    # 2. Add theoretical Normal Contour (Circle)
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    
    from matplotlib.patches import Ellipse
    ax = plt.gca()
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=lambda_[0]*2*2, height=lambda_[1]*2*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])),
                  edgecolor='blue', fc='None', lw=2, linestyle='--', label='Normal Gaussian Contour (2σ)')
    ax.add_patch(ell)
    
    plt.title("Tail Dependence: Where models go to die", fontsize=16)
    plt.xlabel("Delta Airlines Returns (%)")
    plt.ylabel("Simon Property Group Returns (%)")
    plt.legend()
    
    filename = os.path.join(output_dir, "3_tail_dependence.png")
    plt.savefig(filename)
    print(f"    Saved: {filename}")
    plt.close()

def plot_var_breach(returns, var_normal, var_evt, output_dir):
    print("\n[Plot 4] Generating VaR Breach Chart (The Money Shot)...")
    
    # Equal Weighted Portfolio Returns for plotting
    weights = np.array([1/len(returns.columns)]*len(returns.columns))
    port_ret = returns.dot(weights)
    
    # Plot Zoom March 2020
    subset = port_ret.loc['2020-01-01':'2020-04-15']
    
    # Reindex passed VaR series to match subset
    vn = var_normal.reindex(subset.index)
    ve = var_evt.reindex(subset.index)
    
    plt.figure(figsize=(15, 7))
    
    # PnL Dots logic
    colors = []
    sizes = []
    
    for dt, r in subset.items():
        limit_n = vn.loc[dt]
        limit_e = ve.loc[dt]
        
        # Handle NaNs from rolling window or mismatch
        if pd.isna(limit_n) or pd.isna(limit_e):
            colors.append('gray')
            sizes.append(30)
            continue

        if r < limit_e: # Failed BOTH
            colors.append('black')
            sizes.append(100)
        elif r < limit_n: # Failed Normal ONLY (The Alpha)
            colors.append('red') 
            sizes.append(100)
        else:
            colors.append('green')
            sizes.append(30)
            
    plt.scatter(subset.index, subset, c=colors, s=sizes, alpha=0.7, label='Daily PnL')
    
    plt.plot(vn.index, vn, color='red', linestyle='--', linewidth=2, label='Standard Normal VaR (99%)')
    plt.plot(ve.index, ve, color='black', linewidth=2, label='GARCH-EVT VaR (99%)')
    
    # Annotate Black Thursday
    plt.annotate('Black Thursday (-10%)', xy=(pd.Timestamp('2020-03-12'), -10), 
                 xytext=(pd.Timestamp('2020-02-15'), -15),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
    
    plt.title("Risk Management Alpha: Preventing Ruin in March 2020", fontsize=16)
    plt.ylabel("Portfolio Return (%)")
    plt.legend()
    
    filename = os.path.join(output_dir, "4_var_breach.png")
    plt.savefig(filename)
    print(f"    Saved: {filename}")
    plt.close()

def plot_topology(returns, output_dir):
    print("\n[Plot 5] Generating Before & After Network Topology...")
    
    def build_graph(data_slice, title, ax):
        corr = data_slice.corr()
        G = nx.Graph()
        
        # Add edges
        for i in corr.columns:
            for j in corr.columns:
                if i != j:
                    w = corr.loc[i, j]
                    if w > 0.3: # Filter weak
                        G.add_edge(i, j, weight=w)
        
        pos = nx.spring_layout(G, seed=42)
        
        colors = ['cyan' if node == 'ZM' else 'gold' if node=='GLD' else '#ff9999' for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1500, ax=ax, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_weight='bold')
        
        weights = [G[u][v]['weight']*3 for u,v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', ax=ax)
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Calm Period (Jan 2020)
    calm_ret = returns.loc['2020-01-01':'2020-01-31']
    build_graph(calm_ret, "Jan 2020: Disconnected (Diversified)", ax1)
    
    # 2. Crash Period (March 2020)
    crash_ret = returns.loc['2020-03-01':'2020-03-31']
    build_graph(crash_ret, "Mar 2020: The Liquidity Black Hole", ax2)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, "5_topology.png")
    plt.savefig(filename)
    print(f"    Saved: {filename}")
    plt.close()

def plot_risk_attribution(cvar_dict, output_dir):
    print("\n[Plot 6] Generating Risk Attribution (Why the Optimizer Chose ZM)...")
    
    # Sort by contribution
    sorted_items = sorted(cvar_dict.items(), key=lambda x: x[1], reverse=True)
    assets = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    
    plt.figure(figsize=(10, 6))
    
    # Color logic: Red for high risk, Green for low/negative (Hedge)
    colors = ['crimson' if v > 0 else 'forestgreen' for v in values]
    
    bars = plt.bar(assets, values, color=colors, alpha=0.8, edgecolor='black')
    
    plt.title("Risk Attribution: Component CVaR Contribution", fontsize=16, fontweight='bold')
    plt.xlabel("Asset")
    plt.ylabel("Contribution to Portfolio CVaR (%)")
    plt.axhline(0, color='black', linewidth=1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%',
                 ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')

    plt.tight_layout()
    filename = os.path.join(output_dir, "6_risk_attribution.png")
    plt.savefig(filename)
    print(f"    Saved: {filename}")
    plt.close()
