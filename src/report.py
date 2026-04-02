from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import BacktestResult
from src.config import RESULTS_DIR


def _extract_extreme_tail_events(portfolio: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    extreme = portfolio.copy().sort_values("portfolio_return").head(top_n)
    extreme = extreme.reset_index().rename(columns={"index": "date"})
    return extreme


def _format_summary(summary: dict[str, float], extreme_tail_events: pd.DataFrame) -> str:
    lines = ["# Backtest Summary", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    if not extreme_tail_events.empty:
        lines.extend(["", "## Extreme Tail Events", ""])
        for _, row in extreme_tail_events.iterrows():
            lines.append(
                "- "
                f"{row['date']}: return={row['portfolio_return']}, "
                f"var={row['portfolio_var_forecast']}, "
                f"cvar={row['portfolio_cvar_forecast']}"
            )
    return "\n".join(lines) + "\n"


def _json_safe_summary(summary: dict[str, float]) -> dict[str, float | None]:
    safe: dict[str, float | None] = {}
    for key, value in summary.items():
        numeric = float(value)
        safe[key] = numeric if pd.notna(numeric) and np.isfinite(numeric) else None
    return safe


def _plot_portfolio_risk(portfolio: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(portfolio.index, portfolio["portfolio_return"], label="Realized Return", color="black", linewidth=1.2)
    ax.plot(portfolio.index, portfolio["portfolio_var_forecast"], label="VaR Forecast", color="crimson", linestyle="--")
    ax.plot(portfolio.index, portfolio["portfolio_cvar_forecast"], label="CVaR Forecast", color="darkorange", linestyle=":")
    ax.set_title("Portfolio Return vs Forecast Risk")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "portfolio_risk.png")
    plt.close(fig)


def _plot_portfolio_weights(weights: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in weights.columns:
        ax.plot(weights.index, weights[col], label=str(col), linewidth=1.0)
    ax.set_title("Portfolio Weights")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "portfolio_weights.png")
    plt.close(fig)


def _plot_copula_diagnostics(portfolio: pd.DataFrame, output_dir: Path) -> None:
    required = {
        "copula_avg_abs_correlation",
        "copula_max_abs_correlation",
        "copula_lower_tail_dependence",
    }
    if not required.issubset(portfolio.columns):
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    metrics = [
        ("copula_avg_abs_correlation", "Average Absolute Copula Correlation", "steelblue"),
        ("copula_max_abs_correlation", "Maximum Absolute Copula Correlation", "darkorange"),
        ("copula_lower_tail_dependence", "Empirical Lower Tail Co-Exceedance", "crimson"),
    ]
    for ax, (col, title, color) in zip(axes, metrics):
        ax.plot(portfolio.index, portfolio[col], color=color, linewidth=1.2)
        ax.set_title(title)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "copula_diagnostics.png")
    plt.close(fig)


def save_backtest_results(
    result: BacktestResult,
    run_name: str,
    base_dir: str = RESULTS_DIR,
) -> Path:
    output_dir = Path(base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    result.asset_forecasts.to_csv(output_dir / "asset_forecasts.csv", index=False)
    result.portfolio_forecasts.to_csv(output_dir / "portfolio_forecasts.csv")
    result.portfolio_weights.to_csv(output_dir / "portfolio_weights.csv")
    extreme_tail_events = _extract_extreme_tail_events(result.portfolio_forecasts)
    extreme_tail_events.to_csv(output_dir / "extreme_tail_events.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="ascii") as f:
        json.dump(_json_safe_summary(result.summary), f, indent=2, allow_nan=False)

    with open(output_dir / "summary.md", "w", encoding="ascii") as f:
        f.write(_format_summary(result.summary, extreme_tail_events))

    if not result.portfolio_forecasts.empty:
        _plot_portfolio_risk(result.portfolio_forecasts, output_dir)
        _plot_copula_diagnostics(result.portfolio_forecasts, output_dir)
    if not result.portfolio_weights.empty:
        _plot_portfolio_weights(result.portfolio_weights, output_dir)

    return output_dir


def make_run_name(basket_name: str, start_date: str, end_date: str) -> str:
    safe_basket = basket_name.replace("/", "_")
    return f"{safe_basket}_{start_date}_{end_date}"
