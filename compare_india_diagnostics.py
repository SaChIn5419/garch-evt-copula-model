from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROLLING_WINDOW
from src.data_loader import fetch_basket
from src.preprocessing import log_returns
from src.report import make_run_name
from src.volatility import ArchVolatilityModel, GJRGARCHVolatilityModel

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _lag1_autocorr(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size < 2:
        return np.nan
    x0 = arr[:-1]
    x1 = arr[1:]
    if np.std(x0) == 0.0 or np.std(x1) == 0.0:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def _residual_diagnostics(std_resid: np.ndarray) -> dict[str, float]:
    z = np.asarray(std_resid, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 3:
        return {
            "std_resid_mean": np.nan,
            "std_resid_variance": np.nan,
            "std_resid_abs_mean": np.nan,
            "std_resid_lag1_acf": np.nan,
            "std_resid_sq_lag1_acf": np.nan,
        }
    return {
        "std_resid_mean": float(np.mean(z)),
        "std_resid_variance": float(np.var(z, ddof=1)),
        "std_resid_abs_mean": float(np.mean(np.abs(z))),
        "std_resid_lag1_acf": _lag1_autocorr(z),
        "std_resid_sq_lag1_acf": _lag1_autocorr(z * z),
    }


def _plot_metric_panel(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    assets = list(df["asset"].drop_duplicates())
    fig, axes = plt.subplots(len(assets), 1, figsize=(12, 3 * len(assets)), sharex=True)
    if len(assets) == 1:
        axes = [axes]

    for ax, asset in zip(axes, assets):
        subset = df[df["asset"] == asset].copy()
        for model_name, color in [("garch_baseline", "steelblue"), ("gjr_custom", "darkorange")]:
            line = subset[subset["model"] == model_name]
            ax.plot(pd.to_datetime(line["date"]), line[metric], label=model_name, linewidth=1.2, color=color)
        ax.set_title(str(asset))
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    summary_df.to_csv(output_dir / "diagnostic_summary.csv", index=False)

    records = summary_df.where(pd.notna(summary_df), None).to_dict(orient="records")
    with open(output_dir / "diagnostic_summary.json", "w", encoding="ascii") as f:
        json.dump(records, f, indent=2, allow_nan=False)

    lines = ["# India Primary Volatility Diagnostics", ""]
    for _, row in summary_df.iterrows():
        lines.append(f"## {row['asset']} - {row['model']}")
        lines.append("")
        for key, value in row.items():
            if key in {"asset", "model"}:
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")

    with open(output_dir / "diagnostic_summary.md", "w", encoding="ascii") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def run_india_diagnostics(base_dir: str = "results_diagnostics", evaluation_days: int = 120) -> Path:
    prices = fetch_basket("india_primary")
    returns = log_returns(prices)
    keep_rows = min(len(returns), ROLLING_WINDOW + evaluation_days)
    returns = returns.tail(keep_rows).dropna(how="any").astype(float)

    run_name = make_run_name(
        "india_primary_diagnostics",
        str(returns.index.min().date()),
        str(returns.index.max().date()),
    )
    output_dir = Path(base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "garch_baseline": ArchVolatilityModel(asymmetry=False),
        "gjr_custom": GJRGARCHVolatilityModel(),
    }

    rows: list[dict[str, float | str | int]] = []
    for model_name, model in models.items():
        for end_idx in range(ROLLING_WINDOW, len(returns)):
            train = returns.iloc[end_idx - ROLLING_WINDOW:end_idx]
            forecasts = model.fit_many(train)
            date = returns.index[end_idx]
            for asset, forecast in forecasts.items():
                diagnostics = _residual_diagnostics(forecast.standardized_residuals)
                rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "model": model_name,
                        "volatility_forecast": forecast.volatility_forecast,
                        "variance_forecast": forecast.variance_forecast,
                        "persistence": forecast.persistence,
                        "fallback_used": int(forecast.fallback_used),
                        "converged": int(forecast.converged),
                        **diagnostics,
                    }
                )

    diagnostics_df = pd.DataFrame(rows)
    diagnostics_df.to_csv(output_dir / "diagnostics.csv", index=False)

    summary_df = (
        diagnostics_df.groupby(["asset", "model"], as_index=False)
        .agg(
            mean_volatility_forecast=("volatility_forecast", "mean"),
            mean_persistence=("persistence", "mean"),
            mean_std_resid_variance=("std_resid_variance", "mean"),
            mean_std_resid_abs_mean=("std_resid_abs_mean", "mean"),
            mean_std_resid_lag1_acf=("std_resid_lag1_acf", "mean"),
            mean_std_resid_sq_lag1_acf=("std_resid_sq_lag1_acf", "mean"),
            fallback_rate=("fallback_used", "mean"),
            convergence_rate=("converged", "mean"),
        )
        .sort_values(["asset", "model"])
    )
    _write_summary(summary_df, output_dir)

    _plot_metric_panel(
        diagnostics_df,
        metric="volatility_forecast",
        output_path=output_dir / "volatility_forecast_panel.png",
        title="India Primary: Forecast Volatility by Model",
    )
    _plot_metric_panel(
        diagnostics_df,
        metric="persistence",
        output_path=output_dir / "persistence_panel.png",
        title="India Primary: Persistence by Model",
    )
    _plot_metric_panel(
        diagnostics_df,
        metric="std_resid_sq_lag1_acf",
        output_path=output_dir / "residual_sq_acf_panel.png",
        title="India Primary: Standardized Residual Squared Lag-1 ACF",
    )
    _plot_metric_panel(
        diagnostics_df,
        metric="std_resid_variance",
        output_path=output_dir / "residual_variance_panel.png",
        title="India Primary: Standardized Residual Variance",
    )

    return output_dir


def main() -> None:
    output_dir = run_india_diagnostics()
    print(f"Saved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
