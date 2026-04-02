from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROLLING_WINDOW
from src.data_loader import fetch_basket
from src.preprocessing import log_returns
from src.report import make_run_name
from src.volatility import ArchVolatilityModel, GJRGARCHVolatilityModel, VolatilityForecast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _lag1_autocorr(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    x0 = arr[:-1]
    x1 = arr[1:]
    if np.std(x0) == 0.0 or np.std(x1) == 0.0:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def _residual_sq_acf(std_resid: np.ndarray) -> float:
    z = np.asarray(std_resid, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 3:
        return np.nan
    return _lag1_autocorr(z * z)


def _extract_scalar_params(forecast: VolatilityForecast) -> dict[str, float]:
    params = forecast.params
    alpha = np.asarray(params.get("alpha", []), dtype=float)
    gamma = np.asarray(params.get("gamma", []), dtype=float)
    beta = np.asarray(params.get("beta", []), dtype=float)
    return {
        "mu": float(params.get("mu", np.nan)),
        "omega": float(params.get("omega", np.nan)),
        "alpha_sum": float(alpha.sum()) if alpha.size else 0.0,
        "gamma_sum": float(gamma.sum()) if gamma.size else 0.0,
        "beta_sum": float(beta.sum()) if beta.size else 0.0,
        "nu": np.nan if forecast.nu is None else float(forecast.nu),
    }


def _plot_gap_panel(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    assets = list(df["asset"].drop_duplicates())
    fig, axes = plt.subplots(len(assets), 1, figsize=(12, 3 * len(assets)), sharex=True)
    if len(assets) == 1:
        axes = [axes]

    for ax, asset in zip(axes, assets):
        subset = df[df["asset"] == asset]
        ax.plot(pd.to_datetime(subset["date"]), subset[metric], color="darkred", linewidth=1.2)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_title(str(asset))
        ax.grid(alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_summary(summary_df: pd.DataFrame, output_dir: Path, basket_name: str) -> None:
    summary_df.to_csv(output_dir / "engine_summary.csv", index=False)

    records = summary_df.where(pd.notna(summary_df), None).to_dict(orient="records")
    with open(output_dir / "engine_summary.json", "w", encoding="ascii") as f:
        json.dump(records, f, indent=2, allow_nan=False)

    lines = [f"# {basket_name} GJR Engine Comparison", ""]
    for _, row in summary_df.iterrows():
        lines.append(f"## {row['asset']}")
        lines.append("")
        for key, value in row.items():
            if key == "asset":
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")

    with open(output_dir / "engine_summary.md", "w", encoding="ascii") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def run_engine_comparison(
    basket_name: str = "india_primary",
    base_dir: str = "results_diagnostics",
    evaluation_days: int = 120,
) -> Path:
    prices = fetch_basket(basket_name)
    returns = log_returns(prices)
    keep_rows = min(len(returns), ROLLING_WINDOW + evaluation_days)
    returns = returns.tail(keep_rows).dropna(how="any").astype(float)

    run_name = make_run_name(
        f"{basket_name}_gjr_engine_compare",
        str(returns.index.min().date()),
        str(returns.index.max().date()),
    )
    output_dir = Path(base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "arch_gjr": ArchVolatilityModel(asymmetry=True),
        "custom_gjr": GJRGARCHVolatilityModel(),
    }

    rows: list[dict[str, float | str | int | pd.Timestamp]] = []
    for end_idx in range(ROLLING_WINDOW, len(returns)):
        train = returns.iloc[end_idx - ROLLING_WINDOW:end_idx]
        date = returns.index[end_idx]

        by_model: dict[str, dict[str, VolatilityForecast]] = {
            model_name: model.fit_many(train) for model_name, model in models.items()
        }

        for asset in returns.columns:
            arch_forecast = by_model["arch_gjr"][asset]
            custom_forecast = by_model["custom_gjr"][asset]

            arch_params = _extract_scalar_params(arch_forecast)
            custom_params = _extract_scalar_params(custom_forecast)

            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "arch_volatility_forecast": arch_forecast.volatility_forecast,
                    "custom_volatility_forecast": custom_forecast.volatility_forecast,
                    "vol_forecast_gap": custom_forecast.volatility_forecast - arch_forecast.volatility_forecast,
                    "vol_forecast_ratio": custom_forecast.volatility_forecast / arch_forecast.volatility_forecast,
                    "arch_persistence": arch_forecast.persistence,
                    "custom_persistence": custom_forecast.persistence,
                    "persistence_gap": custom_forecast.persistence - arch_forecast.persistence,
                    "arch_std_resid_sq_lag1_acf": _residual_sq_acf(arch_forecast.standardized_residuals),
                    "custom_std_resid_sq_lag1_acf": _residual_sq_acf(custom_forecast.standardized_residuals),
                    "std_resid_sq_acf_gap": _residual_sq_acf(custom_forecast.standardized_residuals)
                    - _residual_sq_acf(arch_forecast.standardized_residuals),
                    "arch_fallback_used": int(arch_forecast.fallback_used),
                    "custom_fallback_used": int(custom_forecast.fallback_used),
                    "arch_converged": int(arch_forecast.converged),
                    "custom_converged": int(custom_forecast.converged),
                    "arch_mu": arch_params["mu"],
                    "custom_mu": custom_params["mu"],
                    "mu_gap": custom_params["mu"] - arch_params["mu"],
                    "arch_omega": arch_params["omega"],
                    "custom_omega": custom_params["omega"],
                    "omega_gap": custom_params["omega"] - arch_params["omega"],
                    "arch_alpha_sum": arch_params["alpha_sum"],
                    "custom_alpha_sum": custom_params["alpha_sum"],
                    "alpha_gap": custom_params["alpha_sum"] - arch_params["alpha_sum"],
                    "arch_gamma_sum": arch_params["gamma_sum"],
                    "custom_gamma_sum": custom_params["gamma_sum"],
                    "gamma_gap": custom_params["gamma_sum"] - arch_params["gamma_sum"],
                    "arch_negative_gamma": int(arch_params["gamma_sum"] < 0.0),
                    "arch_beta_sum": arch_params["beta_sum"],
                    "custom_beta_sum": custom_params["beta_sum"],
                    "beta_gap": custom_params["beta_sum"] - arch_params["beta_sum"],
                    "arch_nu": arch_params["nu"],
                    "custom_nu": custom_params["nu"],
                    "nu_gap": custom_params["nu"] - arch_params["nu"],
                }
            )

    diagnostics_df = pd.DataFrame(rows)
    diagnostics_df.to_csv(output_dir / "engine_diagnostics.csv", index=False)

    summary_df = (
        diagnostics_df.groupby("asset", as_index=False)
        .agg(
            mean_arch_volatility_forecast=("arch_volatility_forecast", "mean"),
            mean_custom_volatility_forecast=("custom_volatility_forecast", "mean"),
            mean_vol_forecast_gap=("vol_forecast_gap", "mean"),
            mean_vol_forecast_ratio=("vol_forecast_ratio", "mean"),
            mean_arch_persistence=("arch_persistence", "mean"),
            mean_custom_persistence=("custom_persistence", "mean"),
            mean_persistence_gap=("persistence_gap", "mean"),
            mean_arch_std_resid_sq_lag1_acf=("arch_std_resid_sq_lag1_acf", "mean"),
            mean_custom_std_resid_sq_lag1_acf=("custom_std_resid_sq_lag1_acf", "mean"),
            mean_std_resid_sq_acf_gap=("std_resid_sq_acf_gap", "mean"),
            mean_alpha_gap=("alpha_gap", "mean"),
            mean_gamma_gap=("gamma_gap", "mean"),
            arch_negative_gamma_rate=("arch_negative_gamma", "mean"),
            mean_beta_gap=("beta_gap", "mean"),
            mean_nu_gap=("nu_gap", "mean"),
            arch_fallback_rate=("arch_fallback_used", "mean"),
            custom_fallback_rate=("custom_fallback_used", "mean"),
            arch_convergence_rate=("arch_converged", "mean"),
            custom_convergence_rate=("custom_converged", "mean"),
        )
        .sort_values("asset")
    )
    _write_summary(summary_df, output_dir, basket_name=basket_name)

    _plot_gap_panel(
        diagnostics_df,
        metric="vol_forecast_gap",
        output_path=output_dir / "vol_forecast_gap_panel.png",
        title=f"{basket_name}: Custom GJR Minus arch GJR Volatility Forecast",
    )
    _plot_gap_panel(
        diagnostics_df,
        metric="persistence_gap",
        output_path=output_dir / "persistence_gap_panel.png",
        title=f"{basket_name}: Custom GJR Minus arch GJR Persistence",
    )
    _plot_gap_panel(
        diagnostics_df,
        metric="std_resid_sq_acf_gap",
        output_path=output_dir / "residual_sq_acf_gap_panel.png",
        title=f"{basket_name}: Custom GJR Minus arch GJR Squared Residual Lag-1 ACF",
    )

    return output_dir


def main() -> None:
    output_dir = run_engine_comparison()
    print(f"Saved engine comparison to: {output_dir}")


if __name__ == "__main__":
    main()
