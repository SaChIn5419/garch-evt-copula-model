from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.copula_model import simulate_portfolio_copula_risk
from src.config import FORECAST_ALPHA, ROLLING_WINDOW
from src.evt import fit_evt_left_tail
from src.risk_metrics import (
    breach_flag,
    christoffersen_test,
    count_exception_clusters,
    cvar_breach_flag,
    kupiec_test,
    parametric_risk_forecast,
    summarize_breaches,
)
from src.strategy import dollar_neutral_vol_weights
from src.volatility import GJRGARCHVolatilityModel


@dataclass
class BacktestResult:
    asset_forecasts: pd.DataFrame
    portfolio_forecasts: pd.DataFrame
    portfolio_weights: pd.DataFrame
    summary: dict[str, float]


def _forecast_covariance(train_returns: pd.DataFrame, forecast_vols: pd.Series) -> np.ndarray:
    corr = np.array(train_returns.corr().fillna(0.0), dtype=float, copy=True)
    np.fill_diagonal(corr, 1.0)
    sigma = forecast_vols.to_numpy(dtype=float)
    cov = corr * np.outer(sigma, sigma)
    return cov


def run_walk_forward_backtest(
    returns: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    alpha: float = FORECAST_ALPHA,
    model: GJRGARCHVolatilityModel | None = None,
) -> BacktestResult:
    clean = returns.dropna(how="any").astype(float)
    if clean.shape[0] <= window:
        raise ValueError("Not enough observations for the requested walk-forward window.")

    model = model or GJRGARCHVolatilityModel()
    asset_rows: list[dict[str, float | str | int | bool]] = []
    portfolio_rows: list[dict[str, float | str | int]] = []
    weights_rows: list[pd.Series] = []

    for end_idx in range(window, len(clean)):
        train = clean.iloc[end_idx - window:end_idx]
        realized = clean.iloc[end_idx]
        forecasts = model.fit_many(train)

        mean_vector = np.array([forecasts[col].mean_forecast for col in clean.columns], dtype=float)
        vol_vector = pd.Series(
            {col: forecasts[col].volatility_forecast for col in clean.columns},
            index=clean.columns,
            dtype=float,
        )
        weights = dollar_neutral_vol_weights(vol_vector)
        standardized_matrix = np.column_stack([forecasts[col].standardized_residuals for col in clean.columns])
        portfolio_risk = simulate_portfolio_copula_risk(
            standardized_matrix,
            mean_vector,
            vol_vector.to_numpy(),
            weights.to_numpy(),
            alpha,
        )
        realized_portfolio = float(weights.to_numpy() @ realized.to_numpy())

        for col in clean.columns:
            forecast = forecasts[col]
            nu = forecast.params.get("nu")
            fallback_risk = parametric_risk_forecast(
                forecast.mean_forecast,
                forecast.variance_forecast,
                alpha,
                distribution=forecast.distribution,
                nu=float(nu) if nu is not None else None,
            )
            evt = fit_evt_left_tail(forecast.standardized_residuals, alpha)
            if evt.valid:
                asset_var = forecast.mean_forecast + forecast.volatility_forecast * float(evt.var_z)
                asset_cvar = forecast.mean_forecast + forecast.volatility_forecast * float(evt.cvar_z)
                risk_model = "evt"
            else:
                asset_var = fallback_risk.var
                asset_cvar = fallback_risk.cvar
                risk_model = "parametric"
            asset_rows.append(
                {
                    "date": clean.index[end_idx],
                    "asset": col,
                    "mean_forecast": forecast.mean_forecast,
                    "variance_forecast": forecast.variance_forecast,
                    "volatility_forecast": forecast.volatility_forecast,
                    "loglik": forecast.loglik,
                    "aic": forecast.aic,
                    "bic": forecast.bic,
                    "persistence": forecast.persistence,
                    "nu": np.nan if forecast.nu is None else forecast.nu,
                    "optimizer_nit": forecast.optimizer_nit,
                    "fallback_used": int(forecast.fallback_used),
                    "fit_message": forecast.message,
                    "risk_model": risk_model,
                    "evt_valid": int(evt.valid),
                    "evt_threshold": evt.threshold,
                    "evt_exceedance_rate": evt.exceedance_rate,
                    "evt_shape": np.nan if evt.shape is None else evt.shape,
                    "evt_scale": np.nan if evt.scale is None else evt.scale,
                    "evt_n_exceedances": evt.n_exceedances,
                    "evt_message": evt.message,
                    "var_forecast": asset_var,
                    "cvar_forecast": asset_cvar,
                    "realized_return": float(realized[col]),
                    "var_breach": breach_flag(float(realized[col]), asset_var),
                    "cvar_breach": cvar_breach_flag(float(realized[col]), asset_cvar),
                    "converged": int(forecast.converged),
                }
            )

        weights_row = weights.copy()
        weights_row.name = clean.index[end_idx]
        weights_rows.append(weights_row)

        portfolio_rows.append(
            {
                "date": clean.index[end_idx],
                "portfolio_return": realized_portfolio,
                "portfolio_mean_forecast": portfolio_risk.mean,
                "portfolio_variance_forecast": portfolio_risk.variance,
                "portfolio_volatility_forecast": portfolio_risk.volatility,
                "portfolio_var_forecast": portfolio_risk.var,
                "portfolio_cvar_forecast": portfolio_risk.cvar,
                "portfolio_risk_model": "studentt_copula_simulation",
                "portfolio_var_breach": breach_flag(realized_portfolio, portfolio_risk.var),
                "portfolio_cvar_breach": cvar_breach_flag(realized_portfolio, portfolio_risk.cvar),
            }
        )

    asset_df = pd.DataFrame(asset_rows)
    portfolio_df = pd.DataFrame(portfolio_rows).set_index("date")
    weights_df = pd.DataFrame(weights_rows)

    summary = summarize_breaches(portfolio_df["portfolio_var_breach"].to_numpy())
    summary["portfolio_exception_clusters"] = count_exception_clusters(portfolio_df["portfolio_var_breach"].to_numpy())
    summary.update(kupiec_test(portfolio_df["portfolio_var_breach"].to_numpy(), alpha))
    summary.update(christoffersen_test(portfolio_df["portfolio_var_breach"].to_numpy()))
    summary["portfolio_cvar_breach_rate"] = float(portfolio_df["portfolio_cvar_breach"].mean())
    summary["mean_absolute_net_exposure"] = float(weights_df.sum(axis=1).abs().mean())
    summary["mean_gross_exposure"] = float(weights_df.abs().sum(axis=1).mean())
    summary["asset_fit_convergence_rate"] = float(asset_df["converged"].mean())
    summary["asset_fit_fallback_rate"] = float(asset_df["fallback_used"].mean())
    summary["mean_persistence"] = float(asset_df["persistence"].mean())
    summary["mean_optimizer_iterations"] = float(asset_df["optimizer_nit"].mean())
    summary["evt_usage_rate"] = float(asset_df["evt_valid"].mean())

    return BacktestResult(
        asset_forecasts=asset_df,
        portfolio_forecasts=portfolio_df,
        portfolio_weights=weights_df,
        summary=summary,
    )
