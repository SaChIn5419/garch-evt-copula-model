from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gjrgarch_fast import GJRGARCH
from src.config import MODEL_DIST, MODEL_MAXITER, MODEL_P, MODEL_Q, MODEL_TOL


@dataclass
class VolatilityForecast:
    asset: str
    mean_forecast: float
    variance_forecast: float
    volatility_forecast: float
    standardized_residuals: np.ndarray
    residuals: np.ndarray
    conditional_variance: np.ndarray
    converged: bool
    message: str
    distribution: str
    loglik: float
    aic: float
    bic: float
    persistence: float
    nu: float | None
    optimizer_nit: int
    fallback_used: bool
    params: dict[str, Any]


class GJRGARCHVolatilityModel:
    """Thin wrapper that makes the custom engine easier to consume in the backtest."""

    def __init__(
        self,
        p: int = MODEL_P,
        q: int = MODEL_Q,
        dist: str = MODEL_DIST,
        maxiter: int = MODEL_MAXITER,
        tol: float = MODEL_TOL,
    ) -> None:
        self.p = p
        self.q = q
        self.dist = dist
        self.maxiter = maxiter
        self.tol = tol
        self._warm_starts: dict[str, np.ndarray] = {}

    def _fallback_forecast(self, clean: pd.Series, asset_name: str, message: str) -> VolatilityForecast:
        residuals = clean.to_numpy(dtype=float) - float(clean.mean())
        variance = float(np.var(residuals, ddof=1)) if clean.size > 1 else 0.0
        variance = max(variance, 1e-8)
        volatility = float(np.sqrt(variance))
        standardized = residuals / volatility
        return VolatilityForecast(
            asset=asset_name,
            mean_forecast=float(clean.mean()),
            variance_forecast=variance,
            volatility_forecast=volatility,
            standardized_residuals=standardized,
            residuals=residuals,
            conditional_variance=np.full(clean.shape[0], variance, dtype=float),
            converged=False,
            message=message,
            distribution="normal",
            loglik=np.nan,
            aic=np.nan,
            bic=np.nan,
            persistence=0.0,
            nu=None,
            optimizer_nit=0,
            fallback_used=True,
            params={
                "mu": float(clean.mean()),
                "omega": variance,
                "alpha": np.zeros(self.q, dtype=float),
                "gamma": np.zeros(self.q, dtype=float),
                "beta": np.zeros(self.p, dtype=float),
                "dist": "normal",
                "p": self.p,
                "q": self.q,
            },
        )

    @staticmethod
    def _diagnose(result: Any) -> tuple[float, float | None]:
        alpha = np.asarray(result.params["alpha"], dtype=float)
        gamma = np.asarray(result.params["gamma"], dtype=float)
        beta = np.asarray(result.params["beta"], dtype=float)
        persistence = float(alpha.sum() + 0.5 * gamma.sum() + beta.sum())
        nu = result.params.get("nu")
        return persistence, (float(nu) if nu is not None else None)

    def fit_one(self, series: pd.Series) -> VolatilityForecast:
        clean = pd.Series(series).dropna().astype(float)
        if clean.size < 25:
            raise ValueError("Need at least 25 observations to fit the GJR-GARCH model.")

        asset_name = str(clean.name) if clean.name is not None else "asset"
        model = GJRGARCH(p=self.p, q=self.q, dist=self.dist)
        x0 = self._warm_starts.get(asset_name)
        try:
            result = model.fit(clean.to_numpy(), x0=x0, maxiter=self.maxiter, tol=self.tol)
            self._warm_starts[asset_name] = result.optimizer_x.copy()
            variance_forecast = float(model.forecast(steps=1)[0])
            volatility_forecast = float(np.sqrt(variance_forecast))
            std_resid = result.residuals / np.sqrt(result.conditional_variance)
            persistence, nu = self._diagnose(result)
        except Exception as exc:
            return self._fallback_forecast(clean, asset_name, f"fallback:model_fit_error:{exc}")

        invalid = (
            (not result.converged)
            or (not np.isfinite(variance_forecast))
            or variance_forecast <= 0.0
            or (not np.all(np.isfinite(std_resid)))
            or (not np.all(np.isfinite(result.conditional_variance)))
        )
        if invalid:
            return self._fallback_forecast(clean, asset_name, f"fallback:invalid_gjr_fit:{result.message}")

        return VolatilityForecast(
            asset=asset_name,
            mean_forecast=float(result.params["mu"]),
            variance_forecast=variance_forecast,
            volatility_forecast=volatility_forecast,
            standardized_residuals=std_resid,
            residuals=result.residuals,
            conditional_variance=result.conditional_variance,
            converged=result.converged,
            message=result.message,
            distribution=self.dist,
            loglik=float(result.loglik),
            aic=float(result.aic),
            bic=float(result.bic),
            persistence=persistence,
            nu=nu,
            optimizer_nit=int(result.optimizer_nit),
            fallback_used=False,
            params=result.params,
        )

    def fit_many(self, returns: pd.DataFrame) -> dict[str, VolatilityForecast]:
        forecasts: dict[str, VolatilityForecast] = {}
        for col in returns.columns:
            forecasts[str(col)] = self.fit_one(returns[col])
        return forecasts
