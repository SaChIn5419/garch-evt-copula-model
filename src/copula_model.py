from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_t, rankdata, t

from src.config import COPULA_DF, COPULA_SIMULATIONS


@dataclass
class CopulaPortfolioForecast:
    mean: float
    variance: float
    volatility: float
    var: float
    cvar: float
    simulations: np.ndarray
    correlation: np.ndarray
    df: int


def _pseudo_observations(z: np.ndarray) -> np.ndarray:
    u = np.empty_like(z, dtype=float)
    for i in range(z.shape[1]):
        u[:, i] = rankdata(z[:, i], method="average") / (z.shape[0] + 1.0)
    return np.clip(u, 1e-6, 1.0 - 1e-6)


def _student_t_latent_corr(u: np.ndarray, df: int) -> np.ndarray:
    latent = t.ppf(u, df=df)
    if latent.ndim != 2:
        latent = np.atleast_2d(latent)
    std = latent.std(axis=0, ddof=1)
    if np.all(std > 0.0):
        corr = np.corrcoef(latent, rowvar=False)
    else:
        corr = np.eye(latent.shape[1], dtype=float)
        valid = std > 0.0
        if np.sum(valid) >= 2:
            sub_corr = np.corrcoef(latent[:, valid], rowvar=False)
            corr[np.ix_(valid, valid)] = np.atleast_2d(sub_corr)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = 0.5 * (corr + corr.T)
    corr += np.eye(corr.shape[0]) * 1e-6
    np.fill_diagonal(corr, 1.0)
    return corr


def _empirical_inverse_cdf(sample: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.quantile(sample, u, method="linear")


def simulate_portfolio_copula_risk(
    standardized_residual_matrix: np.ndarray,
    mean_vector: np.ndarray,
    volatility_vector: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    df: int = COPULA_DF,
    n_sims: int = COPULA_SIMULATIONS,
    seed: int = 123,
) -> CopulaPortfolioForecast:
    z = np.asarray(standardized_residual_matrix, dtype=float)
    mean_vector = np.asarray(mean_vector, dtype=float)
    volatility_vector = np.asarray(volatility_vector, dtype=float)
    weights = np.asarray(weights, dtype=float)

    u = _pseudo_observations(z)
    corr = _student_t_latent_corr(u, df=df)
    latent = multivariate_t.rvs(
        loc=np.zeros(z.shape[1]),
        shape=corr,
        df=df,
        size=n_sims,
        random_state=seed,
    )
    latent = np.atleast_2d(latent)
    sim_u = t.cdf(latent, df=df)
    sim_u = np.clip(sim_u, 1e-6, 1.0 - 1e-6)

    sim_z = np.empty_like(sim_u)
    for i in range(z.shape[1]):
        sim_z[:, i] = _empirical_inverse_cdf(z[:, i], sim_u[:, i])

    sim_returns = mean_vector + sim_z * volatility_vector
    portfolio_sim = sim_returns @ weights
    var = float(np.quantile(portfolio_sim, alpha))
    tail = portfolio_sim[portfolio_sim <= var]
    cvar = float(tail.mean()) if tail.size else var

    return CopulaPortfolioForecast(
        mean=float(portfolio_sim.mean()),
        variance=float(portfolio_sim.var(ddof=1)),
        volatility=float(portfolio_sim.std(ddof=1)),
        var=var,
        cvar=cvar,
        simulations=portfolio_sim,
        correlation=corr,
        df=df,
    )
