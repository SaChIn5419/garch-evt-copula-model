from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm, t


@dataclass
class RiskForecast:
    alpha: float
    mean: float
    variance: float
    volatility: float
    var: float
    cvar: float


def _standardized_student_t_quantile(alpha: float, nu: float) -> float:
    return float(t.ppf(alpha, df=nu) / np.sqrt(nu / (nu - 2.0)))


def _standardized_student_t_cvar(alpha: float, nu: float) -> float:
    q = float(t.ppf(alpha, df=nu))
    pdf = float(t.pdf(q, df=nu))
    raw_tail_mean = -pdf * (nu + q * q) / ((nu - 1.0) * alpha)
    return raw_tail_mean / np.sqrt(nu / (nu - 2.0))


def parametric_risk_forecast(
    mean: float,
    variance: float,
    alpha: float,
    distribution: str = "normal",
    nu: float | None = None,
) -> RiskForecast:
    sigma = float(np.sqrt(max(variance, 1e-18)))
    if distribution == "studentt":
        if nu is None:
            raise ValueError("Student-t risk forecast requires nu.")
        q = _standardized_student_t_quantile(alpha, nu)
        tail_mean = _standardized_student_t_cvar(alpha, nu)
    else:
        q = float(norm.ppf(alpha))
        tail_mean = float(-norm.pdf(q) / alpha)

    var = mean + sigma * q
    cvar = mean + sigma * tail_mean
    return RiskForecast(alpha=alpha, mean=mean, variance=variance, volatility=sigma, var=var, cvar=cvar)


def portfolio_normal_risk_forecast(
    mean_vector: np.ndarray,
    covariance: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> RiskForecast:
    mu = float(weights @ mean_vector)
    variance = float(weights @ covariance @ weights)
    return parametric_risk_forecast(mu, variance, alpha, distribution="normal")


def simulation_risk_forecast(simulated_returns: np.ndarray, alpha: float) -> RiskForecast:
    sim = np.asarray(simulated_returns, dtype=float)
    var = float(np.quantile(sim, alpha))
    tail = sim[sim <= var]
    cvar = float(tail.mean()) if tail.size else var
    return RiskForecast(
        alpha=alpha,
        mean=float(sim.mean()),
        variance=float(sim.var(ddof=1)),
        volatility=float(sim.std(ddof=1)),
        var=var,
        cvar=cvar,
    )


def breach_flag(realized_return: float, var_level: float) -> int:
    return int(realized_return < var_level)


def cvar_breach_flag(realized_return: float, cvar_level: float) -> int:
    return int(realized_return < cvar_level)


def summarize_breaches(flags: np.ndarray) -> dict[str, float]:
    flags = np.asarray(flags, dtype=float)
    return {
        "n_obs": float(flags.size),
        "n_breaches": float(flags.sum()),
        "breach_rate": float(flags.mean()) if flags.size else np.nan,
    }


def count_exception_clusters(flags: np.ndarray) -> float:
    seq = np.asarray(flags, dtype=int).reshape(-1)
    if seq.size == 0:
        return np.nan
    clusters = int(np.sum((seq == 1) & np.concatenate(([True], seq[:-1] == 0))))
    return float(clusters)


def kupiec_test(flags: np.ndarray, alpha: float) -> dict[str, float]:
    seq = np.asarray(flags, dtype=int).reshape(-1)
    n_obs = seq.size
    n_breaches = int(seq.sum())
    if n_obs == 0 or alpha <= 0.0 or alpha >= 1.0:
        return {"kupiec_lr": np.nan, "kupiec_pvalue": np.nan}

    breach_rate = n_breaches / n_obs
    if breach_rate in (0.0, 1.0):
        return {"kupiec_lr": np.nan, "kupiec_pvalue": np.nan}

    log_l_null = (
        n_breaches * np.log(alpha)
        + (n_obs - n_breaches) * np.log(1.0 - alpha)
    )
    log_l_alt = (
        n_breaches * np.log(breach_rate)
        + (n_obs - n_breaches) * np.log(1.0 - breach_rate)
    )
    lr = -2.0 * (log_l_null - log_l_alt)
    return {"kupiec_lr": float(lr), "kupiec_pvalue": float(1.0 - chi2.cdf(lr, df=1))}


def christoffersen_test(flags: np.ndarray) -> dict[str, float]:
    seq = np.asarray(flags, dtype=int).reshape(-1)
    if seq.size < 2:
        return {
            "christoffersen_lr": np.nan,
            "christoffersen_pvalue": np.nan,
            "n00": np.nan,
            "n01": np.nan,
            "n10": np.nan,
            "n11": np.nan,
        }

    prev = seq[:-1]
    curr = seq[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _term(successes: int, failures: int, prob: float) -> float:
        if prob <= 0.0:
            return 0.0 if successes == 0 else -np.inf
        if prob >= 1.0:
            return 0.0 if failures == 0 else -np.inf
        return successes * np.log(prob) + failures * np.log(1.0 - prob)

    log_l_indep = _term(n01 + n11, n00 + n10, pi)
    log_l_markov = _term(n01, n00, pi0) + _term(n11, n10, pi1)
    if not np.isfinite(log_l_indep) or not np.isfinite(log_l_markov):
        lr = np.nan
        pvalue = np.nan
    else:
        lr = -2.0 * (log_l_indep - log_l_markov)
        pvalue = 1.0 - chi2.cdf(lr, df=1)

    return {
        "christoffersen_lr": float(lr) if np.isfinite(lr) else np.nan,
        "christoffersen_pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
        "n00": float(n00),
        "n01": float(n01),
        "n10": float(n10),
        "n11": float(n11),
    }
