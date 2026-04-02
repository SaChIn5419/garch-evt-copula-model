
"""
gjrgarch_fast.py

A fast, self-contained GJR-GARCH toolkit built around:
- a Numba-compiled likelihood/recursion core
- unconstrained parameterization with automatic stationarity enforcement
- MLE fit, forecasting, simulation, and basic diagnostics

This is a solid v1 for a custom Python package.
It is written to be easy to expand to richer distributions, higher orders, and risk tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Tuple

import math
import numpy as np

try:
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = None  # type: ignore
    prange = range  # type: ignore

try:
    from scipy.optimize import minimize
except Exception as exc:  # pragma: no cover
    raise ImportError("scipy is required for optimization") from exc


DistName = Literal["normal", "studentt"]


# ---------------------------------------------------------------------
# Small numerically stable helpers
# ---------------------------------------------------------------------

def _softplus(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x)
    return np.where(x > 30.0, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _ensure_1d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains NaN or infinite values.")
    if arr.size < 10:
        raise ValueError("Series is too short for stable estimation.")
    return arr


def _student_t_logpdf_standardized(z: np.ndarray, nu: float) -> np.ndarray:
    """
    Standardized Student-t with unit variance.
    nu > 2.0.
    """
    c = (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log((nu - 2.0) * math.pi)
    )
    return c - 0.5 * (nu + 1.0) * np.log1p((z * z) / (nu - 2.0))


# ---------------------------------------------------------------------
# Numba core
# ---------------------------------------------------------------------

if njit is not None:
    @njit(cache=True)
    def _persistence(alpha, gamma, beta):
        out = 0.0
        for i in range(alpha.shape[0]):
            out += alpha[i] + 0.5 * gamma[i]
        for i in range(beta.shape[0]):
            out += beta[i]
        return out

    @njit(cache=True)
    def _initial_variance(eps, omega, alpha, gamma, beta):
        persistence = _persistence(alpha, gamma, beta)
        uncond = omega / max(1e-6, 1.0 - persistence)
        if uncond <= 1e-12 or not np.isfinite(uncond):
            uncond = 1e-6

        m = min(eps.shape[0], 75)
        weighted = 0.0
        weight_sum = 0.0
        decay = 0.94
        weight = 1.0
        for i in range(m):
            e2 = eps[i] * eps[i]
            weighted += weight * e2
            weight_sum += weight
            weight *= decay

        backcast = weighted / weight_sum if weight_sum > 0.0 else uncond
        if backcast <= 1e-12 or not np.isfinite(backcast):
            backcast = uncond

        s0 = 0.5 * uncond + 0.5 * backcast
        if s0 <= 1e-12 or not np.isfinite(s0):
            s0 = uncond
        if s0 <= 1e-12 or not np.isfinite(s0):
            s0 = 1e-6
        return s0

    @njit(cache=True)
    def _pack_stationary(raw_omega, raw_alpha, raw_gamma, raw_beta, raw_scale):
        """
        Map unconstrained raw parameters to:
          omega > 0
          alpha_j >= 0
          gamma_j >= 0
          beta_j  >= 0
          sum(alpha) + 0.5 sum(gamma) + sum(beta) < 1
        """
        omega = np.log1p(np.exp(raw_omega)) + 1e-12

        q = raw_alpha.shape[0]
        p = raw_beta.shape[0]

        alpha = np.empty(q, dtype=np.float64)
        gamma = np.empty(q, dtype=np.float64)
        beta = np.empty(p, dtype=np.float64)

        sa = 0.0
        sg = 0.0
        sb = 0.0

        for i in range(q):
            alpha[i] = np.log1p(np.exp(raw_alpha[i])) + 1e-12
            gamma[i] = np.log1p(np.exp(raw_gamma[i])) + 1e-12
            sa += alpha[i]
            sg += gamma[i]

        for i in range(p):
            beta[i] = np.log1p(np.exp(raw_beta[i])) + 1e-12
            sb += beta[i]

        # leverage term is counted at half-weight for persistence under symmetric innovations
        denom = sa + 0.5 * sg + sb + 1e-12
        cap = 0.999 * (1.0 / (1.0 + np.exp(-raw_scale)))
        scale = cap / denom

        for i in range(q):
            alpha[i] *= scale
            gamma[i] *= scale
        for i in range(p):
            beta[i] *= scale

        return omega, alpha, gamma, beta

    @njit(cache=True)
    def _gjr_recursion(returns, mu, omega, alpha, gamma, beta):
        n = returns.shape[0]
        q = alpha.shape[0]
        p = beta.shape[0]

        eps = np.empty(n, dtype=np.float64)
        sig2 = np.empty(n, dtype=np.float64)

        for i in range(n):
            eps[i] = returns[i] - mu
        sig2[0] = _initial_variance(eps, omega, alpha, gamma, beta)

        for t in range(1, n):
            v = omega
            for j in range(q):
                idx = t - 1 - j
                if idx >= 0:
                    e2 = eps[idx] * eps[idx]
                    v += alpha[j] * e2
                    if eps[idx] < 0.0:
                        v += gamma[j] * e2
            for j in range(p):
                idx = t - 1 - j
                if idx >= 0:
                    v += beta[j] * sig2[idx]
            if v < 1e-18:
                v = 1e-18
            sig2[t] = v

        return eps, sig2

    @njit(cache=True)
    def _normal_nll(eps, sig2):
        nll = 0.0
        for i in range(eps.shape[0]):
            s2 = sig2[i]
            z2 = eps[i] * eps[i] / s2
            nll += 0.5 * (math.log(2.0 * math.pi) + math.log(s2) + z2)
        return nll

    @njit(cache=True)
    def _student_t_nll(eps, sig2, nu):
        nll = 0.0
        c = (
            math.lgamma((nu + 1.0) / 2.0)
            - math.lgamma(nu / 2.0)
            - 0.5 * math.log((nu - 2.0) * math.pi)
        )
        for i in range(eps.shape[0]):
            s2 = sig2[i]
            z = eps[i] / math.sqrt(s2)
            nll -= c - 0.5 * (nu + 1.0) * math.log1p((z * z) / (nu - 2.0)) - 0.5 * math.log(s2)
        return nll

    @njit(cache=True)
    def _forecast_path(omega, alpha, gamma, beta, eps_hist, sig2_hist, steps):
        """
        Recursive multi-step forecast.
        For the first step, the latest residual sign is known and is used exactly.
        For later steps, future shocks use E[I(e<0)] = 0.5 under symmetry.
        """
        q = alpha.shape[0]
        p = beta.shape[0]

        eps2_buf = np.empty(q if q > 0 else 1, dtype=np.float64)
        sig2_buf = np.empty(p if p > 0 else 1, dtype=np.float64)

        # Fill with the latest available histories
        for i in range(q):
            idx = eps_hist.shape[0] - 1 - i
            eps2_buf[i] = eps_hist[idx] * eps_hist[idx]
        for i in range(p):
            idx = sig2_hist.shape[0] - 1 - i
            sig2_buf[i] = sig2_hist[idx]

        out = np.empty(steps, dtype=np.float64)

        for h in range(steps):
            v = omega
            for j in range(q):
                v += alpha[j] * eps2_buf[j]
                if h == 0:
                    if eps_hist[eps_hist.shape[0] - 1 - j] < 0.0:
                        v += gamma[j] * eps2_buf[j]
                else:
                    v += 0.5 * gamma[j] * eps2_buf[j]
            for j in range(p):
                v += beta[j] * sig2_buf[j]
            if v < 1e-18:
                v = 1e-18
            out[h] = v

            # roll buffers forward using E[eps^2] = forecast variance
            if q > 0:
                for j in range(q - 1, 0, -1):
                    eps2_buf[j] = eps2_buf[j - 1]
                eps2_buf[0] = v
            if p > 0:
                for j in range(p - 1, 0, -1):
                    sig2_buf[j] = sig2_buf[j - 1]
                sig2_buf[0] = v

        return out

    @njit(cache=True)
    def _simulate_path(n, burn, mu, omega, alpha, gamma, beta, dist_code, nu, seed):
        np.random.seed(seed)
        total = n + burn
        q = alpha.shape[0]
        p = beta.shape[0]

        ret = np.empty(total, dtype=np.float64)
        eps = np.empty(total, dtype=np.float64)
        sig2 = np.empty(total, dtype=np.float64)

        s0 = _initial_variance(np.zeros(min(total, 75), dtype=np.float64), omega, alpha, gamma, beta)

        for t in range(total):
            if t == 0:
                sig2[t] = s0
            else:
                v = omega
                for j in range(q):
                    idx = t - 1 - j
                    if idx >= 0:
                        e2 = eps[idx] * eps[idx]
                        v += alpha[j] * e2
                        if eps[idx] < 0.0:
                            v += gamma[j] * e2
                for j in range(p):
                    idx = t - 1 - j
                    if idx >= 0:
                        v += beta[j] * sig2[idx]
                if v < 1e-18:
                    v = 1e-18
                sig2[t] = v

            z = np.random.standard_normal()
            if dist_code == 1:
                # Student-t with variance 1
                x = np.random.standard_t(nu)
                z = x / math.sqrt(nu / (nu - 2.0))
            eps[t] = math.sqrt(sig2[t]) * z
            ret[t] = mu + eps[t]

        return ret[burn:], sig2[burn:], eps[burn:]

else:  # pragma: no cover
    def _pack_stationary(*args, **kwargs):
        raise RuntimeError("Numba is required for this module.")
    def _gjr_recursion(*args, **kwargs):
        raise RuntimeError("Numba is required for this module.")
    def _normal_nll(*args, **kwargs):
        raise RuntimeError("Numba is required for this module.")
    def _student_t_nll(*args, **kwargs):
        raise RuntimeError("Numba is required for this module.")
    def _forecast_path(*args, **kwargs):
        raise RuntimeError("Numba is required for this module.")
    def _simulate_path(*args, **kwargs):
        raise RuntimeError("Numba is required for this module.")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

@dataclass
class GJRGARCHResult:
    params: Dict[str, Any]
    loglik: float
    aic: float
    bic: float
    converged: bool
    message: str
    nobs: int
    residuals: np.ndarray
    conditional_variance: np.ndarray
    optimizer_x: np.ndarray
    optimizer_fun: float
    optimizer_nit: int

    @property
    def conditional_volatility(self) -> np.ndarray:
        return np.sqrt(self.conditional_variance)


class GJRGARCH:
    """
    Fast GJR-GARCH(p, q) estimator.

    Parameters
    ----------
    p : int
        Number of GARCH lags.
    q : int
        Number of ARCH/leverage lags.
    dist : {"normal", "studentt"}
        Innovation distribution.
    """

    def __init__(self, p: int = 1, q: int = 1, dist: DistName = "normal"):
        if p < 1 or q < 1:
            raise ValueError("p and q must both be >= 1.")
        if dist not in ("normal", "studentt"):
            raise ValueError("dist must be 'normal' or 'studentt'.")
        self.p = int(p)
        self.q = int(q)
        self.dist = dist
        self.result_: Optional[GJRGARCHResult] = None

    @staticmethod
    def _default_start(theta_y: np.ndarray, p: int, q: int, dist: DistName) -> np.ndarray:
        mu0 = float(np.mean(theta_y))
        var0 = float(np.var(theta_y, ddof=1))
        if not np.isfinite(var0) or var0 <= 0:
            var0 = 1e-4

        # Raw unconstrained vector:
        # [mu, raw_omega, raw_alpha(q), raw_gamma(q), raw_beta(p), raw_scale, raw_nu?]
        raw = [mu0, math.log(math.exp(0.1 * var0) - 1.0 + 1e-12)]
        raw += [math.log(math.exp(0.08) - 1.0 + 1e-12)] * q
        raw += [math.log(math.exp(0.03) - 1.0 + 1e-12)] * q
        raw += [math.log(math.exp(0.88 / max(p, 1)) - 1.0 + 1e-12)] * p
        raw += [2.0]  # raw_scale
        if dist == "studentt":
            raw += [math.log(math.exp(8.0 - 2.0) - 1.0 + 1e-12)]  # raw nu
        return np.asarray(raw, dtype=np.float64)

    def _unpack(self, theta: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
        i = 0
        mu = float(theta[i]); i += 1
        raw_omega = theta[i]; i += 1
        raw_alpha = theta[i:i + self.q]; i += self.q
        raw_gamma = theta[i:i + self.q]; i += self.q
        raw_beta = theta[i:i + self.p]; i += self.p
        raw_scale = theta[i]; i += 1

        omega, alpha, gamma, beta = _pack_stationary(
            raw_omega,
            raw_alpha,
            raw_gamma,
            raw_beta,
            raw_scale,
        )

        nu = None
        if self.dist == "studentt":
            raw_nu = theta[i]
            nu = float(2.0 + _softplus(raw_nu))
            nu = max(nu, 2.05)
        return mu, omega, alpha, gamma, beta, nu

    def _neg_loglik(self, theta: np.ndarray, returns: np.ndarray) -> float:
        mu, omega, alpha, gamma, beta, nu = self._unpack(theta)
        eps, sig2 = _gjr_recursion(returns, mu, omega, alpha, gamma, beta)
        if self.dist == "normal":
            return float(_normal_nll(eps, sig2))
        return float(_student_t_nll(eps, sig2, float(nu)))

    def fit(
        self,
        returns: np.ndarray,
        x0: Optional[np.ndarray] = None,
        maxiter: int = 2000,
        tol: float = 1e-8,
        method: str = "L-BFGS-B",
    ) -> GJRGARCHResult:
        y = _ensure_1d_float(returns)

        if x0 is None:
            x0 = self._default_start(y, self.p, self.q, self.dist)

        opt = minimize(
            fun=lambda th: self._neg_loglik(th, y),
            x0=np.asarray(x0, dtype=np.float64),
            method=method,
            options={"maxiter": maxiter, "ftol": tol},
        )

        mu, omega, alpha, gamma, beta, nu = self._unpack(opt.x)
        eps, sig2 = _gjr_recursion(y, mu, omega, alpha, gamma, beta)
        ll = -float(self._neg_loglik(opt.x, y))

        k = len(opt.x)
        n = y.shape[0]
        aic = 2.0 * k - 2.0 * ll
        bic = math.log(n) * k - 2.0 * ll

        params: Dict[str, Any] = {
            "mu": mu,
            "omega": omega,
            "alpha": alpha.copy(),
            "gamma": gamma.copy(),
            "beta": beta.copy(),
            "dist": self.dist,
            "p": self.p,
            "q": self.q,
        }
        if nu is not None:
            params["nu"] = nu

        self.result_ = GJRGARCHResult(
            params=params,
            loglik=ll,
            aic=aic,
            bic=bic,
            converged=bool(opt.success),
            message=str(opt.message),
            nobs=n,
            residuals=eps,
            conditional_variance=sig2,
            optimizer_x=np.asarray(opt.x, dtype=np.float64).copy(),
            optimizer_fun=float(opt.fun),
            optimizer_nit=int(getattr(opt, "nit", -1)),
        )
        return self.result_

    def summary(self) -> Dict[str, Any]:
        if self.result_ is None:
            raise RuntimeError("Model is not fit yet.")
        r = self.result_
        alpha = r.params["alpha"]
        gamma = r.params["gamma"]
        beta = r.params["beta"]

        persistence = float(np.sum(alpha) + 0.5 * np.sum(gamma) + np.sum(beta))
        out = {
            "converged": r.converged,
            "message": r.message,
            "loglik": r.loglik,
            "aic": r.aic,
            "bic": r.bic,
            "persistence": persistence,
            "unconditional_variance": float(r.params["omega"] / max(1e-12, 1.0 - persistence)),
            "params": r.params,
        }
        return out

    def forecast(self, steps: int = 5) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Model is not fit yet.")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        r = self.result_
        p = self.p
        q = self.q
        eps_hist = r.residuals[-max(q, 1):]
        sig2_hist = r.conditional_variance[-max(p, 1):]
        return _forecast_path(
            r.params["omega"],
            r.params["alpha"],
            r.params["gamma"],
            r.params["beta"],
            eps_hist,
            sig2_hist,
            steps,
        )

    def simulate(
        self,
        n: int,
        burn: int = 1000,
        seed: int = 123,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.result_ is None:
            raise RuntimeError("Fit the model first or pass parameters via set_params-like extension.")
        r = self.result_
        dist_code = 0 if self.dist == "normal" else 1
        nu = float(r.params.get("nu", 8.0))
        return _simulate_path(
            int(n),
            int(burn),
            float(r.params["mu"]),
            float(r.params["omega"]),
            r.params["alpha"],
            r.params["gamma"],
            r.params["beta"],
            int(dist_code),
            float(nu),
            int(seed),
        )

    def get_params(self) -> Dict[str, Any]:
        if self.result_ is None:
            raise RuntimeError("Model is not fit yet.")
        return dict(self.result_.params)

    def residual_diagnostics(self, lags: int = 10) -> Dict[str, float]:
        """
        Lightweight diagnostics:
        - standardized residual mean/variance
        - autocorrelation proxy on squared standardized residuals
        """
        if self.result_ is None:
            raise RuntimeError("Model is not fit yet.")
        eps = self.result_.residuals
        sig = np.sqrt(self.result_.conditional_variance)
        z = eps / sig
        z2 = z * z

        mean_z = float(np.mean(z))
        var_z = float(np.var(z, ddof=1))

        if lags < 1:
            lags = 1
        n = len(z2)
        acf2 = 0.0
        if n > lags + 1:
            x = z2 - np.mean(z2)
            denom = np.sum(x * x)
            if denom > 0:
                acf2 = float(np.sum(x[lags:] * x[:-lags]) / denom)
        return {
            "mean_std_resid": mean_z,
            "var_std_resid": var_z,
            "acf_sq_std_resid_lag": acf2,
        }


def fit_gjr_garch(
    returns: np.ndarray,
    p: int = 1,
    q: int = 1,
    dist: DistName = "normal",
    **fit_kwargs: Any,
) -> GJRGARCHResult:
    """
    Convenience wrapper for one-liner usage.
    """
    model = GJRGARCH(p=p, q=q, dist=dist)
    return model.fit(returns, **fit_kwargs)


__all__ = [
    "GJRGARCH",
    "GJRGARCHResult",
    "fit_gjr_garch",
]
