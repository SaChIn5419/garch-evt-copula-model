from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto

from src.config import EVT_MIN_EXCEEDANCES, EVT_THRESHOLD_QUANTILE


@dataclass
class EVTForecast:
    valid: bool
    var_z: float | None
    cvar_z: float | None
    threshold: float
    exceedance_rate: float
    shape: float | None
    scale: float | None
    n_exceedances: int
    message: str


def fit_evt_left_tail(
    standardized_residuals: np.ndarray,
    alpha: float,
    threshold_quantile: float = EVT_THRESHOLD_QUANTILE,
    min_exceedances: int = EVT_MIN_EXCEEDANCES,
) -> EVTForecast:
    z = np.asarray(standardized_residuals, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < max(min_exceedances * 2, 50):
        return EVTForecast(False, None, None, np.nan, np.nan, None, None, 0, "insufficient_residual_history")

    losses = -z
    threshold = float(np.quantile(losses, threshold_quantile))
    exceedances = losses[losses > threshold] - threshold
    n_exc = int(exceedances.size)
    exc_rate = n_exc / z.size if z.size else np.nan

    if n_exc < min_exceedances:
        return EVTForecast(False, None, None, threshold, exc_rate, None, None, n_exc, "insufficient_exceedances")
    if not np.isfinite(exc_rate) or alpha >= exc_rate:
        return EVTForecast(False, None, None, threshold, exc_rate, None, None, n_exc, "alpha_not_in_evt_tail")

    try:
        shape, _, scale = genpareto.fit(exceedances, floc=0.0)
    except Exception as exc:
        return EVTForecast(False, None, None, threshold, exc_rate, None, None, n_exc, f"gpd_fit_failed:{exc}")

    shape = float(shape)
    scale = float(scale)
    if scale <= 0.0 or not np.isfinite(scale):
        return EVTForecast(False, None, None, threshold, exc_rate, shape, scale, n_exc, "invalid_scale")
    if shape >= 1.0:
        return EVTForecast(False, None, None, threshold, exc_rate, shape, scale, n_exc, "shape_ge_1")

    ratio = alpha / exc_rate
    if shape == 0.0:
        var_loss = threshold - scale * np.log(ratio)
    else:
        var_loss = threshold + (scale / shape) * (ratio ** (-shape) - 1.0)

    cvar_loss = (var_loss + scale - shape * threshold) / (1.0 - shape)
    if not np.isfinite(var_loss) or not np.isfinite(cvar_loss):
        return EVTForecast(False, None, None, threshold, exc_rate, shape, scale, n_exc, "non_finite_tail_forecast")

    return EVTForecast(
        valid=True,
        var_z=float(-var_loss),
        cvar_z=float(-cvar_loss),
        threshold=threshold,
        exceedance_rate=exc_rate,
        shape=shape,
        scale=scale,
        n_exceedances=n_exc,
        message="ok",
    )
