from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import GROSS_EXPOSURE, NET_EXPOSURE, POSITION_BOUND


def _allocate_equal_weight(indices: np.ndarray, target_sum: float, size: int) -> np.ndarray:
    alloc = np.zeros(size, dtype=float)
    if indices.size == 0 or target_sum <= 0.0:
        return alloc
    alloc[indices] = target_sum / indices.size
    return alloc


def _select_side(signal: np.ndarray, count: int, descending: bool) -> np.ndarray:
    order = np.argsort(signal)
    if descending:
        order = order[::-1]
    return np.sort(order[:count])


def deterministic_dollar_neutral_weights(
    signal: pd.Series,
    gross_target: float = GROSS_EXPOSURE,
    net_target: float = NET_EXPOSURE,
    position_bound: float = POSITION_BOUND,
) -> pd.Series:
    if position_bound <= 0.0:
        raise ValueError("position_bound must be positive.")
    if gross_target < abs(net_target):
        raise ValueError("gross_target must be at least the absolute net_target.")

    n_assets = len(signal)
    if n_assets < 2:
        raise ValueError("Need at least two assets for a dollar-neutral portfolio.")

    max_gross = n_assets * position_bound
    if gross_target > max_gross + 1e-12:
        raise ValueError("Requested gross exposure is infeasible under the position bound.")

    centered = signal.astype(float).fillna(0.0)
    centered = centered - centered.mean()
    if np.allclose(centered.to_numpy(), 0.0):
        ranks = np.arange(n_assets, dtype=float)
        centered = pd.Series(ranks - ranks.mean(), index=signal.index, dtype=float)

    long_budget = 0.5 * (gross_target + net_target)
    short_budget = 0.5 * (gross_target - net_target)
    long_count = max(1, int(math.ceil(long_budget / position_bound))) if long_budget > 0.0 else 0
    short_count = max(1, int(math.ceil(short_budget / position_bound))) if short_budget > 0.0 else 0
    if long_count + short_count > n_assets:
        raise ValueError("Not enough assets to satisfy gross exposure under the position bound.")

    signal_np = centered.to_numpy(dtype=float)
    long_idx = _select_side(signal_np, long_count, descending=True)
    remaining = np.setdiff1d(np.arange(n_assets), long_idx, assume_unique=True)
    short_candidates = signal_np[remaining]
    short_pick = _select_side(short_candidates, short_count, descending=False)
    short_idx = np.sort(remaining[short_pick])

    weights = _allocate_equal_weight(long_idx, long_budget, n_assets)
    weights -= _allocate_equal_weight(short_idx, short_budget, n_assets)
    return pd.Series(weights, index=signal.index, name="weight")


def dollar_neutral_vol_weights(
    forecast_volatility: pd.Series,
    gross_target: float = GROSS_EXPOSURE,
    net_target: float = NET_EXPOSURE,
    position_bound: float = POSITION_BOUND,
) -> pd.Series:
    sigma = forecast_volatility.astype(float).replace(0.0, np.nan)
    signal = -np.log(sigma)
    return deterministic_dollar_neutral_weights(
        signal.fillna(0.0),
        gross_target=gross_target,
        net_target=net_target,
        position_bound=position_bound,
    )
