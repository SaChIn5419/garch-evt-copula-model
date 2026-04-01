import numpy as np
import pandas as pd


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Computes log returns securely.
    Ensures that t is only computed from t and t-1.
    """
    if (prices <= 0).any().any() if isinstance(prices, pd.DataFrame) else (prices <= 0).any():
        raise ValueError("Prices must be strictly positive for log-return calculations.")
    return np.log(prices).diff().dropna()


def expanding_zscore(x: pd.Series) -> pd.Series:
    """
    Computes an expanding Z-Score safely.
    Mean and Std are calculated *strictly* up to t-1 and applied to t to prevent look-ahead bias.
    """
    mean = x.expanding().mean().shift(1)
    std = x.expanding().std(ddof=1).shift(1)
    return (x - mean) / std.replace(0.0, np.nan)


def realized_volatility(x: pd.Series, window: int = 21) -> pd.Series:
    """Rolling realized volatility using only past observations."""
    return x.rolling(window=window).std(ddof=1)


def rolling_drawdown(x: pd.Series, window: int = 63) -> pd.Series:
    """Rolling drawdown on a returns series converted to cumulative wealth."""
    wealth = (1.0 + x).cumprod()
    peak = wealth.rolling(window=window, min_periods=1).max()
    return wealth / peak - 1.0
