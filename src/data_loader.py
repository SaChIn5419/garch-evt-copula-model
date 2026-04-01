import pandas as pd
import yfinance as yf
import os

from src.config import BASKETS, CACHE_DIR, DEFAULT_END_DATE, DEFAULT_START_DATE


def _validate_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("Downloaded price frame is empty.")
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive for log-return calculations.")
    return prices


def fetch_basket(
    basket_name,
    start_date=DEFAULT_START_DATE,
    end_date=DEFAULT_END_DATE,
    cache_dir=CACHE_DIR,
):
    if basket_name not in BASKETS:
        raise ValueError(f"Basket '{basket_name}' not defined in config.")

    tickers = BASKETS[basket_name]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{basket_name}_{start_date}_{end_date}.parquet")

    if os.path.exists(cache_path):
        print(f"Loading {basket_name} from cache: {cache_path}")
        return _validate_price_frame(pd.read_parquet(cache_path))

    print(f"Downloading {basket_name} data ({len(tickers)} tickers) from {start_date} to {end_date}...")
    df = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by="column")

    # We want 'Adj Close'
    if "Adj Close" in df:
        adj_close = df["Adj Close"]
    elif "Close" in df:
        adj_close = df["Close"]
    else:
        raise KeyError("Could not find 'Adj Close' or 'Close' in downloaded data.")

    # Handle single ticker edge cases correctly
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=tickers[0])

    # Mixed-market holiday gaps should not be forward-filled into synthetic flat returns.
    adj_close = _validate_price_frame(adj_close).dropna(how="any")

    # Save to local cache
    adj_close.columns = adj_close.columns.astype(str)
    adj_close.to_parquet(cache_path)

    print(f"Downloaded and cached {basket_name} block. Data shape: {adj_close.shape}.")
    return adj_close
