from __future__ import annotations

from src.config import BASKETS
from src.backtest import run_walk_forward_backtest
from src.data_loader import fetch_basket
from src.preprocessing import log_returns
from src.report import make_run_name, save_backtest_results


def main() -> None:
    for basket_name in BASKETS:
        prices = fetch_basket(basket_name)
        returns = log_returns(prices)
        result = run_walk_forward_backtest(returns)
        output_dir = save_backtest_results(
            result,
            run_name=make_run_name(basket_name, str(prices.index.min().date()), str(prices.index.max().date())),
        )

        print(f"\nBasket: {basket_name}")
        print("Portfolio summary")
        for key, value in result.summary.items():
            print(f"{key}: {value}")
        print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
