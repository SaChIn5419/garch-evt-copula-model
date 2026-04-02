from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest import BacktestResult, run_walk_forward_backtest
from src.config import ROLLING_WINDOW
from src.data_loader import fetch_basket
from src.preprocessing import log_returns
from src.report import make_run_name, save_backtest_results
from src.volatility import GJRGARCHVolatilityModel


def _comparison_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    metrics = [
        "n_obs",
        "n_breaches",
        "breach_rate",
        "portfolio_exception_clusters",
        "kupiec_pvalue",
        "christoffersen_pvalue",
        "portfolio_cvar_breach_rate",
        "asset_fit_convergence_rate",
        "asset_fit_fallback_rate",
        "mean_gross_exposure",
        "mean_absolute_net_exposure",
        "evt_usage_rate",
        "mean_copula_avg_abs_correlation",
        "mean_copula_max_abs_correlation",
        "mean_copula_lower_tail_dependence",
    ]
    rows: list[dict[str, float | str | None]] = []
    for model_name, result in results.items():
        row: dict[str, float | str | None] = {"copula_family": model_name}
        for metric in metrics:
            row[metric] = result.summary.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_comparison_summary(output_dir: Path, basket_name: str, frame: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "comparison.csv", index=False)

    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    with open(output_dir / "comparison.json", "w", encoding="ascii") as f:
        json.dump({"basket": basket_name, "copulas": records}, f, indent=2, allow_nan=False)

    lines = [f"# Copula Family Comparison: {basket_name}", ""]
    for _, row in frame.iterrows():
        lines.append(f"## {row['copula_family']}")
        lines.append("")
        for key, value in row.items():
            if key == "copula_family":
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")

    with open(output_dir / "comparison.md", "w", encoding="ascii") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def run_copula_comparison(
    basket_name: str = "india_primary",
    base_dir: str = "results_copula_compare",
    evaluation_days: int = 252,
) -> Path:
    prices = fetch_basket(basket_name)
    returns = log_returns(prices)
    keep_rows = min(len(returns), ROLLING_WINDOW + evaluation_days)
    returns = returns.tail(keep_rows).dropna(how="any").astype(float)

    run_name = make_run_name(
        f"{basket_name}_copula_compare",
        str(returns.index.min().date()),
        str(returns.index.max().date()),
    )
    output_dir = Path(base_dir) / run_name

    results: dict[str, BacktestResult] = {}
    for family in ("gaussian", "studentt"):
        result = run_walk_forward_backtest(
            returns,
            model=GJRGARCHVolatilityModel(),
            copula_family=family,
        )
        results[family] = result
        save_backtest_results(result, run_name=f"{run_name}_{family}", base_dir=str(output_dir))

    frame = _comparison_frame(results)
    _write_comparison_summary(output_dir, basket_name, frame)
    return output_dir


def main() -> None:
    output_dir = run_copula_comparison()
    print(f"Saved copula family comparison to: {output_dir}")


if __name__ == "__main__":
    main()
