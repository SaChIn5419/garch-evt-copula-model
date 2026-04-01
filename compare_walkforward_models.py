from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest import BacktestResult, run_walk_forward_backtest
from src.config import ROLLING_WINDOW
from src.data_loader import fetch_basket
from src.preprocessing import log_returns
from src.report import make_run_name, save_backtest_results
from src.volatility import ArchVolatilityModel, GJRGARCHVolatilityModel


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
    ]
    rows: list[dict[str, float | str | None]] = []
    for model_name, result in results.items():
        row: dict[str, float | str | None] = {"model": model_name}
        for metric in metrics:
            row[metric] = result.summary.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_comparison_summary(output_dir: Path, basket_name: str, frame: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "comparison.csv", index=False)

    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        record: dict[str, object] = {}
        for key, value in row.items():
            if key == "model":
                record[key] = value
            elif pd.isna(value):
                record[key] = None
            else:
                record[key] = float(value)
        records.append(record)
    with open(output_dir / "comparison.json", "w", encoding="ascii") as f:
        json.dump({"basket": basket_name, "models": records}, f, indent=2, allow_nan=False)

    lines = [f"# Walk-Forward Model Comparison: {basket_name}", ""]
    for _, row in frame.iterrows():
        lines.append(f"## {row['model']}")
        lines.append("")
        for col, value in row.items():
            if col == "model":
                continue
            lines.append(f"- {col}: {value}")
        lines.append("")

    with open(output_dir / "comparison.md", "w", encoding="ascii") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def run_comparison(
    basket_name: str = "us_stress",
    base_dir: str = "results_validation",
    evaluation_days: int = 252,
) -> Path:
    prices = fetch_basket(basket_name)
    returns = log_returns(prices)
    if evaluation_days > 0:
        keep_rows = min(len(returns), ROLLING_WINDOW + evaluation_days)
        returns = returns.tail(keep_rows)
    run_name = make_run_name(
        basket_name,
        str(returns.index.min().date()),
        str(returns.index.max().date()),
    )
    comparison_dir = Path(base_dir) / run_name

    models = {
        "garch_baseline": ArchVolatilityModel(asymmetry=False),
        "gjr_custom": GJRGARCHVolatilityModel(),
    }
    results: dict[str, BacktestResult] = {}

    for model_name, model in models.items():
        result = run_walk_forward_backtest(returns, model=model)
        results[model_name] = result
        save_backtest_results(result, run_name=f"{run_name}_{model_name}", base_dir=str(comparison_dir))

    frame = _comparison_frame(results)
    _write_comparison_summary(comparison_dir, basket_name, frame)
    return comparison_dir


def main() -> None:
    output_dir = run_comparison()
    print(f"Saved validation comparison to: {output_dir}")


if __name__ == "__main__":
    main()
