from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.config import ROLLING_WINDOW
from src.data_loader import fetch_basket
from src.preprocessing import log_returns


@dataclass(frozen=True)
class Scenario:
    basket: str
    evaluation_days: int


def _make_run_name(prefix: str, start_date: str, end_date: str) -> str:
    return f"{prefix}_{start_date}_{end_date}"


def _run_scenario(s: Scenario) -> dict[str, float | str]:
    prices = fetch_basket(s.basket)
    returns = log_returns(prices).dropna(how="any").astype(float)
    returns = returns.tail(min(len(returns), ROLLING_WINDOW + s.evaluation_days))

    gaussian = run_walk_forward_backtest(returns, copula_family="gaussian").summary
    studentt = run_walk_forward_backtest(returns, copula_family="studentt").summary

    return {
        "basket": s.basket,
        "evaluation_days": s.evaluation_days,
        "start_date": str(returns.index.min().date()),
        "end_date": str(returns.index.max().date()),
        "gaussian_n_breaches": float(gaussian["n_breaches"]),
        "studentt_n_breaches": float(studentt["n_breaches"]),
        "gaussian_breach_rate": float(gaussian["breach_rate"]),
        "studentt_breach_rate": float(studentt["breach_rate"]),
        "gaussian_cvar_breach_rate": float(gaussian["portfolio_cvar_breach_rate"]),
        "studentt_cvar_breach_rate": float(studentt["portfolio_cvar_breach_rate"]),
        "gaussian_avg_abs_corr": float(gaussian["mean_copula_avg_abs_correlation"]),
        "studentt_avg_abs_corr": float(studentt["mean_copula_avg_abs_correlation"]),
        "gaussian_tail_dep": float(gaussian["mean_copula_lower_tail_dependence"]),
        "studentt_tail_dep": float(studentt["mean_copula_lower_tail_dependence"]),
    }


def main() -> None:
    scenarios = [
        Scenario("india_primary", 252),
        Scenario("us_stress", 252),
    ]
    rows: list[dict[str, float | str]] = []
    with ProcessPoolExecutor(max_workers=min(len(scenarios), os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_run_scenario, s): s for s in scenarios}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['basket']} eval={row['evaluation_days']} "
                f"gaussian_breaches={row['gaussian_n_breaches']} studentt_breaches={row['studentt_n_breaches']} "
                f"gaussian_rate={row['gaussian_breach_rate']:.6f} studentt_rate={row['studentt_breach_rate']:.6f} "
                f"gaussian_cvar={row['gaussian_cvar_breach_rate']:.6f} studentt_cvar={row['studentt_cvar_breach_rate']:.6f}"
            )

    frame = pd.DataFrame(rows).sort_values(["basket", "evaluation_days"])
    output_dir = Path("results_copula_compare") / _make_run_name("fast_copula_family_check", "core", "2023-12-29")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "scenario_results.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="ascii") as f:
        json.dump(frame.where(pd.notna(frame), None).to_dict(orient="records"), f, indent=2, allow_nan=False)

    lines = ["# Fast Copula Family Check", ""]
    for _, row in frame.iterrows():
        lines.append(f"## {row['basket']}")
        lines.append("")
        for key, value in row.items():
            if key == "basket":
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")
    with open(output_dir / "summary.md", "w", encoding="ascii") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"Saved fast copula outputs to: {output_dir}")


if __name__ == "__main__":
    main()
