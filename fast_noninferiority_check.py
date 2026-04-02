from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from dataclasses import dataclass

import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.config import ROLLING_WINDOW
from src.data_loader import fetch_basket
from src.preprocessing import log_returns
from src.volatility import ArchVolatilityModel, GJRGARCHVolatilityModel


@dataclass(frozen=True)
class Scenario:
    basket: str
    evaluation_days: int
    anchor_offset: int = 0


def _run_scenario(s: Scenario) -> dict[str, float | int | str]:
    prices = fetch_basket(s.basket)
    returns = log_returns(prices).dropna(how="any").astype(float)
    if s.anchor_offset > 0:
        returns = returns.iloc[:-s.anchor_offset]
    returns = returns.tail(min(len(returns), ROLLING_WINDOW + s.evaluation_days))

    arch_summary = run_walk_forward_backtest(returns, model=ArchVolatilityModel(asymmetry=True)).summary
    custom_summary = run_walk_forward_backtest(returns, model=GJRGARCHVolatilityModel()).summary

    n_obs = int(arch_summary["n_obs"])
    one_breach_tol = 1.0 / max(n_obs, 1)
    noninferior = int(
        (custom_summary["breach_rate"] - arch_summary["breach_rate"] <= one_breach_tol + 1e-12)
        and (custom_summary["n_breaches"] - arch_summary["n_breaches"] <= 1.0 + 1e-12)
        and (
            custom_summary["portfolio_cvar_breach_rate"] - arch_summary["portfolio_cvar_breach_rate"]
            <= one_breach_tol + 1e-12
        )
        and (custom_summary["asset_fit_convergence_rate"] >= arch_summary["asset_fit_convergence_rate"] - 1e-12)
        and (custom_summary["asset_fit_fallback_rate"] <= arch_summary["asset_fit_fallback_rate"] + 1e-12)
    )

    return {
        "basket": s.basket,
        "evaluation_days": s.evaluation_days,
        "anchor_offset": s.anchor_offset,
        "start_date": str(returns.index.min().date()),
        "end_date": str(returns.index.max().date()),
        "n_obs": n_obs,
        "arch_n_breaches": float(arch_summary["n_breaches"]),
        "custom_n_breaches": float(custom_summary["n_breaches"]),
        "arch_breach_rate": float(arch_summary["breach_rate"]),
        "custom_breach_rate": float(custom_summary["breach_rate"]),
        "arch_cvar_breach_rate": float(arch_summary["portfolio_cvar_breach_rate"]),
        "custom_cvar_breach_rate": float(custom_summary["portfolio_cvar_breach_rate"]),
        "arch_fit_convergence_rate": float(arch_summary["asset_fit_convergence_rate"]),
        "custom_fit_convergence_rate": float(custom_summary["asset_fit_convergence_rate"]),
        "arch_fit_fallback_rate": float(arch_summary["asset_fit_fallback_rate"]),
        "custom_fit_fallback_rate": float(custom_summary["asset_fit_fallback_rate"]),
        "one_breach_rate_tolerance": one_breach_tol,
        "breach_rate_diff": float(custom_summary["breach_rate"] - arch_summary["breach_rate"]),
        "breach_count_diff": float(custom_summary["n_breaches"] - arch_summary["n_breaches"]),
        "cvar_breach_rate_diff": float(
            custom_summary["portfolio_cvar_breach_rate"] - arch_summary["portfolio_cvar_breach_rate"]
        ),
        "noninferior": noninferior,
    }


def _make_run_name(prefix: str, start_date: str, end_date: str) -> str:
    return f"{prefix}_{start_date}_{end_date}"


def main() -> None:
    scenarios = [
        Scenario("india_primary", 120),
        Scenario("india_primary", 252),
        Scenario("us_stress", 120),
        Scenario("us_stress", 252),
    ]

    rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(max_workers=min(len(scenarios), os.cpu_count() or 1)) as pool:
        futures = {pool.submit(_run_scenario, s): s for s in scenarios}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['basket']} eval={row['evaluation_days']} "
                f"arch_breaches={row['arch_n_breaches']} custom_breaches={row['custom_n_breaches']} "
                f"arch_rate={row['arch_breach_rate']:.6f} custom_rate={row['custom_breach_rate']:.6f} "
                f"arch_cvar={row['arch_cvar_breach_rate']:.6f} custom_cvar={row['custom_cvar_breach_rate']:.6f} "
                f"noninferior={row['noninferior']}"
            )

    frame = pd.DataFrame(rows).sort_values(["basket", "evaluation_days", "anchor_offset"])
    output_dir = Path("results_validation_noninferiority") / _make_run_name(
        "fast_custom_gjr_noninferiority", "core", "2023-12-29"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "scenario_results.csv", index=False)

    summary = (
        frame.groupby("basket", as_index=False)
        .agg(
            scenarios=("noninferior", "count"),
            pass_rate=("noninferior", "mean"),
            mean_breach_rate_diff=("breach_rate_diff", "mean"),
            mean_breach_count_diff=("breach_count_diff", "mean"),
            mean_cvar_breach_rate_diff=("cvar_breach_rate_diff", "mean"),
        )
        .sort_values("basket")
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="ascii") as f:
        json.dump(
            {
                "summary": summary.where(pd.notna(summary), None).to_dict(orient="records"),
                "scenarios": frame.where(pd.notna(frame), None).to_dict(orient="records"),
            },
            f,
            indent=2,
            allow_nan=False,
        )

    lines = ["# Fast Custom GJR Non-Inferiority Check", ""]
    lines.append("Rule:")
    lines.append("- custom may not exceed arch by more than one breach-equivalent on breach rate or CVaR breach rate")
    lines.append("- custom may not exceed arch by more than 1 breach count")
    lines.append("- custom may not have worse fit stability than arch")
    lines.append("")
    lines.append("## Basket Summary")
    lines.append("")
    for _, row in summary.iterrows():
        lines.append(f"### {row['basket']}")
        lines.append("")
        lines.append(f"- scenarios: {int(row['scenarios'])}")
        lines.append(f"- pass rate: {row['pass_rate']}")
        lines.append(f"- mean breach-rate diff: {row['mean_breach_rate_diff']}")
        lines.append(f"- mean breach-count diff: {row['mean_breach_count_diff']}")
        lines.append(f"- mean cvar-breach-rate diff: {row['mean_cvar_breach_rate_diff']}")
        lines.append("")
    lines.append("## Scenario Results")
    lines.append("")
    for _, row in frame.iterrows():
        lines.append(
            f"- {row['basket']} | eval={int(row['evaluation_days'])} | "
            f"{row['start_date']} to {row['end_date']} | "
            f"arch_breaches={row['arch_n_breaches']} | custom_breaches={row['custom_n_breaches']} | "
            f"noninferior={int(row['noninferior'])}"
        )
    with open(output_dir / "summary.md", "w", encoding="ascii") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"Saved fast noninferiority outputs to: {output_dir}")


if __name__ == "__main__":
    main()
