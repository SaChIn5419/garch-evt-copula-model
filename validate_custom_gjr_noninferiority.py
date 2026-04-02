from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.config import BASKETS, ROLLING_WINDOW
from src.data_loader import fetch_basket
from src.preprocessing import log_returns
from src.report import make_run_name
from src.volatility import ArchVolatilityModel, GJRGARCHVolatilityModel


EVALUATION_DAYS = (120, 252)
ANCHOR_OFFSETS = (0,)


def _scenario_frame(
    basket_name: str,
    evaluation_days: int,
    anchor_offset: int,
) -> pd.DataFrame:
    prices = fetch_basket(basket_name)
    returns = log_returns(prices).dropna(how="any").astype(float)
    if anchor_offset > 0:
        if len(returns) <= anchor_offset + ROLLING_WINDOW + evaluation_days:
            raise ValueError("Not enough observations for requested anchor offset.")
        returns = returns.iloc[:-anchor_offset]

    keep_rows = min(len(returns), ROLLING_WINDOW + evaluation_days)
    returns = returns.tail(keep_rows).dropna(how="any").astype(float)
    return returns


def _run_model_pair(returns: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    arch_result = run_walk_forward_backtest(returns, model=ArchVolatilityModel(asymmetry=True))
    custom_result = run_walk_forward_backtest(returns, model=GJRGARCHVolatilityModel())
    return arch_result.summary, custom_result.summary


def _evaluate_noninferiority(arch_summary: dict[str, float], custom_summary: dict[str, float]) -> dict[str, float | int]:
    n_obs = int(arch_summary["n_obs"])
    one_breach_rate = 1.0 / max(n_obs, 1)

    breach_diff = float(custom_summary["breach_rate"] - arch_summary["breach_rate"])
    cvar_diff = float(custom_summary["portfolio_cvar_breach_rate"] - arch_summary["portfolio_cvar_breach_rate"])
    breach_count_diff = float(custom_summary["n_breaches"] - arch_summary["n_breaches"])

    pass_breach = int(breach_diff <= one_breach_rate + 1e-12)
    pass_breach_count = int(breach_count_diff <= 1.0 + 1e-12)
    pass_cvar = int(cvar_diff <= one_breach_rate + 1e-12)
    pass_fit = int(
        float(custom_summary["asset_fit_convergence_rate"]) >= float(arch_summary["asset_fit_convergence_rate"]) - 1e-12
        and float(custom_summary["asset_fit_fallback_rate"]) <= float(arch_summary["asset_fit_fallback_rate"]) + 1e-12
    )

    return {
        "one_breach_rate_tolerance": one_breach_rate,
        "breach_rate_diff": breach_diff,
        "breach_count_diff": breach_count_diff,
        "cvar_breach_rate_diff": cvar_diff,
        "pass_breach_rate": pass_breach,
        "pass_breach_count": pass_breach_count,
        "pass_cvar_breach_rate": pass_cvar,
        "pass_fit_stability": pass_fit,
        "noninferior": int(pass_breach and pass_breach_count and pass_cvar and pass_fit),
    }


def run_noninferiority_validation(base_dir: str = "results_validation_noninferiority") -> Path:
    output_dir = Path(base_dir) / make_run_name("custom_gjr_noninferiority", "multiwindow", "2023-12-29")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for basket_name in BASKETS:
        for evaluation_days in EVALUATION_DAYS:
            for anchor_offset in ANCHOR_OFFSETS:
                try:
                    returns = _scenario_frame(basket_name, evaluation_days, anchor_offset)
                except ValueError:
                    continue

                arch_summary, custom_summary = _run_model_pair(returns)
                verdict = _evaluate_noninferiority(arch_summary, custom_summary)

                rows.append(
                    {
                        "basket": basket_name,
                        "evaluation_days": evaluation_days,
                        "anchor_offset_days": anchor_offset,
                        "start_date": str(returns.index.min().date()),
                        "end_date": str(returns.index.max().date()),
                        "n_obs": int(arch_summary["n_obs"]),
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
                        **verdict,
                    }
                )

    frame = pd.DataFrame(rows).sort_values(["basket", "evaluation_days", "anchor_offset_days"])
    frame.to_csv(output_dir / "scenario_results.csv", index=False)

    summary = (
        frame.groupby("basket", as_index=False)
        .agg(
            scenarios=("noninferior", "count"),
            pass_rate=("noninferior", "mean"),
            breach_rate_pass_rate=("pass_breach_rate", "mean"),
            breach_count_pass_rate=("pass_breach_count", "mean"),
            cvar_pass_rate=("pass_cvar_breach_rate", "mean"),
            fit_pass_rate=("pass_fit_stability", "mean"),
            mean_breach_rate_diff=("breach_rate_diff", "mean"),
            mean_breach_count_diff=("breach_count_diff", "mean"),
            mean_cvar_breach_rate_diff=("cvar_breach_rate_diff", "mean"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    records = {
        "scenarios": frame.where(pd.notna(frame), None).to_dict(orient="records"),
        "summary": summary.where(pd.notna(summary), None).to_dict(orient="records"),
    }
    with open(output_dir / "summary.json", "w", encoding="ascii") as f:
        json.dump(records, f, indent=2, allow_nan=False)

    lines = ["# Custom GJR Non-Inferiority Validation", ""]
    lines.append("Decision rule:")
    lines.append("- custom breach rate may not exceed arch breach rate by more than one additional breach on the scenario horizon")
    lines.append("- custom breach count may not exceed arch breach count by more than 1")
    lines.append("- custom CVaR breach rate may not exceed arch CVaR breach rate by more than one additional breach on the scenario horizon")
    lines.append("- custom fit stability may not be worse than arch on convergence/fallback rate")
    lines.append("")

    lines.append("## Basket Summary")
    lines.append("")
    for _, row in summary.iterrows():
        lines.append(f"### {row['basket']}")
        lines.append("")
        lines.append(f"- scenarios: {int(row['scenarios'])}")
        lines.append(f"- noninferiority pass rate: {row['pass_rate']}")
        lines.append(f"- breach-rate pass rate: {row['breach_rate_pass_rate']}")
        lines.append(f"- breach-count pass rate: {row['breach_count_pass_rate']}")
        lines.append(f"- cvar pass rate: {row['cvar_pass_rate']}")
        lines.append(f"- fit-stability pass rate: {row['fit_pass_rate']}")
        lines.append(f"- mean breach-rate diff: {row['mean_breach_rate_diff']}")
        lines.append(f"- mean breach-count diff: {row['mean_breach_count_diff']}")
        lines.append(f"- mean cvar-breach-rate diff: {row['mean_cvar_breach_rate_diff']}")
        lines.append("")

    lines.append("## Scenario Results")
    lines.append("")
    for _, row in frame.iterrows():
        lines.append(
            f"- {row['basket']} | eval={int(row['evaluation_days'])} | offset={int(row['anchor_offset_days'])} | "
            f"arch_breaches={row['arch_n_breaches']} | custom_breaches={row['custom_n_breaches']} | "
            f"breach_diff={row['breach_rate_diff']} | noninferior={int(row['noninferior'])}"
        )

    with open(output_dir / "summary.md", "w", encoding="ascii") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return output_dir


def main() -> None:
    output_dir = run_noninferiority_validation()
    print(f"Saved noninferiority validation to: {output_dir}")


if __name__ == "__main__":
    main()
