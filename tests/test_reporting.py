from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.backtest import BacktestResult
from src.report import save_backtest_results


class ReportingTests(unittest.TestCase):
    def test_summary_json_is_strict_json(self) -> None:
        portfolio = pd.DataFrame(
            {
                "portfolio_return": [-0.1, -0.2],
                "portfolio_var_forecast": [-0.05, -0.06],
                "portfolio_cvar_forecast": [-0.08, -0.09],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        weights = pd.DataFrame({"a": [0.1, 0.2], "b": [-0.1, -0.2]}, index=portfolio.index)
        asset = pd.DataFrame(
            {
                "date": portfolio.index,
                "asset": ["a", "b"],
                "converged": [1, 1],
                "fallback_used": [0, 0],
                "persistence": [0.5, 0.5],
                "optimizer_nit": [1, 1],
                "evt_valid": [1, 1],
            }
        )
        result = BacktestResult(asset, portfolio, weights, {"x": float("nan")})

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = save_backtest_results(result, "run", base_dir=tmp_dir)
            summary = json.loads(Path(output_dir / "summary.json").read_text(encoding="ascii"))

        self.assertIsNone(summary["x"])


if __name__ == "__main__":
    unittest.main()
