from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.strategy import dollar_neutral_vol_weights


class StrategyTests(unittest.TestCase):
    def test_dollar_neutral_weights_respect_constraints(self) -> None:
        forecast_vol = pd.Series(
            [0.05, 0.07, 0.09, 0.15, 0.20],
            index=["a", "b", "c", "d", "e"],
            dtype=float,
        )
        weights = dollar_neutral_vol_weights(forecast_vol)

        self.assertAlmostEqual(float(weights.sum()), 0.0, places=10)
        self.assertAlmostEqual(float(weights.abs().sum()), 1.0, places=10)
        self.assertTrue((weights.abs() <= 0.30 + 1e-12).all())


if __name__ == "__main__":
    unittest.main()
