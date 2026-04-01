from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.volatility import ArchVolatilityModel, GJRGARCHVolatilityModel


class VolatilityFallbackTests(unittest.TestCase):
    def test_failed_fit_uses_fallback_forecast(self) -> None:
        series = pd.Series(np.linspace(-0.02, 0.02, 60), name="asset", dtype=float)
        model = GJRGARCHVolatilityModel()

        with patch("src.volatility.GJRGARCH.fit", side_effect=RuntimeError("boom")):
            forecast = model.fit_one(series)

        self.assertFalse(forecast.converged)
        self.assertTrue(forecast.fallback_used)
        self.assertEqual(forecast.distribution, "normal")
        self.assertTrue(np.isfinite(forecast.variance_forecast))
        self.assertTrue(np.all(np.isfinite(forecast.standardized_residuals)))

    def test_arch_baseline_fit_returns_valid_forecast(self) -> None:
        rng = np.random.default_rng(7)
        series = pd.Series(rng.normal(scale=0.01, size=120), name="asset", dtype=float)
        model = ArchVolatilityModel(asymmetry=False)
        forecast = model.fit_one(series)

        self.assertTrue(np.isfinite(forecast.variance_forecast))
        self.assertEqual(forecast.asset, "asset")
        self.assertEqual(len(forecast.standardized_residuals), len(series))


if __name__ == "__main__":
    unittest.main()
