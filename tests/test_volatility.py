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

    def test_custom_gjr_internal_rescaling_and_bounds(self) -> None:
        rng = np.random.default_rng(42)
        # Generate some volatile data that might trigger numerical issues if unscaled
        series = pd.Series(rng.standard_t(df=4, size=150) * 0.005, name="stress_asset", dtype=float)
        model = GJRGARCHVolatilityModel(rescale_factor=100.0)

        forecast = model.fit_one(series)

        # Ensure it works, doesn't fallback
        self.assertFalse(forecast.fallback_used)
        self.assertTrue(forecast.converged)

        # Verify conditional variance is bounded > 0
        self.assertTrue(np.all(forecast.conditional_variance > 0))
        # Ensure parameters fit within our specified stationary/positivity constraints
        self.assertTrue(forecast.persistence < 1.0)
        # Even if gamma is negative, the variance must have been strictly positive
        alpha_sum = float(np.sum(forecast.params.get("alpha", [])))
        gamma_sum = float(np.sum(forecast.params.get("gamma", [])))
        self.assertTrue(alpha_sum + gamma_sum >= 0.0, "alpha + gamma must be >= 0 for variance positivity")

    def test_custom_gjr_warm_start_disabled_by_default(self) -> None:
        model = GJRGARCHVolatilityModel()
        self.assertFalse(model.use_warm_starts)

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
