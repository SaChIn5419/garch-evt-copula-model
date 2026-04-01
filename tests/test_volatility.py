from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.volatility import GJRGARCHVolatilityModel


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


if __name__ == "__main__":
    unittest.main()
