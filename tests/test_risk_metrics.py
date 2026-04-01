from __future__ import annotations

import unittest

import numpy as np

from src.risk_metrics import christoffersen_test, count_exception_clusters, kupiec_test


class RiskMetricTests(unittest.TestCase):
    def test_exception_clusters_are_counted(self) -> None:
        flags = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 0], dtype=int)
        self.assertEqual(count_exception_clusters(flags), 3.0)

    def test_backtests_return_finite_statistics_on_mixed_sequence(self) -> None:
        flags = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 0] * 5, dtype=int)
        kupiec = kupiec_test(flags, alpha=0.2)
        christoffersen = christoffersen_test(flags)

        self.assertTrue(np.isfinite(kupiec["kupiec_lr"]))
        self.assertTrue(np.isfinite(kupiec["kupiec_pvalue"]))
        self.assertTrue(np.isfinite(christoffersen["christoffersen_lr"]))
        self.assertTrue(np.isfinite(christoffersen["christoffersen_pvalue"]))


if __name__ == "__main__":
    unittest.main()
