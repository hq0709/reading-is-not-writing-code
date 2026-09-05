"""Analytic checks for the validation screen's bidirectional decision."""

import unittest

import numpy as np

from src import llava_validation_opportunity as core


class AnalyticDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indices = np.random.default_rng(20260915).integers(0, 100, size=(5000, 100), dtype=np.int64)
        cls.labels = np.arange(100) % 2

    def evaluate(self, baseline, clinical, sham, random=(-.1, .1)):
        margins = np.full((45, 100), baseline, dtype=float)
        margins[1], margins[2] = clinical
        for start in range(3, 43, 2):
            margins[start], margins[start + 1] = random
        margins[43], margins[44] = sham
        return core.write_statistics(margins, self.labels, self.indices)

    def test_constant_response_has_exact_analytic_control_excess(self):
        result, arrays = self.evaluate(0, (-1, 1), (-.3, .3))
        expected = 1 / (1 + np.exp(-1)) - 1 / (1 + np.exp(-.3))
        self.assertAlmostEqual(result['probability']['excess'], expected, places=12)
        np.testing.assert_allclose(arrays['bootstrap_probability_excess'], expected, atol=1e-12, rtol=0)
        self.assertTrue(result['opportunity_pass'])

    def test_oppositely_oriented_sham_competes_by_absolute_response(self):
        result, arrays = self.evaluate(0, (-1, 1), (2, -2))
        expected = 1 / (1 + np.exp(-1)) - 1 / (1 + np.exp(-2))
        self.assertAlmostEqual(result['probability']['excess'], expected, places=12)
        np.testing.assert_allclose(arrays['bootstrap_probability_excess'], expected, atol=1e-12, rtol=0)
        self.assertFalse(result['opportunity_pass'])

    def test_positive_central_difference_still_requires_both_baseline_signs(self):
        result, _ = self.evaluate(-2, (.5, 1), (0, 0), random=(0, 0))
        self.assertGreater(result['probability']['excess_lower_95'], 0)
        self.assertFalse(result['opportunity_pass'])


if __name__ == '__main__':
    unittest.main()
