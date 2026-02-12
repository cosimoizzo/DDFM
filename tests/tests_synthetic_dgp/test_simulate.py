import unittest
import numpy as np

from synthetic_dgp.simulate import SIMULATE, QuarterlyVars, AggregationInstr


class TestSIMULATE(unittest.TestCase):
    def setUp(self) -> None:
        self.simulate = SIMULATE(seed=1)
        self._simulate_repl = SIMULATE(seed=1)

    def tearDown(self) -> None:
        del self.simulate

    def test_simulate(self):
        x = self.simulate.simulate(100)
        self.assertEqual(x.shape, (100, 10))
        self.assertEqual(np.sum(np.isnan(x)), 0)
        self.assertEqual(np.sum(np.isfinite(x)), 10 * 100)
        self.assertEqual(self.simulate.f.shape, (100, 1))
        self.assertEqual(np.sum(np.isnan(self.simulate.f)), 0)
        self.assertEqual(np.sum(np.isfinite(self.simulate.f)), 1 * 100)
        self.assertEqual(np.sum(np.isfinite(self.simulate.f)), 1 * 100)

    def test_simulate_missings(self):
        x = self.simulate.simulate(100, portion_missings=0.2)
        n_missings = int(100 * 10 * 0.2)
        self.assertEqual(x.shape, (100, 10))
        self.assertEqual(np.sum(np.isnan(x)), n_missings)
        self.assertEqual(np.sum(np.isfinite(x)), 10 * 100 - n_missings)
        self.assertEqual(self.simulate.f.shape, (100, 1))
        self.assertEqual(np.sum(np.isnan(self.simulate.f)), 0)
        self.assertEqual(np.sum(np.isfinite(self.simulate.f)), 1 * 100)
        self.assertEqual(np.sum(np.isfinite(self.simulate.f)), 1 * 100)

    def test_quarterly_simulate(self):
        quartely_vars = QuarterlyVars([1, 2], aggregation=AggregationInstr.MM)
        x = self.simulate.simulate(100, quarterly_vars=quartely_vars)
        x_repl = self._simulate_repl.simulate(100)
        x_repl = quartely_vars._aggregate_slow(x_repl)
        self.assertEqual(x.shape, (100, 10))
        # quarterly from monthly with MM aggregation restrictions
        exp_missing = np.ceil(2 / 3 * 100 * 2 + 1 * 2)
        self.assertEqual(np.sum(np.isnan(x)), exp_missing)
        self.assertEqual(np.sum(np.isfinite(x)), 10 * 100 - exp_missing)
        np.testing.assert_array_almost_equal(x, x_repl)
        self.assertEqual(self.simulate.f.shape, (100, 1))
        self.assertEqual(np.sum(np.isnan(self.simulate.f)), 0)
        self.assertEqual(np.sum(np.isfinite(self.simulate.f)), 1 * 100)
        self.assertEqual(np.sum(np.isfinite(self.simulate.f)), 1 * 100)

    def test_evaluate(self):
        _ = self.simulate.simulate(100, portion_missings=0.2)
        self.assertEqual(self.simulate.evaluate(self.simulate.f), 1)


if __name__ == "__main__":
    unittest.main()
