import unittest
import numpy as np

from synthetic_dgp.simulate import SIMULATE


class TestSIMULATE(unittest.TestCase):
    """
    A class containing tests for the SIMULATE.
    """

    def setUp(self) -> None:
        self.simulate = SIMULATE(seed=1)

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

    def test_evaluate(self):
        _ = self.simulate.simulate(100, portion_missings=0.2)
        self.assertEqual(self.simulate.evaluate(self.simulate.f), 1)


if __name__ == '__main__':
    unittest.main()
