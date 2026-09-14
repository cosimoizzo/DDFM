import unittest

import numpy as np
import pandas as pd

from models.classical import VARPCA, DFM
from synthetic_dgp.simulate import SIMULATE


def _df(x):
    return pd.DataFrame(x)


class TestVARPCA(unittest.TestCase):
    def setUp(self):
        self.sim = SIMULATE(seed=42, r=2, n=10)
        self.x = self.sim.simulate(200, portion_missings=0.0)

    def test_fit_get_factors_shape(self):
        model = VARPCA(r=2)
        model.fit(_df(self.x))
        f_hat = model.get_factors(_df(self.x))
        self.assertEqual(f_hat.shape, (200, 2))

    def test_evaluate_returns_float(self):
        model = VARPCA(r=2)
        model.fit(_df(self.x))
        f_hat = model.get_factors(_df(self.x))
        score = self.sim.evaluate(f_hat.values)
        self.assertIsInstance(float(score), float)
        self.assertFalse(np.isnan(score))

    def test_fill_na_no_nan(self):
        sim = SIMULATE(seed=42, r=2, n=10)
        x_miss = sim.simulate(200, portion_missings=0.2)
        model = VARPCA(r=2)
        model.fit(_df(x_miss))
        x_filled = model.fill_na(_df(x_miss))
        self.assertEqual(x_filled.shape, x_miss.shape)
        self.assertFalse(np.any(np.isnan(x_filled)))

    def test_fill_na_no_nan_squared_pc(self):
        sim = SIMULATE(seed=42, r=2, n=10)
        x_miss = sim.simulate(200, portion_missings=0.2)
        model = VARPCA(r=2, squared_pc=True)
        model.fit(_df(x_miss))
        x_filled = model.fill_na(_df(x_miss))
        self.assertEqual(x_filled.shape, x_miss.shape)
        self.assertFalse(np.any(np.isnan(x_filled)))

    def test_predict_shape(self):
        model = VARPCA(r=2)
        model.fit(_df(self.x))
        x_pred = model.predict(_df(self.x), steps_ahead=5)
        self.assertEqual(x_pred.shape, (6, 10))

    def test_predict_shape_squared_pc(self):
        model = VARPCA(r=2, squared_pc=True)
        model.fit(_df(self.x))
        x_pred = model.predict(_df(self.x), steps_ahead=5)
        self.assertEqual(x_pred.shape, (6, 10))

    def test_with_missings(self):
        sim = SIMULATE(seed=1, r=2, n=10)
        x_miss = sim.simulate(200, portion_missings=0.2)
        model = VARPCA(r=2)
        model.fit(_df(x_miss))
        f_hat = model.get_factors(_df(x_miss))
        self.assertEqual(f_hat.shape, (200, 2))
        self.assertFalse(np.any(np.isnan(f_hat)))

    def test_with_missings_squared_pc(self):
        sim = SIMULATE(seed=1, r=2, n=10)
        x_miss = sim.simulate(200, portion_missings=0.2)
        model = VARPCA(r=2, squared_pc=True)
        model.fit(_df(x_miss))
        f_hat = model.get_factors(_df(x_miss))
        self.assertEqual(f_hat.shape, (200, 2))
        self.assertFalse(np.any(np.isnan(f_hat)))


class TestDFM(unittest.TestCase):
    def setUp(self):
        self.sim = SIMULATE(seed=42, r=2, n=10)
        self.x = self.sim.simulate(200, portion_missings=0.2)

    def test_fit_get_factors_shape(self):
        model = DFM(r=2)
        model.fit(_df(self.x))
        f_hat = model.get_factors(_df(self.x))
        self.assertEqual(f_hat.shape[1], 2)
        self.assertFalse(np.any(np.isnan(f_hat)))

    def test_fill_na_no_nan(self):
        model = DFM(r=2)
        model.fit(_df(self.x))
        x_filled = model.fill_na(_df(self.x))
        self.assertFalse(np.any(np.isnan(x_filled)))

    def test_evaluate_returns_float(self):
        model = DFM(r=2)
        model.fit(_df(self.x))
        f_hat = model.get_factors(_df(self.x))
        score = self.sim.evaluate(f_hat.values)
        self.assertFalse(np.isnan(score))


class TestVARPCASingleFactor(unittest.TestCase):
    """r=1 routes through the AutoReg wrapper, which used to raise on predict()."""

    def test_predict_runs_and_conditions_on_its_input(self):
        sim = SIMULATE(seed=42, r=1, n=6)
        x = sim.simulate(200, portion_missings=0.0)
        model = VARPCA(r=1, var_lags=2)
        model.fit(_df(x))
        full = model.predict(_df(x), steps_ahead=3)
        half = model.predict(_df(x[:100]), steps_ahead=3)
        self.assertEqual(full.shape, (4, 6))
        self.assertFalse(np.any(np.isnan(full.values)))
        # forecasts must depend on the data passed in, not on the training tail
        self.assertFalse(np.allclose(full.values[1:], half.values[1:]))


class TestDFMPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = SIMULATE(seed=7, r=2, n=6)
        cls.x = cls.sim.simulate(120, portion_missings=0.1)
        cls.model = DFM(r=2)
        cls.model.fit(_df(cls.x[:100]))

    def test_predict_shape_and_no_nan(self):
        pred = self.model.predict(_df(self.x[:100]), steps_ahead=3)
        self.assertEqual(pred.shape, (4, 6))
        self.assertFalse(np.any(np.isnan(pred.values)))

    def test_one_step_ahead_equals_rolling_forecast(self):
        """
        Single filter pass over [history; future] must reproduce, row by row, the
        1-step forecast obtained by re-filtering the growing history at each step.
        """
        T_train = 100
        single = self.model.predict_one_step_ahead(_df(self.x)).values
        for t in range(T_train, self.x.shape[0]):
            rolling = self.model.predict(_df(self.x[:t]), steps_ahead=1).values[1]
            np.testing.assert_allclose(single[t], rolling, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
