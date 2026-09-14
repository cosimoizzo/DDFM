import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from models.deep_learning import VARAE, VARVAE, _column_mean_impute
from synthetic_dgp.simulate import SIMULATE


class _DLModelTestBase:
    """Shared test logic for all deep learning factor models."""

    model_class = None
    model_kwargs = {}
    r = 2
    T = 100
    N = 10

    def setUp(self):
        self.sim = SIMULATE(seed=7, r=self.r, n=self.N)
        self.x = self.sim.simulate(self.T, portion_missings=0.2)

    def _make_model(self):
        return self.model_class(r=self.r, **self.model_kwargs)

    def test_get_factors_shape(self):
        model = self._make_model()
        model.fit(pd.DataFrame(self.x))
        f_hat = model.get_factors(pd.DataFrame(self.x))
        self.assertEqual(f_hat.shape, (self.T, self.r))

    def test_get_factors_no_nan(self):
        model = self._make_model()
        model.fit(pd.DataFrame(self.x))
        f_hat = model.get_factors(pd.DataFrame(self.x))
        self.assertFalse(np.any(np.isnan(f_hat)))

    def test_evaluate_returns_float(self):
        model = self._make_model()
        model.fit(pd.DataFrame(self.x))
        f_hat = model.get_factors(pd.DataFrame(self.x))
        score = self.sim.evaluate(f_hat.values)
        self.assertIsInstance(float(score), float)
        self.assertFalse(np.isnan(score))

    def test_fill_na_no_nan(self):
        model = self._make_model()
        model.fit(pd.DataFrame(self.x))
        x_filled = model.fill_na(pd.DataFrame(self.x))
        self.assertEqual(x_filled.shape, self.x.shape)
        self.assertFalse(np.any(np.isnan(x_filled)))

    def test_predict_shape(self):
        model = self._make_model()
        model.fit(pd.DataFrame(self.x))
        steps = 5
        # predict returns (steps_ahead + 1, N): row 0 = h=0 reconstruction, rows 1..steps = forecasts
        x_pred = model.predict(pd.DataFrame(self.x), steps_ahead=steps)
        self.assertEqual(x_pred.shape, (steps + 1, self.N))


class TestVARAE(_DLModelTestBase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = VARAE
        cls.model_kwargs = {"hidden_structure": (16, 8), "epochs": 50}


class TestVARVAE(_DLModelTestBase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = VARVAE
        cls.model_kwargs = {"hidden_structure": (16, 8), "epochs": 50}


class TestVARAENansIters(_DLModelTestBase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = VARAE
        cls.model_kwargs = {"hidden_structure": (16, 8), "epochs": 20, "nan_max_iterations": 10}


class TestVARVAENansIters(_DLModelTestBase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = VARVAE
        cls.model_kwargs = {"hidden_structure": (16, 8), "epochs": 20, "nan_max_iterations": 10}


class TestVARAEArIdio(unittest.TestCase):
    """
    The ar_idio=True path: eps is measured against the reconstruction, phi is a
    length-N vector, leading-missing columns do not produce NaN, and fill_na
    actually applies the AR term.
    """

    @classmethod
    def setUpClass(cls):
        cls.sim = SIMULATE(seed=7, r=2, n=6)
        x = cls.sim.simulate(80, portion_missings=0.2)
        x[0, 0] = np.nan  # leading-missing column (boundary case)
        x[:3, 1] = np.nan  # longer leading gap
        cls.x = x
        cls.model = VARAE(
            r=2, hidden_structure=(8, 4), epochs=30, ar_idio=True, nan_max_iterations=3
        )
        cls.model.fit(pd.DataFrame(x))

    def test_idio_parameters(self):
        phi, var = self.model._phi_idios, self.model._var_idios
        self.assertEqual(phi.shape, (6,))
        self.assertTrue(np.all(np.isfinite(phi)))
        self.assertLessEqual(np.abs(phi).max(), 0.99)
        self.assertTrue(np.all(np.isfinite(var)) and np.all(var >= 0))

    def test_fill_na_no_nan_and_preserves_observed(self):
        filled = self.model.fill_na(pd.DataFrame(self.x))
        obs = ~np.isnan(self.x)
        self.assertFalse(np.any(np.isnan(filled.values)))
        np.testing.assert_allclose(filled.values[obs], self.x[obs], rtol=1e-5, atol=1e-5)

    def test_fill_na_applies_ar_term(self):
        """
        Filled values must differ from the bare reconstruction wherever a prior
        observation exists to carry an idiosyncratic residual forward from.
        """
        m = self.model
        x_std = m._std(pd.DataFrame(self.x))
        nan = np.isnan(x_std)
        x_imp = m._nans_iter(_column_mean_impute(x_std), nan)
        recon = m._autoencoder(tf.constant(x_imp, tf.float32), training=False)[0].numpy()
        bare = x_imp.copy()
        bare[nan] = recon[nan]
        filled_std = m._std(m.fill_na(pd.DataFrame(self.x)))
        has_prior = nan & (np.cumsum(~nan, axis=0) > 0)
        self.assertGreater(np.abs(filled_std[has_prior] - bare[has_prior]).max(), 1e-6)

    def test_predict_shape_and_no_nan(self):
        pred = self.model.predict(pd.DataFrame(self.x), steps_ahead=3)
        self.assertEqual(pred.shape, (4, 6))
        self.assertFalse(np.any(np.isnan(pred.values)))


if __name__ == "__main__":
    unittest.main()
