import unittest

import numpy as np
from pykalman import KalmanFilter as pykalman_kf

from models.state_space.state_space_wrapper import StateSpace
from models.state_space.kf_utils import KalmanFilter

# Linear system with low noise should achieve near-perfect reconstruction
R2_THRESHOLD_FILTER = 0.99


class TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(123456)
        cls.n = 5
        cls.d = 2
        cls.F = 0.9 * np.eye(cls.d)
        cls.F[0, 1] = -0.5
        cls.H = cls.rng.normal(size=(cls.n, cls.d))
        cls.Q = np.eye(cls.d)
        cls.R = 0.1 * np.eye(cls.n)

    def _gen_values(self, n_obs: int = 100, perc_missing: float = None):
        """
        x_{t+1}  = F x_{t} + N(0, Q)
        y_{t}    = H x_{t} + N(0, R)
        """
        x_t = np.zeros((n_obs, self.d))
        for t in range(n_obs):
            x_t[t, :] = self.F @ x_t[t - 1, :] + np.linalg.cholesky(
                self.Q
            ) @ self.rng.normal(size=(self.d,))
        y_t = (
            x_t @ self.H.T
            + self.rng.normal(size=(n_obs, self.n)) @ np.linalg.cholesky(self.R).T
        )
        if perc_missing:
            n_missing = int(n_obs * perc_missing)
            flat_idx = self.rng.choice(n_obs * self.n, size=n_missing, replace=False)
            rows = flat_idx // self.n
            cols = flat_idx % self.n
            y_t[rows, cols] = np.nan
        return y_t, x_t


class TestKalmanFilter(TestBase):

    def test_filter(self):
        """
        Given a simulated LGSSM, check filtered states are close to extracted ones and the same of PyKalman
        implementation.
        """
        y_t, x_t = self._gen_values()
        kalman_filter = pykalman_kf(
            transition_matrices=self.F,
            observation_matrices=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
        )
        kalman_filter_tf = KalmanFilter(
            transition_map=self.F,
            observation_map=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=np.zeros(self.d),
            P0=np.eye(self.d),
        )
        hat_mod_x_t = kalman_filter_tf.filter(y_t)[0]
        r2 = 1 - np.sum(np.power(hat_mod_x_t - x_t, 2)) / np.sum(np.power(x_t, 2))
        self.assertGreater(r2, R2_THRESHOLD_FILTER)
        hat_x_t = kalman_filter.filter(y_t)[0]
        np.testing.assert_allclose(hat_x_t, hat_mod_x_t, rtol=1e-5)

    def test_smooth(self):
        """
        Same as test_filter but with smoothing
        """
        y_t, x_t = self._gen_values()
        kalman_filter = pykalman_kf(
            transition_matrices=self.F,
            observation_matrices=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
        )
        kalman_filter_tf = KalmanFilter(
            transition_map=self.F,
            observation_map=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=np.zeros(self.d),
            P0=np.eye(self.d),
        )
        hat_mod_x_t = kalman_filter_tf.smooth(y_t)[0]
        r2 = 1 - np.sum(np.power(hat_mod_x_t - x_t, 2)) / np.sum(np.power(x_t, 2))
        self.assertGreater(r2, R2_THRESHOLD_FILTER)
        hat_x_t = kalman_filter.smooth(y_t)[0]
        np.testing.assert_allclose(hat_x_t, hat_mod_x_t, rtol=1e-5)

    def test_predict(self):
        """
        Comparing predicted values with calculated from smoothed factors
        """
        y_t, _ = self._gen_values()
        kalman_filter_tf = KalmanFilter(
            transition_map=self.F,
            observation_map=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=np.zeros(self.d),
            P0=np.eye(self.d),
        )
        hat_x = kalman_filter_tf.smooth(y_t)[0]
        preds_y = kalman_filter_tf.predict(y_t, steps_ahead=1)[0]
        preds_manual_y = np.zeros_like(preds_y)
        preds_manual_y[0] = np.dot(self.H, hat_x[-1, :])
        preds_manual_y[1] = np.dot(self.H, np.dot(self.F, hat_x[-1, :]))
        np.testing.assert_allclose(preds_y, preds_manual_y, rtol=1e-5)

    def test_fill_na(self):
        """
        Comparing fill-na with calculated from smoothed factors
        """
        y_t, _ = self._gen_values()
        y_t[-1, :2] = np.nan
        kalman_filter_tf = KalmanFilter(
            transition_map=self.F,
            observation_map=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=np.zeros(self.d),
            P0=np.eye(self.d),
        )
        hat_x = kalman_filter_tf.smooth(y_t)[0]
        preds_y = kalman_filter_tf.fill_na(y_t)[0][-1]
        preds_manual_y = np.dot(self.H, hat_x[-1, :])
        np.testing.assert_allclose(preds_y, preds_manual_y, rtol=1e-5)

    def test_filter_with_missing(self):
        """
        Given a simulated LGSSM, now with missing data. Check:
            1. tested version has no missing values on filtered states, while benchmarked one does
            2. r2 on non-missing data is approximately the same
            3. r2 on missing data for tested version is larger than 0
            4. when filling missing data with predicted states on the benchmark version, r2 of tested version is still larger
        """
        y_t, x_t = self._gen_values(perc_missing=0.1)
        kalman_filter = pykalman_kf(
            transition_matrices=self.F,
            observation_matrices=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
        )
        kalman_filter_tf = KalmanFilter(
            transition_map=self.F,
            observation_map=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=np.zeros(self.d),
            P0=np.eye(self.d),
        )
        hat_mod_x_t = kalman_filter_tf.filter(y_t)[0]
        hat_x_t = kalman_filter.filter(y_t)[0]
        # 1. checking modified has no missing values, while original version does
        self.assertEqual(np.sum(np.isnan(hat_mod_x_t)), 0)
        self.assertGreater(np.sum(np.isnan(hat_x_t)), 0)
        # 2. R2 on common points is the same
        r2 = 1 - np.nansum(np.power(hat_x_t - x_t, 2)) / np.sum(np.power(x_t, 2))
        hat_mod_x_t[np.isnan(hat_x_t)] = np.nan
        r2_mod = 1 - np.nansum(np.power(hat_mod_x_t - x_t, 2)) / np.sum(
            np.power(x_t, 2)
        )
        self.assertAlmostEqual(r2_mod, r2, places=10)
        # 3. r2 on missing data for modified is larger than 0
        hat_mod_x_t = kalman_filter_tf.filter(y_t)[0]
        hat_mod_x_t[~np.isnan(hat_x_t)] = np.nan
        r2_mod_missing = 1 - np.nansum(np.power(hat_mod_x_t - x_t, 2)) / np.sum(
            np.power(x_t, 2)
        )
        self.assertGreater(r2_mod_missing, 0)
        # 4. r2 with modified is better than r2 with original and predicted states for missing values.
        hat_mod_x_t = kalman_filter_tf.filter(y_t)[0]
        r2_mod_missing = 1 - np.sum(np.power(hat_mod_x_t - x_t, 2)) / np.sum(
            np.power(x_t, 2)
        )
        hat_next_based_on_mod = np.concatenate(
            [np.zeros((1, self.d)), hat_mod_x_t[:-1] @ self.F.T]
        )
        hat_x_t[np.isnan(hat_x_t)] = hat_next_based_on_mod[np.isnan(hat_x_t)]
        r2 = 1 - np.sum(np.power(hat_x_t - x_t, 2)) / np.sum(np.power(x_t, 2))
        self.assertGreater(r2_mod_missing, r2)


class TestStateSpaceKf(TestBase):
    """
    Testing predict, filter and smooth of the state space wrapper against KalmanFilter tested above.
    """

    @classmethod
    def setUpClass(cls):
        super(TestStateSpaceKf, cls).setUpClass()
        cls.kf = KalmanFilter(
            transition_map=cls.F,
            observation_map=cls.H,
            transition_covariance=cls.Q,
            observation_covariance=cls.R,
            x0=np.zeros(cls.d),
            P0=np.eye(cls.d),
        )
        cls.ssm_kf = StateSpace(
            {"transition_map": cls.F, "transition_covariance": cls.Q},
            {"observation_map": cls.H, "observation_covariance": cls.R},
            np.zeros(cls.n),
            np.ones(cls.n),
            x0=np.zeros(cls.d),
            P0=np.eye(cls.d),
        )

    def test_predict(self):
        y, _ = self._gen_values()
        mean1, cov1 = self.kf.predict(y, steps_ahead=10)
        mean2, cov2 = self.ssm_kf.predict(y, steps_ahead=10)
        np.testing.assert_array_almost_equal(mean1, mean2)
        np.testing.assert_array_almost_equal(cov1, cov2)

    def test_filter(self):
        y, _ = self._gen_values()
        mean1, cov1 = self.kf.filter(y)
        mean2, cov2 = self.ssm_kf.filter(y)
        np.testing.assert_array_almost_equal(mean1, mean2)
        np.testing.assert_array_almost_equal(cov1, cov2)

    def test_smooth(self):
        y, _ = self._gen_values()
        mean1, cov1 = self.kf.smooth(y)
        mean2, cov2 = self.ssm_kf.smooth(y)
        np.testing.assert_array_almost_equal(mean1, mean2)
        np.testing.assert_array_almost_equal(cov1, cov2)


if __name__ == "__main__":
    unittest.main()
