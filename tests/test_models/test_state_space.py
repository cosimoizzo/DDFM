import unittest

import numpy as np
from pykalman import KalmanFilter

from models.state_space import KalmanFilterMod

class TestKalmanFilterMod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(123456)
        cls.n = 5
        cls.d = 2
        cls.F = 0.9 * np.eye(cls.d)
        cls.H = cls.rng.normal(size=(cls.n, cls.d))
        cls.Q = np.eye(cls.d)
        cls.R = 0.1 * np.eye(cls.n)

    def _gen_values(self, n_obs: int = 100, perc_missing: float = None):
        """
        x_{t+1}   &= F_{t} x_{t} + \\text{Normal}(0, Q_{t}) \\\\
        y_{t}     &= H_{t} x_{t} + \\text{Normal}(0, R_{t})
        """
        x_t = np.zeros((n_obs, self.d))
        for t in range(n_obs):
            x_t[t, :] = self.F @ x_t[t - 1, :] + np.linalg.cholesky(self.Q) @ self.rng.normal(size=(self.d,))
        y_t = x_t @ self.H.T + self.rng.normal(size=(n_obs, self.n)) @ np.linalg.cholesky(self.R).T
        if perc_missing:
            list_missing_loc = set()
            while len(list_missing_loc) < int(n_obs * perc_missing):
                _row, _col = self.rng.choice(n_obs, 1)[0], self.rng.choice(self.n, 1)[0]
                if (_row, _col) not in list_missing_loc:
                    list_missing_loc.add((_row, _col))
                    y_t[_row, _col] = np.nan
        return y_t, x_t

    def test_predict(self):
        raise ValueError("To add test.")

    def test_filter(self):
        """
        Given a simulated LGSSM, check filtered states are close to extracted ones and the same of KalmanFilter.
        """
        y_t, x_t = self._gen_values()
        kalman_filter = KalmanFilter(transition_matrices=self.F, observation_matrices=self.H,
                                     transition_covariance=self.Q, observation_covariance=self.R)
        kalman_filter_mod = KalmanFilterMod(transition_matrices=self.F, observation_matrices=self.H,
                                            transition_covariance=self.Q, observation_covariance=self.R)
        hat_mod_x_t = kalman_filter_mod.filter(y_t)[0]
        r2 = 1 - np.sum(np.pow(hat_mod_x_t - x_t, 2))/np.sum(np.pow(x_t, 2))
        self.assertGreater(r2, 0.99)
        hat_x_t = kalman_filter.filter(y_t)[0]
        np.testing.assert_allclose(hat_x_t, hat_mod_x_t, rtol=1e-5)

    def test_smooth(self):
        """
        Same as test_filter but with smoothing
        """
        y_t, x_t = self._gen_values()
        kalman_filter = KalmanFilter(transition_matrices=self.F, observation_matrices=self.H,
                                     transition_covariance=self.Q, observation_covariance=self.R)
        kalman_filter_mod = KalmanFilterMod(transition_matrices=self.F, observation_matrices=self.H,
                                            transition_covariance=self.Q, observation_covariance=self.R)
        hat_mod_x_t = kalman_filter_mod.smooth(y_t)[0]
        r2 = 1 - np.sum(np.pow(hat_mod_x_t - x_t, 2))/np.sum(np.pow(x_t, 2))
        self.assertGreater(r2, 0.99)
        hat_x_t = kalman_filter.smooth(y_t)[0]
        np.testing.assert_allclose(hat_x_t, hat_mod_x_t, rtol=1e-5)

    def test_filter_with_missing(self):
        """
        Given a simulated LGSSM, now with missing data. Check:
            1. modified has no missing values on filtered states, while original does
            2. r2 on non-missing data is the same
            3. r2 on missing data for modified is larger than 0
            4. when filling missing data with predicted states, r2 of modified is still larger
        """
        y_t, x_t = self._gen_values(perc_missing= 0.1)
        kalman_filter = KalmanFilter(transition_matrices=self.F, observation_matrices=self.H,
                                     transition_covariance=self.Q, observation_covariance=self.R)
        kalman_filter_mod = KalmanFilterMod(transition_matrices=self.F, observation_matrices=self.H,
                                            transition_covariance=self.Q, observation_covariance=self.R)
        hat_mod_x_t = kalman_filter_mod.filter(y_t)[0]
        hat_x_t = kalman_filter.filter(y_t)[0]
        # 1. checking modified has no missing values, while original version does
        self.assertEqual(np.sum(np.isnan(hat_mod_x_t)), 0)
        self.assertGreater(np.sum(np.isnan(hat_x_t)), 0)
        # 2. R2 on common points is the same
        r2 = 1 - np.nansum(np.pow(hat_x_t - x_t, 2)) / np.sum(np.pow(x_t, 2))
        hat_mod_x_t[np.isnan(hat_x_t)] = np.nan
        r2_mod = 1 - np.nansum(np.pow(hat_mod_x_t - x_t, 2)) / np.sum(np.pow(x_t, 2))
        self.assertAlmostEqual(r2_mod, r2, places=10)
        # 3. r2 on missing data for modified is larger than 0
        hat_mod_x_t = kalman_filter_mod.filter(y_t)[0]
        hat_mod_x_t[~np.isnan(hat_x_t)] = np.nan
        r2_mod_missing = 1 - np.nansum(np.pow(hat_mod_x_t - x_t, 2)) / np.sum(np.pow(x_t, 2))
        self.assertGreater(r2_mod_missing, 0)
        # 4. r2 with modified is better than r2 with original and predicted states for missing values.
        hat_mod_x_t = kalman_filter_mod.filter(y_t)[0]
        r2_mod_missing = 1 - np.sum(np.pow(hat_mod_x_t - x_t, 2)) / np.sum(np.pow(x_t, 2))
        hat_next_based_on_mod = np.concatenate([np.zeros((1, self.d)), hat_mod_x_t[:-1] @ self.F.T])
        hat_x_t[np.isnan(hat_x_t)] = hat_next_based_on_mod[np.isnan(hat_x_t)]
        r2 = 1 - np.sum(np.pow(hat_x_t - x_t, 2)) / np.sum(np.pow(x_t, 2))
        self.assertGreater(r2_mod_missing, r2)

    def test_smooth_with_missing(self):
        """
        Same as test_filter_with_missing but with smoothing. This time only checking:
            1. modified has no missing values on filtered states, while original does
            2. r2 is above 0.95
        """
        y_t, x_t = self._gen_values(perc_missing= 0.1)
        kalman_filter = KalmanFilter(transition_matrices=self.F, observation_matrices=self.H,
                                     transition_covariance=self.Q, observation_covariance=self.R)
        kalman_filter_mod = KalmanFilterMod(transition_matrices=self.F, observation_matrices=self.H,
                                            transition_covariance=self.Q, observation_covariance=self.R)
        hat_mod_x_t = kalman_filter_mod.smooth(y_t)[0]
        hat_x_t = kalman_filter.smooth(y_t)[0]
        # 1. checking modified has no missing values, while original version does
        self.assertEqual(np.sum(np.isnan(hat_mod_x_t)), 0)
        self.assertGreater(np.sum(np.isnan(hat_x_t)), 0)
        # 2. R2 on common points is the same
        r2_mod = 1 - np.nansum(np.pow(hat_mod_x_t - x_t, 2)) / np.sum(np.pow(x_t, 2))
        self.assertGreater(r2_mod, 0.95)


if __name__ == '__main__':
    unittest.main()