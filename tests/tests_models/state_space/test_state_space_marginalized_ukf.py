import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tests.tests_models.state_space.test_state_space_kf import (
    TestBase,
    R2_THRESHOLD_FILTER,
)
from models.state_space.state_space_wrapper import StateSpace, FilterType
from models.state_space.kf_utils import KalmanFilter
from models.state_space.marginalized_ukf_utils import MarginalizedUKF

TF_DTYPE = tf.float64


class TestMarginalizedUKF(TestBase):
    """
    Benchmarking MarginalizedUKF to standard KF in the linear case.
    """

    @staticmethod
    def _make_linear_keras_model(in_dim, out_dim, weight_matrix):
        inputs = keras.Input(shape=(in_dim,), dtype=TF_DTYPE)
        outputs = keras.layers.Dense(
            out_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.Constant(weight_matrix),
            dtype=TF_DTYPE,
        )(inputs)
        model = keras.Model(inputs, outputs)
        model.trainable = False
        return model

    def test_filter(self):
        y_t, x_t = self._gen_values()
        kalman_filter = KalmanFilter(
            transition_map=self.F,
            observation_map=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
        )
        m_ukf = MarginalizedUKF(
            transition_map=self.F,
            observation_map=self._make_linear_keras_model(
                self.F.shape[0], self.H.shape[0], self.H.T
            ),
            transition_covariance=self.Q,
            observation_covariance=self.R,
            linear_observation_map=self.H[:, 1:],
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
            dtype=TF_DTYPE,
        )
        hat_mod_x_t, covariance_kf = kalman_filter.filter(y_t)
        hat_m_ukf_x_t, covariance_m_ukf = m_ukf.filter(y_t)
        r2 = 1 - np.sum(np.power(hat_m_ukf_x_t - hat_mod_x_t, 2)) / np.sum(
            np.power(hat_mod_x_t, 2)
        )
        self.assertGreater(r2, R2_THRESHOLD_FILTER)
        np.testing.assert_allclose(covariance_kf, covariance_m_ukf)

    def test_smooth(self):
        y_t, x_t = self._gen_values()
        kalman_filter = KalmanFilter(
            transition_map=self.F,
            observation_map=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
        )
        m_ukf = MarginalizedUKF(
            transition_map=self.F,
            observation_map=self._make_linear_keras_model(
                self.F.shape[0], self.H.shape[0], self.H.T
            ),
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
            linear_observation_map=self.H[:, 1:],
            dtype=TF_DTYPE,
        )
        hat_mod_x_t, covariance_kf = kalman_filter.smooth(y_t)
        hat_m_ukf_x_t, covariance_m_ukf = m_ukf.smooth(y_t)
        r2 = 1 - np.sum(np.power(hat_m_ukf_x_t - hat_mod_x_t, 2)) / np.sum(
            np.power(hat_mod_x_t, 2)
        )
        self.assertGreater(r2, R2_THRESHOLD_FILTER)
        np.testing.assert_allclose(covariance_kf, covariance_m_ukf)

    def test_smooth_with_missing(self):
        y_t, x_t = self._gen_values(perc_missing=0.1)
        kalman_filter = KalmanFilter(
            transition_map=self.F,
            observation_map=self.H,
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
        )
        m_ukf = MarginalizedUKF(
            transition_map=self.F,
            observation_map=self._make_linear_keras_model(
                self.F.shape[0], self.H.shape[0], self.H.T
            ),
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
            linear_observation_map=self.H[:, 1:],
            dtype=TF_DTYPE,
        )
        hat_mod_x_t, covariance_kf = kalman_filter.smooth(y_t)
        hat_m_ukf_x_t, covariance_m_ukf = m_ukf.smooth(y_t)
        r2 = 1 - np.sum(np.power(hat_m_ukf_x_t - hat_mod_x_t, 2)) / np.sum(
            np.power(hat_mod_x_t, 2)
        )
        self.assertGreater(r2, R2_THRESHOLD_FILTER)
        np.testing.assert_allclose(covariance_kf, covariance_m_ukf)

    def test_predict(self):
        """
        Comparing predicted values with calculated from smoothed factors
        """
        y_t, x_t = self._gen_values()
        m_ukf = MarginalizedUKF(
            transition_map=self.F,
            observation_map=self._make_linear_keras_model(
                self.F.shape[0], self.H.shape[0], self.H.T
            ),
            linear_observation_map=self.H[:, 1:],
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
            dtype=TF_DTYPE,
        )
        hat_x, cov_x = m_ukf.smooth(y_t)
        preds_y, cov_pred = m_ukf.predict(y_t, steps_ahead=1)
        preds_manual_y = np.zeros_like(preds_y)
        preds_manual_y[0] = np.dot(self.H, hat_x[-1, :])
        preds_manual_y[1] = np.dot(self.H, np.dot(self.F, hat_x[-1, :]))
        cov_pred_manual = np.zeros_like(cov_pred)
        cov_pred_manual[0] = np.dot(np.dot(self.H, cov_x[-1, ...]), self.H.T) + self.R
        cov_pred_manual[1] = (
            np.dot(
                np.dot(self.H, self.F @ cov_x[-1, ...] @ self.F.T + self.Q), self.H.T
            )
            + self.R
        )
        np.testing.assert_allclose(preds_y, preds_manual_y, rtol=1e-5)
        np.testing.assert_allclose(cov_pred, cov_pred_manual)

    def test_fill_na(self):
        """
        Comparing fill-na with calculated from smoothed factors
        """
        y_t, x_t = self._gen_values()
        y_t[-1, :2] = np.nan
        m_ukf = MarginalizedUKF(
            transition_map=self.F,
            observation_map=self._make_linear_keras_model(
                self.F.shape[0], self.H.shape[0], self.H.T
            ),
            transition_covariance=self.Q,
            observation_covariance=self.R,
            linear_observation_map=self.H[:, 1:],
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
            dtype=TF_DTYPE,
        )
        hat_x = m_ukf.smooth(y_t)[0]
        preds_y = m_ukf.fill_na(y_t)[0][-1]
        preds_manual_y = np.dot(self.H, hat_x[-1, :])
        np.testing.assert_allclose(preds_y, preds_manual_y, rtol=1e-5)


class TestStateSpaceMUkf(TestBase):
    """
    Testing predict, filter and smooth of the state space wrapper against Ukf tested separately.
    """

    @classmethod
    def setUpClass(cls):
        super(TestStateSpaceMUkf, cls).setUpClass()
        cls.m_ukf = MarginalizedUKF(
            transition_map=cls.F,
            observation_map=TestMarginalizedUKF._make_linear_keras_model(
                cls.F.shape[0], cls.H.shape[0], cls.H.T
            ),
            transition_covariance=cls.Q,
            observation_covariance=cls.R,
            x0=np.zeros(cls.Q.shape[0]),
            P0=np.eye(cls.Q.shape[0]),
            linear_observation_map=cls.H[:, 1:],
            dtype=TF_DTYPE,
        )
        cls.ssm_ukf = StateSpace(
            {
                "transition_map": cls.F,
                "transition_covariance": cls.Q,
            },
            {
                "observation_map": TestMarginalizedUKF._make_linear_keras_model(
                    cls.F.shape[0], cls.H.shape[0], cls.H.T
                ),
                "observation_covariance": cls.R,
                "linear_observation_map": cls.H[:, 1:],
            },
            np.zeros(cls.n),
            np.ones(cls.n),
            x0=np.zeros(cls.Q.shape[0]),
            P0=np.eye(cls.Q.shape[0]),
            filter_type=FilterType.Marginalized_UKF,
        )

    def test_predict(self):
        y, _ = self._gen_values()
        mean1, cov1 = self.m_ukf.predict(y, steps_ahead=10)
        mean2, cov2 = self.ssm_ukf.predict(y, steps_ahead=10)
        np.testing.assert_array_almost_equal(mean1, mean2)
        np.testing.assert_array_almost_equal(cov1, cov2)

    def test_filter(self):
        y, _ = self._gen_values()
        mean1, cov1 = self.m_ukf.filter(y)
        mean2, cov2 = self.ssm_ukf.filter(y)
        np.testing.assert_array_almost_equal(mean1, mean2)
        np.testing.assert_array_almost_equal(cov1, cov2)

    def test_smooth(self):
        y, _ = self._gen_values()
        mean1, cov1 = self.m_ukf.smooth(y)
        mean2, cov2 = self.ssm_ukf.smooth(y)
        np.testing.assert_array_almost_equal(mean1, mean2)
        np.testing.assert_array_almost_equal(cov1, cov2)


if __name__ == "__main__":
    unittest.main()
