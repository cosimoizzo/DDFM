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
from models.state_space.ukf_utils import AdditiveUKF

TF_DTYPE = tf.float64


class TestAdditiveUKF(TestBase):
    """
    Benchmarking AdditiveUKF to standard KF in the linear case.
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
        ukf = AdditiveUKF(
            transition_map=self._make_linear_keras_model(
                self.F.shape[0], self.F.shape[0], self.F.T
            ),
            observation_map=self._make_linear_keras_model(
                self.F.shape[0], self.H.shape[0], self.H.T
            ),
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
            dtype=TF_DTYPE,
        )
        hat_mod_x_t = kalman_filter.filter(y_t)[0]
        hat_ukf_x_t = ukf.filter(y_t)[0]
        r2 = 1 - np.sum(np.power(hat_ukf_x_t - hat_mod_x_t, 2)) / np.sum(
            np.power(hat_mod_x_t, 2)
        )
        self.assertGreater(r2, R2_THRESHOLD_FILTER)

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
        ukf = AdditiveUKF(
            transition_map=self._make_linear_keras_model(
                self.F.shape[0], self.F.shape[0], self.F.T
            ),
            observation_map=self._make_linear_keras_model(
                self.F.shape[0], self.H.shape[0], self.H.T
            ),
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
            dtype=TF_DTYPE,
        )
        hat_mod_x_t = kalman_filter.smooth(y_t)[0]
        hat_ukf_x_t = ukf.smooth(y_t)[0]
        r2 = 1 - np.sum(np.power(hat_ukf_x_t - hat_mod_x_t, 2)) / np.sum(
            np.power(hat_mod_x_t, 2)
        )
        self.assertGreater(r2, R2_THRESHOLD_FILTER)

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
        ukf = AdditiveUKF(
            transition_map=self._make_linear_keras_model(
                self.F.shape[0], self.F.shape[0], self.F.T
            ),
            observation_map=self._make_linear_keras_model(
                self.F.shape[0], self.H.shape[0], self.H.T
            ),
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
            dtype=TF_DTYPE,
        )
        hat_mod_x_t = kalman_filter.smooth(y_t)[0]
        hat_ukf_x_t = ukf.smooth(y_t)[0]
        r2 = 1 - np.sum(np.power(hat_ukf_x_t - hat_mod_x_t, 2)) / np.sum(
            np.power(hat_mod_x_t, 2)
        )
        self.assertGreater(r2, R2_THRESHOLD_FILTER)

    def test_predict(self):
        """
        Comparing predicted values with calculated from smoothed factors
        """
        y_t, x_t = self._gen_values()
        ukf = AdditiveUKF(
            transition_map=self._make_linear_keras_model(
                self.F.shape[0], self.F.shape[0], self.F.T
            ),
            observation_map=self._make_linear_keras_model(
                self.F.shape[0], self.H.shape[0], self.H.T
            ),
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
            dtype=TF_DTYPE,
        )
        hat_x = ukf.smooth(y_t)[0]
        preds_y = ukf.predict(y_t, steps_ahead=1)[0]
        preds_manual_y = np.zeros_like(preds_y)
        preds_manual_y[0] = np.dot(self.H, hat_x[-1, :])
        preds_manual_y[1] = np.dot(self.H, np.dot(self.F, hat_x[-1, :]))
        np.testing.assert_allclose(preds_y, preds_manual_y, rtol=1e-5)

    def test_fill_na(self):
        """
        Comparing fill-na with calculated from smoothed factors
        """
        y_t, x_t = self._gen_values()
        y_t[-1, :2] = np.nan
        ukf = AdditiveUKF(
            transition_map=self._make_linear_keras_model(
                self.F.shape[0], self.F.shape[0], self.F.T
            ),
            observation_map=self._make_linear_keras_model(
                self.F.shape[0], self.H.shape[0], self.H.T
            ),
            transition_covariance=self.Q,
            observation_covariance=self.R,
            x0=x_t[0],
            P0=np.cov(x_t, rowvar=False),
            dtype=TF_DTYPE,
        )
        hat_x = ukf.smooth(y_t)[0]
        preds_y = ukf.fill_na(y_t)[0][-1]
        preds_manual_y = np.dot(self.H, hat_x[-1, :])
        np.testing.assert_allclose(preds_y, preds_manual_y, rtol=1e-5)


class TestStateSpaceUkf(TestBase):
    """
    Testing predict, filter and smooth of the state space wrapper against Ukf tested separately.
    """

    @classmethod
    def setUpClass(cls):
        super(TestStateSpaceUkf, cls).setUpClass()
        cls.ukf = AdditiveUKF(
            transition_map=TestAdditiveUKF._make_linear_keras_model(
                cls.F.shape[0], cls.F.shape[0], cls.F.T
            ),
            observation_map=TestAdditiveUKF._make_linear_keras_model(
                cls.F.shape[0], cls.H.shape[0], cls.H.T
            ),
            transition_covariance=cls.Q,
            observation_covariance=cls.R,
            x0=np.zeros(cls.Q.shape[0]),
            P0=np.eye(cls.Q.shape[0]),
            dtype=TF_DTYPE,
        )
        cls.ssm_ukf = StateSpace(
            {
                "transition_map": TestAdditiveUKF._make_linear_keras_model(
                    cls.F.shape[0], cls.F.shape[0], cls.F.T
                ),
                "transition_covariance": cls.Q,
            },
            {
                "observation_map": TestAdditiveUKF._make_linear_keras_model(
                    cls.F.shape[0], cls.H.shape[0], cls.H.T
                ),
                "observation_covariance": cls.R,
            },
            np.zeros(cls.n),
            np.ones(cls.n),
            x0=np.zeros(cls.Q.shape[0]),
            P0=np.eye(cls.Q.shape[0]),
            filter_type=FilterType.UnscentedKalmanFilter,
        )

    def test_predict(self):
        y, _ = self._gen_values()
        mean1, cov1 = self.ukf.predict(y, steps_ahead=10)
        mean2, cov2 = self.ssm_ukf.predict(y, steps_ahead=10)
        np.testing.assert_array_almost_equal(mean1, mean2)
        np.testing.assert_array_almost_equal(cov1, cov2)

    def test_filter(self):
        y, _ = self._gen_values()
        mean1, cov1 = self.ukf.filter(y)
        mean2, cov2 = self.ssm_ukf.filter(y)
        np.testing.assert_array_almost_equal(mean1, mean2)
        np.testing.assert_array_almost_equal(cov1, cov2)

    def test_smooth(self):
        y, _ = self._gen_values()
        mean1, cov1 = self.ukf.smooth(y)
        mean2, cov2 = self.ssm_ukf.smooth(y)
        np.testing.assert_array_almost_equal(mean1, mean2)
        np.testing.assert_array_almost_equal(cov1, cov2)


if __name__ == "__main__":
    unittest.main()
