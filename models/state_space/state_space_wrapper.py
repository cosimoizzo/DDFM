from typing import Tuple, Optional
from enum import StrEnum

import numpy as np
import tensorflow as tf

from models.state_space.kf_utils import KalmanFilter
from models.state_space.ukf_utils import AdditiveUKF


class FilterType(StrEnum):
    KalmanFilter = "KalmanFilter"
    UnscentedKalmanFilter = "UnscentedKalmanFilter"


class StateSpace:
    """
    State-space model wrapper.
    """

    def __init__(
        self,
        transition_params: dict,
        measurement_params: dict,
        mean_y: np.ndarray = None,
        sigma_y: np.ndarray = None,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
        filter_type: FilterType = FilterType.KalmanFilter,
        dtype: Optional[tf.DType] = tf.float64,
    ):
        """
        The init method will build the state space model according to the selected filter.
        Args:
            transition_params: parameters of the transition equation
            measurement_params: parameters of the measurement equation
            mean_y: mean of the measurement variable
            sigma_y: standard deviation of the measurement variable
            x0: default starting values for the state mean
            P0: default starting values for the state covariance
            filter_type: the type of filter selected

        """
        super().__init__()
        self.mean_y = mean_y
        self.sigma_y = sigma_y
        self.x0 = x0
        self.P0 = P0
        # init parameters of the state space to None
        self.observation_map = None
        self.observation_covariance = None
        self.transition_map = None
        self.transition_covariance = None
        self.observation_offsets = None
        self.ssm_repr = None
        self.dtype = dtype
        if filter_type == FilterType.KalmanFilter:
            # build a linear gaussian state-space model
            self._build_lgssm(transition_params, measurement_params)
        elif filter_type == FilterType.UnscentedKalmanFilter:
            self._build_ukf(transition_params, measurement_params)
        else:
            raise NotImplementedError("{} not implemented".format(filter_type))

    def _scale_data(self, y: np.ndarray) -> np.ndarray:
        y_cpy = y.copy()
        if self.mean_y is not None:
            y_cpy -= self.mean_y
        if self.sigma_y is not None:
            y_cpy /= self.sigma_y
            # make dimensions consistent
        if y_cpy.ndim == 1:
            y_cpy = np.reshape(y_cpy, (1, y_cpy.shape[0]))
        return y_cpy

    def _build_lgssm(self, transition_params: dict, measurement_params: dict) -> None:
        """
        Build a linear gaussian state space model of the following form:
            measurement: y_t = b + H x_t + v_t; v_t ∼ N(0, R)
            transition: x_t = F x_t-1 + w_t; w_t ∼ N(0, Q)
        Args:
            transition_params: parameters of the transition equation
            measurement_params: parameters of the measurement equation

        Returns:
            None, it updates the class attributes.
        """

        self.observation_map, self.observation_covariance = (
            measurement_params["observation_map"],
            measurement_params["observation_covariance"],
        )
        self.observation_offsets = measurement_params.get("observation_offsets", None)
        self.transition_map, self.transition_covariance = (
            transition_params["transition_map"],
            transition_params["transition_covariance"],
        )
        self.ssm_repr = KalmanFilter(
            transition_map=self.transition_map,
            observation_map=self.observation_map,
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance,
            x0=self.x0,
            P0=self.P0,
            observation_offsets=self.observation_offsets,
            dtype=self.dtype,
        )

    def _build_ukf(self, transition_params: dict, measurement_params: dict) -> None:
        """
        Build a state space model of the following form:
            measurement: y_t = H(x_t) + v_t; v_t ∼ N(0, R)
            transition: x_t = F(x_t-1) + w_t; w_t ∼ N(0, Q)
        Args:
            transition_params: parameters of the transition equation
            measurement_params: parameters of the measurement equation

        Returns:
            None, it updates the class attributes.
        """

        self.observation_map, self.observation_covariance = (
            measurement_params["observation_map"],
            measurement_params["observation_covariance"],
        )
        self.observation_offsets = None
        self.transition_map, self.transition_covariance = (
            transition_params["transition_map"],
            transition_params["transition_covariance"],
        )
        self.ssm_repr = AdditiveUKF(
            transition_map=self.transition_map,
            observation_map=self.observation_map,
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance,
            x0=self.x0,
            P0=self.P0,
            dtype=self.dtype,
        )

    def predict(self, y: np.ndarray, steps_ahead: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict observables up to steps_ahead forecasting horizon
        Args:
            y: observable variable
            steps_ahead: the maximum forecasting horizon

        Returns:
            mean and covariance over the forecasting horizon
        """
        y_cpy = self._scale_data(y)
        mean, cov = self.ssm_repr.predict(y_cpy, steps_ahead=steps_ahead)
        return self._undo_scale_data(mean, cov)

    def predict_from_state(
        self, state_mean: np.ndarray, state_cov: np.ndarray, steps_ahead: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean, cov = self.ssm_repr.predict_from_state(
            state_mean, state_cov, steps_ahead=steps_ahead
        )
        return self._undo_scale_data(mean, cov)

    def filter(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        State Space filtering step
        Args:
            y: observable realised values

        Returns:
            filtered states and variance-covariance matrix
        """
        y_cpy = self._scale_data(y)
        filtered_state_means, filtered_state_covariances = self.ssm_repr.filter(y_cpy)
        return filtered_state_means, filtered_state_covariances

    def smooth(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        State Space smoothing step
        Args:
            y: observable realised values

        Returns:
            smoothed states and variance-covariance matrix
        """
        y_cpy = self._scale_data(y)
        smoothed_state_means, smoothed_state_covariances = self.ssm_repr.smooth(y_cpy)
        return smoothed_state_means, smoothed_state_covariances

    def _undo_scale_data(
        self, mean: np.ndarray, cov: np.ndarray, round_to: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        if round_to:
            mean = np.round(mean, round_to)
            cov = np.round(cov, round_to)
        if self.sigma_y is not None:
            mean *= self.sigma_y[None, :]
            cov *= np.outer(self.sigma_y, self.sigma_y)[None, :, :]
        if self.mean_y is not None:
            mean += self.mean_y[None, :]
        return mean, cov
