from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union

import tensorflow as tf
from tensorflow import keras
import numpy as np


def _convert_to_tensor(matrix, dtype):
    return (
        tf.convert_to_tensor(matrix, dtype=dtype)
        if not isinstance(matrix, tf.Tensor)
        else matrix
    )


class BaseFilter(ABC):
    def __init__(
        self,
        transition_map: Union[tf.Tensor, np.ndarray, keras.Model],
        observation_map: Union[tf.Tensor, np.ndarray, keras.Model],
        transition_covariance: Union[tf.Tensor, np.ndarray],
        observation_covariance: Union[tf.Tensor, np.ndarray],
        x0: Union[tf.Tensor, np.ndarray],
        P0: Union[tf.Tensor, np.ndarray],
        transition_offsets: Union[tf.Tensor, np.ndarray] = None,
        observation_offsets: Union[tf.Tensor, np.ndarray] = None,
        dtype: Optional[tf.DType] = tf.float64,
    ):
        obs_size = observation_covariance.shape[0]
        state_size = transition_covariance.shape[0]
        self.dtype = dtype
        self._filter_graph = tf.function(
            self._filter_impl,
            input_signature=[
                tf.TensorSpec([None, obs_size], self.dtype),
                tf.TensorSpec([state_size], self.dtype),
                tf.TensorSpec([state_size, state_size], self.dtype),
            ],
        )
        self._smoother_graph = tf.function(
            self._smoother_impl,
            input_signature=[
                tf.TensorSpec([None, state_size], self.dtype),
                tf.TensorSpec([None, state_size, state_size], self.dtype),
                tf.TensorSpec([None, state_size], self.dtype),
                tf.TensorSpec([None, state_size, state_size], self.dtype),
            ],
        )
        self._smoother_withcrosscov_graph = tf.function(
            self._smoother_withcrosscov_impl,
            input_signature=[
                tf.TensorSpec([None, state_size], self.dtype),
                tf.TensorSpec([None, state_size, state_size], self.dtype),
                tf.TensorSpec([None, state_size], self.dtype),
                tf.TensorSpec([None, state_size, state_size], self.dtype),
            ],
        )
        self.obs_size = obs_size
        self._fillna_graph = tf.function(
            self._fillna_impl,
            input_signature=[
                tf.TensorSpec([None, obs_size], self.dtype),
                tf.TensorSpec([state_size], self.dtype),
                tf.TensorSpec([state_size, state_size], self.dtype),
            ],
        )

    @abstractmethod
    def predict_from_state(
        self,
        predicted_state_mean: np.ndarray,
        predicted_state_covariance: np.ndarray,
        steps_ahead: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict observables, from 0 to steps_ahead forecasting.
        """
        pass

    @abstractmethod
    def _get_filter_function(self):
        """
        Get filter function.
        """
        pass

    @abstractmethod
    def _get_smoother_function(self):
        """
        Get smoother function.
        """
        pass

    @abstractmethod
    def _get_fillna_from_state_function(self):
        """
        Get function to fill missing values in the observable space from .
        """
        pass

    def _get_smoother_with_cross_cov(self):
        """
        Get smoother function with cross-covariance.
        """
        pass

    def get_default_initial_state(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.x0, self.P0

    def predict(self, y: np.ndarray, steps_ahead: int):
        """
        Predict observables, from 0 (fill missing) to steps_ahead forecasting horizon
        """
        filtered_state_means, filtered_state_covariances = self.filter(y)
        predicted_state_mean = filtered_state_means[-1]
        predicted_state_covariance = filtered_state_covariances[-1]
        return self.predict_from_state(
            predicted_state_mean, predicted_state_covariance, steps_ahead
        )

    def _filter_impl(
        self, ys: tf.Tensor, x_start: tf.Tensor, P_start: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        xs_pred, Ps_pred, xs_filt, Ps_filt = tf.scan(
            fn=self._get_filter_function(),
            elems=ys,
            initializer=(x_start, P_start, x_start, P_start),
        )
        return xs_pred, Ps_pred, xs_filt, Ps_filt

    def _smoother_impl(
        self,
        xs_pred: tf.Tensor,
        Ps_pred: tf.Tensor,
        xs_filt: tf.Tensor,
        Ps_filt: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        elems = (
            xs_filt[:-1],
            Ps_filt[:-1],
            xs_pred[:-1],
            Ps_pred[:-1],
        )
        xs_smooth, Ps_smooth = tf.scan(
            fn=self._get_smoother_function(),
            elems=elems,
            initializer=(xs_filt[-1], Ps_filt[-1]),
            reverse=True,
        )
        return xs_smooth, Ps_smooth

    def _smoother_withcrosscov_impl(
        self,
        xs_pred: tf.Tensor,
        Ps_pred: tf.Tensor,
        xs_filt: tf.Tensor,
        Ps_filt: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        dummy_cross = tf.zeros_like(Ps_filt[0])
        elems = (xs_filt[:-1], Ps_filt[:-1], xs_pred[:-1], Ps_pred[:-1])
        xs_smooth_vals, Ps_smooth_vals, Ps_cross_vals = tf.scan(
            fn=self._get_smoother_with_cross_cov(),
            elems=elems,
            initializer=(xs_filt[-1], Ps_filt[-1], dummy_cross),
            reverse=True,
        )
        return xs_smooth_vals, Ps_smooth_vals, Ps_cross_vals

    def _fillna_impl(
        self, ys: tf.Tensor, x_start: tf.Tensor, P_start: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        xs_pred, Ps_pred, xs_filt, Ps_filt = self._filter_impl(ys, x_start, P_start)
        xs_smooth, Ps_smooth = self._smoother_impl(xs_pred, Ps_pred, xs_filt, Ps_filt)
        states_mean = tf.concat([xs_smooth, xs_filt[-1:]], axis=0)
        states_cov = tf.concat([Ps_smooth, Ps_filt[-1:]], axis=0)
        f, use_tf_map = self._get_fillna_from_state_function()
        if use_tf_map:  # UKF / MUKF: per-step loop
            y_mean, y_cov = tf.map_fn(
                fn=f,
                elems=(states_mean, states_cov),
                fn_output_signature=(
                    tf.TensorSpec((self.obs_size,), self.dtype),
                    tf.TensorSpec((self.obs_size, self.obs_size), self.dtype),
                ),
            )
        else:  # KF: already vectorized einsum
            y_mean, y_cov = f(states_mean, states_cov)
        return y_mean, y_cov

    def filter(
        self,
        y: np.ndarray,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Unscented Kalman Filter
        Args:
            y: observable variables
            x0: initial state mean (optional, if not provided default starting state is used), this is used as the
                predicted state for the first observation in y.
            P0: initial state covariance (optional, if not provided default starting covariance is used)

        Returns:
            xs_filt: filtered states mean
            Ps_filt: filtered states covariance
        """
        x, P = self.get_default_initial_state()
        x_start = x if x0 is None else tf.convert_to_tensor(x0, dtype=self.dtype)
        P_start = P if P0 is None else tf.convert_to_tensor(P0, dtype=self.dtype)

        y_as_tf = tf.convert_to_tensor(y, dtype=self.dtype)
        _, _, xs_filt, Ps_filt = self._filter_graph(y_as_tf, x_start, P_start)
        # xs_filt: (T, dim_x)
        # Ps_filt: (T, dim_x, dim_x)
        return xs_filt.numpy(), Ps_filt.numpy()

    def smooth(
        self,
        y: np.ndarray,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Unscented Kalman Smoother
        Args:
            y: observable variables
            x0: initial state mean (optional, if not provided default starting state is used), this is used as the
                predicted state for the first observation in y.
            P0: initial state covariance (optional, if not provided default starting covariance is used)

        Returns:
            xs_smooth: smoothed states mean
            Ps_smooth: smoothed states covariance
        """
        x, P = self.get_default_initial_state()
        x_start = x if x0 is None else tf.convert_to_tensor(x0, dtype=self.dtype)
        P_start = P if P0 is None else tf.convert_to_tensor(P0, dtype=self.dtype)

        y_as_tf = tf.convert_to_tensor(y, dtype=self.dtype)

        xs_pred, Ps_pred, xs_filt, Ps_filt = self._filter_graph(
            y_as_tf, x_start, P_start
        )
        xs_smooth, Ps_smooth = self._smoother_graph(xs_pred, Ps_pred, xs_filt, Ps_filt)
        xs_smooth = tf.concat([xs_smooth, xs_filt[-1:]], axis=0)
        Ps_smooth = tf.concat([Ps_smooth, Ps_filt[-1:]], axis=0)

        # xs_smooth: (T, dim_x)
        # Ps_smooth: (T, dim_x, dim_x)
        return xs_smooth.numpy(), Ps_smooth.numpy()

    def smooth_with_cross_cov(
        self,
        y: np.ndarray,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Unscented Kalman Smoother
        Args:
            y: observable variables
            x0: initial state mean (optional, if not provided default starting state is used), this is used as the
                predicted state for the first observation in y.
            P0: initial state covariance (optional, if not provided default starting covariance is used)

        Returns:
            xs_smooth: smoothed states mean
            Ps_smooth: smoothed states covariance
            Ps_cross: np.ndarray (T-1, state_size, state_size)
                Ps_cross[t] = Cov(x_t, x_{t+1} | y_{1:T})
        """
        x, P = self.get_default_initial_state()
        x_start = x if x0 is None else tf.convert_to_tensor(x0, dtype=self.dtype)
        P_start = P if P0 is None else tf.convert_to_tensor(P0, dtype=self.dtype)

        y_tf = tf.convert_to_tensor(y, dtype=self.dtype)

        xs_pred, Ps_pred, xs_filt, Ps_filt = self._filter_graph(y_tf, x_start, P_start)

        xs_smooth_vals, Ps_smooth_vals, Ps_cross_vals = (
            self._smoother_withcrosscov_graph(xs_pred, Ps_pred, xs_filt, Ps_filt)
        )

        xs_smooth = tf.concat([xs_smooth_vals, xs_filt[-1:]], axis=0)
        Ps_smooth = tf.concat([Ps_smooth_vals, Ps_filt[-1:]], axis=0)

        return xs_smooth.numpy(), Ps_smooth.numpy(), Ps_cross_vals.numpy()

    def fill_na(
        self,
        y: np.ndarray,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        keep_non_missing: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill missing values in the observables
        Args:
            y: observable variables
            x0: initial state mean (optional, if not provided default starting state is used), this is used as the
                predicted state for the first observation in y.
            P0: initial state covariance (optional, if not provided default starting covariance is used)
            keep_non_missing: whether to keep non-missing values in the mean vector equal to the realized values

        Returns:
            y_mean: smoothed sates implied observable mean
            y_cov: smoothed states implied observable covariance
        """
        x, P = self.get_default_initial_state()
        x_start = x if x0 is None else tf.convert_to_tensor(x0, dtype=self.dtype)
        P_start = P if P0 is None else tf.convert_to_tensor(P0, dtype=self.dtype)
        y_as_tf = tf.convert_to_tensor(y, dtype=self.dtype)
        y_mean, y_cov = self._fillna_graph(y_as_tf, x_start, P_start)
        y_mean = y_mean.numpy()
        y_cov = y_cov.numpy()
        if keep_non_missing:
            not_nan = ~np.isnan(y)
            y_mean[not_nan] = y[not_nan]
        return y_mean, y_cov
