from abc import ABC, abstractmethod
from typing import Tuple, Optional

import tensorflow as tf
import numpy as np


class BaseFilter(ABC):
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

    # @abstractmethod
    def _get_filter_function(self):
        """
        Get filter function.
        """
        pass

    # @abstractmethod
    def _get_smoother_function(self):
        """
        Get smoother function.
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

    def fill_na(self, y: np.ndarray):
        """
        Fill missing values in the observables
        """
        mean, cov = self.predict(y, steps_ahead=0)
        return mean[0, ...], cov[0, ...]

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
        _, _, xs_filt, Ps_filt = tf.scan(
            fn=self._get_filter_function(),
            elems=y_as_tf,
            initializer=(x_start, P_start, x_start, P_start),
        )
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

        xs_pred, Ps_pred, xs_filt, Ps_filt = tf.scan(
            fn=self._get_filter_function(),
            elems=y_as_tf,
            initializer=(x_start, P_start, x_start, P_start),
        )

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

        xs_smooth = tf.concat([xs_smooth, xs_filt[-1:]], axis=0)
        Ps_smooth = tf.concat([Ps_smooth, Ps_filt[-1:]], axis=0)

        # xs_smooth: (T, dim_x)
        # Ps_smooth: (T, dim_x, dim_x)
        return xs_smooth.numpy(), Ps_smooth.numpy()
