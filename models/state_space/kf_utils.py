from typing import Union, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.state_space.base_filter import BaseFilter, _convert_to_tensor


def _get_linear_smoother_function(transition_map: tf.Tensor):
    def scan_fn(carry, elems):
        x_s_next, P_s_next = carry
        x_f, P_f, x_p, P_p = elems

        Pxy = P_f @ tf.transpose(transition_map)
        G = tf.transpose(tf.linalg.solve(tf.transpose(P_p), tf.transpose(Pxy)))

        x_smooth = x_f + tf.linalg.matvec(G, x_s_next - x_p)
        P_smooth = P_f + G @ (P_s_next - P_p) @ tf.transpose(G)
        P_smooth = 0.5 * (P_smooth + tf.transpose(P_smooth))
        return x_smooth, P_smooth

    return scan_fn


class KalmanFilter(BaseFilter):
    """
    Tensorflow implementation of Kalman filter and smoother.
    """

    def __init__(
        self,
        transition_map: Union[tf.Tensor, np.ndarray],
        observation_map: Union[tf.Tensor, np.ndarray],
        transition_covariance: Union[tf.Tensor, np.ndarray],
        observation_covariance: Union[tf.Tensor, np.ndarray],
        x0: Union[tf.Tensor, np.ndarray],
        P0: Union[tf.Tensor, np.ndarray],
        transition_offsets: Union[tf.Tensor, np.ndarray] = None,
        observation_offsets: Union[tf.Tensor, np.ndarray] = None,
        dtype: Optional[tf.DType] = tf.float64,
    ):
        """

        Args:
            transition_map: transition matrix
            observation_map: observation matrix
            transition_covariance: transition covariance matrix
            observation_covariance: observation covariance matrix
            x0: initial kalman state mean
            P0: initial kalman state covariance
            transition_offsets: intercept term of state equation
            observation_offsets: intercept term of observation equation
            dtype:

        """
        super().__init__(
            transition_map,
            observation_map,
            transition_covariance,
            observation_covariance,
            x0,
            P0,
            transition_offsets,
            observation_offsets,
            dtype,
        )
        self.transition_map = _convert_to_tensor(transition_map, self.dtype)
        self.observation_map = _convert_to_tensor(observation_map, self.dtype)
        self.transition_covariance = _convert_to_tensor(
            transition_covariance, self.dtype
        )
        self.observation_covariance = _convert_to_tensor(
            observation_covariance, self.dtype
        )
        if transition_offsets is None:
            self.transition_offsets = tf.zeros(
                transition_covariance.shape[0], dtype=dtype
            )
        else:
            self.transition_offsets = _convert_to_tensor(transition_offsets, self.dtype)
        if observation_offsets is None:
            self.observation_offsets = tf.zeros(
                observation_covariance.shape[0], dtype=dtype
            )
        else:
            self.observation_offsets = _convert_to_tensor(
                observation_offsets, self.dtype
            )
        self.x0 = _convert_to_tensor(x0, self.dtype)
        self.P0 = _convert_to_tensor(P0, self.dtype)
        state_size = self.transition_covariance.shape[-1]
        assert (
            state_size == P0.shape[0] == P0.shape[1] == x0.shape[0]
        ), "Dimensions mismatch."
        self.state_size = state_size

    @tf.function
    def _predict_states_one_step(
        self,
        x: tf.Tensor,
        P: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        State prediction step.
        """
        x_pred = tf.linalg.matvec(self.transition_map, x) + self.transition_offsets
        P_pred = (
            self.transition_map @ P @ tf.transpose(self.transition_map)
            + self.transition_covariance
        )
        P_pred = 0.5 * (P_pred + tf.transpose(P_pred))
        return x_pred, P_pred

    @tf.function
    def _update(
        self, x_pred: tf.Tensor, P_pred: tf.Tensor, y: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Update step.
        """
        nan_mask = tf.math.is_nan(y)
        nan_mask = tf.reshape(nan_mask, [-1])

        y_pred = (
            tf.linalg.matvec(self.observation_map, x_pred) + self.observation_offsets
        )
        tmp_cov = tf.identity(self.observation_covariance)
        tmp_observation_map = tf.identity(self.observation_map)

        # Shumway and Stoffer (2000,1982) applied to KF
        def _apply_nan_mask(tmp_cov, y_pred, tmp_observation_map):
            mask_float = tf.cast(~nan_mask, tmp_cov.dtype)
            outer_mask = tf.tensordot(mask_float, mask_float, axes=0)
            tmp_cov = tmp_cov * outer_mask
            diag = tf.linalg.diag_part(self.observation_covariance)
            diag_mask = tf.cast(nan_mask, tmp_cov.dtype)
            tmp_cov += tf.linalg.diag(diag * diag_mask)
            y_pred = y_pred * mask_float
            tmp_observation_map = tmp_observation_map * mask_float[:, tf.newaxis]
            return tmp_cov, y_pred, tmp_observation_map

        tmp_cov, y_pred, tmp_observation_map = tf.cond(
            tf.reduce_any(nan_mask),
            true_fn=lambda: _apply_nan_mask(tmp_cov, y_pred, tmp_observation_map),
            false_fn=lambda: (tmp_cov, y_pred, tmp_observation_map),
        )
        Pxy = tf.linalg.matmul(tmp_observation_map, P_pred)
        S = tf.linalg.matmul(Pxy, tf.transpose(tmp_observation_map)) + tmp_cov
        K = tf.transpose(tf.linalg.solve(S, Pxy))

        x_upd = x_pred + tf.linalg.matvec(K, keras.ops.nan_to_num(y, nan=0.0) - y_pred)
        I_KH = tf.eye(self.state_size, dtype=self.dtype) - K @ self.observation_map
        # Joseph form
        P_upd = I_KH @ P_pred @ tf.transpose(
            I_KH
        ) + K @ self.observation_covariance @ tf.transpose(K)
        P_upd = 0.5 * (P_upd + tf.transpose(P_upd))
        return x_upd, P_upd

    def _get_filter_function(self):
        def scan_fn(carry, obs):
            x_pred, P_pred, _, _ = carry
            x_filt, P_filt = self._update(x_pred, P_pred, obs)
            x_pred_next, P_pred_next = self._predict_states_one_step(x_filt, P_filt)
            return x_pred_next, P_pred_next, x_filt, P_filt

        return scan_fn

    def _get_smoother_function(self):
        return _get_linear_smoother_function(self.transition_map)

    def _get_fillna_from_state_function(self):
        use_tf_map = False

        def map_fn(state_mean, state_covariance):
            y_pred = (
                tf.linalg.matvec(self.observation_map, state_mean)
                + self.observation_offsets
            )
            S_pred = tf.einsum(
                "ab,tbc,dc->tad",
                self.observation_map,
                state_covariance,
                self.observation_map,
            )
            return y_pred, S_pred

        return map_fn, use_tf_map

    def predict_from_state(
        self,
        predicted_state_mean: np.ndarray,
        predicted_state_covariance: np.ndarray,
        steps_ahead: int,
    ) -> Tuple[np.ndarray, np.ndarray]:

        assert steps_ahead >= 0, "Steps Ahead must be positive."
        predicted_state_mean = tf.convert_to_tensor(
            predicted_state_mean, dtype=self.dtype
        )
        predicted_state_covariance = tf.convert_to_tensor(
            predicted_state_covariance, dtype=self.dtype
        )
        y_pred = (
            tf.linalg.matvec(self.observation_map, predicted_state_mean)
            + self.observation_offsets
        )
        S_pred = (
            self.observation_map
            @ predicted_state_covariance
            @ tf.transpose(self.observation_map)
        )
        # steps ahead
        if steps_ahead > 0:

            def scan_fn(carry, _):
                x, P, _, _ = carry
                x_pred, P_pred = self._predict_states_one_step(x, P)
                y_pred = (
                    tf.linalg.matvec(self.observation_map, x_pred)
                    + self.observation_offsets
                )
                S = self.observation_map @ P_pred @ tf.transpose(self.observation_map)
                return x_pred, P_pred, y_pred, S

            dummy = tf.zeros([steps_ahead], dtype=self.dtype)  # (n_steps,)
            _, _, y_preds, S_preds = tf.scan(
                fn=scan_fn,
                elems=dummy,
                initializer=(
                    tf.convert_to_tensor(predicted_state_mean, dtype=self.dtype),
                    tf.convert_to_tensor(predicted_state_covariance, dtype=self.dtype),
                    y_pred,
                    S_pred,
                ),
            )
            y_pred = tf.concat([y_pred[tf.newaxis], y_preds], axis=0)
            S_pred = tf.concat([S_pred[tf.newaxis], S_preds], axis=0)
        else:
            y_pred = y_pred[tf.newaxis]
            S_pred = S_pred[tf.newaxis]
        return y_pred.numpy(), S_pred.numpy()
