from typing import Union, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.state_space.base_filter import BaseFilter


class AdditiveUKF(BaseFilter):
    """
    Tensorflow implementation of Additive Unscented Kalman filter and smoother.
    Reference:
        Wan, E.A. and Van Der Merwe, R., 2000, October. The unscented Kalman filter for nonlinear estimation.
        In Proceedings of the IEEE 2000 adaptive systems for signal processing, communications, and control symposium
        (Cat. No. 00EX373) (pp. 153-158). Ieee.
        https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    """

    def __init__(
        self,
        transition_map: keras.Model,
        observation_map: keras.Model,
        transition_covariance: Union[tf.Tensor, np.ndarray],
        observation_covariance: Union[tf.Tensor, np.ndarray],
        x0: Union[tf.Tensor, np.ndarray],
        P0: Union[tf.Tensor, np.ndarray],
        alpha: Optional[float] = 0.5,
        kappa: Optional[float] = 0.0,
        beta: Optional[float] = 2.0,
        dtype: Optional[tf.DType] = tf.float64,
    ):
        """

        Args:
            transition_map: transition function
            observation_map: observation function
            transition_covariance: transition covariance matrix
            observation_covariance: observation covariance matrix
            x0: initial kalman state mean
            P0: initial kalman state covariance
            alpha: determines spread of the sigma points
            kappa: secondary scale parameter
            beta: incorporates prior knowledge of the distribution of the latent states (for gaussian 2.0 is optimal)
            dtype:

        """
        if not isinstance(transition_map, keras.Model) or not isinstance(
            observation_map, keras.Model
        ):
            raise ValueError(
                "transition_map and observation_map must be of type keras.Model"
            )
        self.transition_map = transition_map
        self.observation_map = observation_map
        self.dtype = dtype
        self.transition_covariance = (
            tf.convert_to_tensor(transition_covariance, dtype=dtype)
            if not isinstance(transition_covariance, tf.Tensor)
            else transition_covariance
        )
        self.observation_covariance = (
            tf.convert_to_tensor(observation_covariance, dtype=dtype)
            if not isinstance(observation_covariance, tf.Tensor)
            else observation_covariance
        )
        self.x0 = (
            tf.convert_to_tensor(x0, dtype=dtype)
            if not isinstance(x0, tf.Tensor)
            else x0
        )
        self.P0 = (
            tf.convert_to_tensor(P0, dtype=dtype)
            if not isinstance(P0, tf.Tensor)
            else P0
        )
        state_size = self.transition_covariance.shape[-1]
        self.alpha = tf.cast(alpha, dtype=dtype)
        self.kappa = tf.cast(kappa, dtype=dtype)
        self.beta = tf.cast(beta, dtype=dtype)
        assert (
            state_size == P0.shape[0] == P0.shape[1] == x0.shape[0]
        ), "Dimensions mismatch."
        self.state_size = state_size
        self.L = tf.cast(self.state_size, dtype=dtype)
        self.lamb = tf.cast((alpha**2) * (state_size + kappa) - state_size, dtype=dtype)
        self.wm, self.wc = self._compute_weights()

    def get_default_initial_state(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.x0, self.P0

    def _compute_weights(self):
        denom = self.L + self.lamb

        num_sigma = 2 * self.L + 1
        wm = tf.fill([num_sigma], 1.0 / (2.0 * denom))
        wc = tf.fill([num_sigma], 1.0 / (2.0 * denom))

        # Overwrite the 0th weight
        wm0 = self.lamb / denom
        wc0 = self.lamb / denom + (1.0 - self.alpha**2 + self.beta)

        wm = tf.tensor_scatter_nd_update(wm, [[0]], [wm0])
        wc = tf.tensor_scatter_nd_update(wc, [[0]], [wc0])

        return wm, wc

    def _sigma_points(self, x: tf.Tensor, P: tf.Tensor):
        cholP = tf.linalg.cholesky(P)
        scaler = tf.transpose(cholP) * tf.sqrt(self.L + self.lamb)
        x_rep = tf.tile(x[tf.newaxis], [self.state_size, 1])
        pos = x_rep + scaler
        neg = x_rep - scaler
        return tf.concat([x[tf.newaxis], pos, neg], axis=0)

    def _unscented_transform(self, sigmas_points: tf.Tensor, noise_cov: tf.Tensor):
        x = tf.reduce_sum(self.wm[:, tf.newaxis] * sigmas_points, axis=0)
        diff = sigmas_points - x[tf.newaxis, :]
        P = tf.einsum("i,ij,ik->jk", self.wc, diff, diff) + noise_cov
        return x, P

    @tf.function
    def _predict_states_one_step(
        self,
        x: tf.Tensor,
        P: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        UKF state prediction step.
        """
        sigmas = self._sigma_points(x, P)
        sigmas_f = self.transition_map(sigmas)
        x_pred, P_pred = self._unscented_transform(sigmas_f, self.transition_covariance)
        return x_pred, P_pred

    @tf.function
    def _update(
        self, x_pred: tf.Tensor, P_pred: tf.Tensor, sigmas_f: tf.Tensor, y: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        UKF update step.
        """
        nan_mask = tf.math.is_nan(y)
        nan_mask = tf.reshape(nan_mask, [-1])

        sigmas_obs = self.observation_map(sigmas_f)
        tmp_cov = tf.identity(self.observation_covariance)

        # Shumway and Stoffer (2000,1982) applied to UKF
        def _apply_nan_mask(tmp_cov, sigmas_obs):
            mask_float = tf.cast(~nan_mask, tmp_cov.dtype)
            outer_mask = tf.tensordot(mask_float, mask_float, axes=0)
            tmp_cov = tmp_cov * outer_mask
            diag = tf.linalg.diag_part(self.observation_covariance)
            diag_mask = tf.cast(nan_mask, tmp_cov.dtype)
            tmp_cov += tf.linalg.diag(diag * diag_mask)
            sigmas_obs = sigmas_obs * mask_float[tf.newaxis, :]
            return tmp_cov, sigmas_obs

        tmp_cov, sigmas_obs = tf.cond(
            tf.reduce_any(nan_mask),
            true_fn=lambda: _apply_nan_mask(tmp_cov, sigmas_obs),
            false_fn=lambda: (tmp_cov, sigmas_obs),
        )

        y_pred, S = self._unscented_transform(sigmas_obs, tmp_cov)
        dx = sigmas_f - x_pred[tf.newaxis, :]
        dz = sigmas_obs - y_pred[tf.newaxis, :]
        Pxz = tf.einsum("i,ij,ik->jk", self.wc, dx, dz)
        K = tf.linalg.solve(tf.transpose(S), tf.transpose(Pxz))
        K = tf.transpose(K)

        x_upd = x_pred + tf.linalg.matvec(K, keras.ops.nan_to_num(y, nan=0.0) - y_pred)
        P_upd = P_pred - K @ S @ tf.transpose(K)

        return x_upd, P_upd

    def _get_filter_function(self):
        def scan_fn(carry, obs):
            x_pred, P_pred, _, _ = carry
            sigmas_f = self._sigma_points(x_pred, P_pred)
            x_filt, P_filt = self._update(x_pred, P_pred, sigmas_f, obs)
            x_pred_next, P_pred_next = self._predict_states_one_step(x_filt, P_filt)
            return x_pred_next, P_pred_next, x_filt, P_filt

        return scan_fn

    def _get_smoother_function(self):
        def scan_fn(carry, elems):
            x_s_next, P_s_next = carry
            x_f, P_f, x_p, P_p = elems

            sigmas_t = self._sigma_points(x_f, P_f)
            sigmas_ft = self.transition_map(sigmas_t)
            dx = sigmas_t - x_f[tf.newaxis, :]
            dxp = sigmas_ft - x_p[tf.newaxis, :]
            Pxy = tf.einsum("i,ij,ik->jk", self.wc, dx, dxp)
            G = tf.linalg.solve(tf.transpose(P_p), tf.transpose(Pxy))
            G = tf.transpose(G)

            x_smooth = x_f + tf.linalg.matvec(G, x_s_next - x_p)
            P_smooth = P_f + G @ (P_s_next - P_p) @ tf.transpose(G)

            return x_smooth, P_smooth

        return scan_fn

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

        # current prediction
        sigmas_f = self._sigma_points(predicted_state_mean, predicted_state_covariance)
        sigmas_obs = self.observation_map(sigmas_f)
        y_pred, S_pred = self._unscented_transform(
            sigmas_obs, self.observation_covariance
        )
        # steps ahead
        if steps_ahead > 0:

            def scan_fn(carry, _):
                x, P, _, _ = carry
                x_pred, P_pred = self._predict_states_one_step(x, P)
                sigmas_f = self._sigma_points(x_pred, P_pred)
                sigmas_obs = self.observation_map(sigmas_f)
                y_pred, S = self._unscented_transform(
                    sigmas_obs, self.observation_covariance
                )
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
