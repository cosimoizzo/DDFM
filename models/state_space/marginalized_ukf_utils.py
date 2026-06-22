from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.state_space.base_filter import BaseFilter, _convert_to_tensor
from models.state_space.kf_utils import _get_linear_smoother_function
from models.state_space.ukf_utils import _compute_weights


class MarginalizedUKF(BaseFilter):
    """
    Rao-Blackwellized UKF for state-space models with linear/nonlinear split:

        z_t = [x_t; ε_t]      x: nonlinear sub-state (e.g. common factors)
                              ε: linear sub-state    (e.g. idio AR(1))
        z_t = A · z_{t-1} + w_t,    A = blkdiag(A_x, A_ε),   w ~ N(0, Q)
        y_t = h([x_t; ε_t]) + v_t,                          v ~ N(0, R), R > 0

    Required: h must be linear in the ε-block (e.g. additive y = h(x) + B ε).
    Sigma points are generated only on the x-block (small); ε is held at its
    conditional mean. The linear part is marginalized analytically (RB-UKF).
    """

    def __init__(
        self,
        transition_map: Union[tf.Tensor, np.ndarray],  # JOINT A, block-diag
        observation_map: keras.Model,  # joint input [x; ε] -> y
        linear_observation_map: Union[
            tf.Tensor, np.ndarray
        ],  # B, the linear part ε of the observation map
        transition_covariance: Union[tf.Tensor, np.ndarray],  # JOINT Q
        observation_covariance: Union[tf.Tensor, np.ndarray],  # R, must be > 0
        x0: Union[tf.Tensor, np.ndarray],  # JOINT x0
        P0: Union[tf.Tensor, np.ndarray],  # JOINT P0
        alpha: Optional[float] = None,
        kappa: Optional[float] = 0.0,
        beta: Optional[float] = 2.0,
        dtype: Optional[tf.DType] = tf.float64,
    ):
        linear_start = transition_covariance.shape[0] - linear_observation_map.shape[1]
        if not isinstance(observation_map, keras.Model):
            raise ValueError("observation_map must be keras.Model")

        self.dtype = dtype
        self.transition_map = _convert_to_tensor(transition_map, self.dtype)
        self.transition_covariance = _convert_to_tensor(
            transition_covariance, self.dtype
        )
        self.observation_map = observation_map
        self.observation_covariance = _convert_to_tensor(
            observation_covariance, self.dtype
        )
        self.linear_observation_map = _convert_to_tensor(
            linear_observation_map, self.dtype
        )
        self.x0 = _convert_to_tensor(x0, self.dtype)
        self.P0 = _convert_to_tensor(P0, self.dtype)

        # Block dimensions
        self.linear_start = linear_start
        self.n_joint = self.x0.shape[0]
        self.n_x = linear_start
        self.n_eps = self.n_joint - self.n_x

        # Sigma point machinery on n_x dimension (NOT n_joint)
        if alpha is None:
            t = 1.0 / (self.n_x + 1)
            alpha = 1.0 / tf.sqrt(1.0 - t)
        self.alpha = tf.cast(alpha, dtype=dtype)
        self.kappa = tf.cast(kappa, dtype=dtype)
        self.beta = tf.cast(beta, dtype=dtype)
        self.L = tf.cast(self.n_x, dtype=dtype)
        self.lamb = tf.cast((alpha**2) * (self.n_x + kappa) - self.n_x, dtype=dtype)
        self.wm, self.wc = _compute_weights(
            self.L, self.lamb, self.n_x, self.alpha, self.beta
        )

    def _sigma_points_x(self, x_block: tf.Tensor, chol_Pxx: tf.Tensor):
        scaler = tf.transpose(chol_Pxx) * tf.sqrt(self.L + self.lamb)
        x_rep = tf.tile(x_block[tf.newaxis], [self.n_x, 1])
        pos = x_rep + scaler
        neg = x_rep - scaler
        return tf.concat([x_block[tf.newaxis], pos, neg], axis=0)

    @tf.function
    def _predict_states_one_step(self, x: tf.Tensor, P: tf.Tensor):
        """
        Predict — pure linear KF on joint state.
        """
        A = self.transition_map
        x_pred = tf.linalg.matvec(A, x)
        P_pred = A @ P @ tf.transpose(A) + self.transition_covariance
        P_pred = 0.5 * (P_pred + tf.transpose(P_pred))
        return x_pred, P_pred

    @tf.function
    def _update(self, x_pred: tf.Tensor, P_pred: tf.Tensor, y: tf.Tensor):
        """
        Update — marginalized UKF
        """
        nan_mask = tf.reshape(tf.math.is_nan(y), [-1])
        mask_float = tf.cast(~nan_mask, self.dtype)
        outer_mask = tf.tensordot(mask_float, mask_float, axes=0)

        # Split blocks
        x_block = x_pred[: self.n_x]
        eps_pred = x_pred[self.n_x :]
        P_xx = P_pred[: self.n_x, : self.n_x]
        P_xeps = P_pred[: self.n_x, self.n_x :]
        P_epseps = P_pred[self.n_x :, self.n_x :]

        # Sigma points on x-block (chol_Pxx reused below for Lh solves)
        chol_Pxx = tf.linalg.cholesky(P_xx)
        sigmas_x = self._sigma_points_x(x_block, chol_Pxx)

        # Stack ε_pred (held at mean), pass joint state through observation_map
        eps_rep = tf.tile(eps_pred[tf.newaxis, :], [2 * self.n_x + 1, 1])
        sigmas_joint = tf.concat([sigmas_x, eps_rep], axis=1)
        gammas = self.observation_map(sigmas_joint)

        # Mask NaN columns
        gammas = gammas * mask_float[tf.newaxis, :]
        B_eff = self.linear_observation_map * mask_float[:, tf.newaxis]

        # UKF moments (ε constant cancels in (γ_i − ŷ_h))
        y_pred = tf.reduce_sum(self.wm[:, tf.newaxis] * gammas, axis=0)
        dh = gammas - y_pred[tf.newaxis, :]
        dx = sigmas_x - x_block[tf.newaxis, :]
        P_hh = tf.einsum("i,ij,ik->jk", self.wc, dh, dh)
        P_xh = tf.einsum("i,ij,ik->jk", self.wc, dx, dh)

        # Conditional-Gaussian "Jacobian": Lh = P_xh^T P_xx^{-1}
        A1 = tf.linalg.triangular_solve(chol_Pxx, P_xh, lower=True)
        Lh_T = tf.linalg.triangular_solve(tf.transpose(chol_Pxx), A1, lower=False)
        Lh = tf.transpose(Lh_T)

        # Mask R (Shumway-Stoffer) and P_εε for NaN observations
        diag_R = tf.linalg.diag_part(self.observation_covariance)
        diag_mask = tf.cast(nan_mask, self.dtype)
        tmp_R = self.observation_covariance * outer_mask + tf.linalg.diag(
            diag_R * diag_mask
        )

        # Innovation covariance
        cross = Lh @ P_xeps @ tf.transpose(B_eff)
        Pyy = (
            P_hh
            + cross
            + tf.transpose(cross)
            + B_eff @ P_epseps @ tf.transpose(B_eff)
            + tmp_R
        )

        # Cross-covariance
        P_xy = P_xh + P_xeps @ tf.transpose(B_eff)
        P_epsy = tf.transpose(P_xeps) @ Lh_T + P_epseps @ tf.transpose(B_eff)
        P_zy = tf.concat([P_xy, P_epsy], axis=0)

        # Joint gain via two triangular solves
        chol_Pyy = tf.linalg.cholesky(Pyy)
        A2 = tf.linalg.triangular_solve(chol_Pyy, tf.transpose(P_zy), lower=True)
        Kt = tf.linalg.triangular_solve(tf.transpose(chol_Pyy), A2, lower=False)
        K = tf.transpose(Kt)

        # Mean update
        innov = keras.ops.nan_to_num(y, nan=0.0) - y_pred
        x_upd = x_pred + tf.linalg.matvec(K, innov)

        # Joint covariance update
        P_upd = P_pred - K @ tf.transpose(P_zy)
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

    def _get_smoother_with_cross_cov(self):
        def scan_fn(carry, elems):
            x_s_next, P_s_next, _ = carry
            x_f, P_f, x_p, P_p = elems
            Pxy = P_f @ tf.transpose(self.transition_map)
            G = tf.transpose(tf.linalg.solve(tf.transpose(P_p), tf.transpose(Pxy)))
            x_smooth = x_f + tf.linalg.matvec(G, x_s_next - x_p)
            P_smooth = P_f + G @ (P_s_next - P_p) @ tf.transpose(G)
            P_smooth = 0.5 * (P_smooth + tf.transpose(P_smooth))
            P_cross = G @ P_s_next
            return x_smooth, P_smooth, P_cross

        return scan_fn

    def _get_fillna_from_state_function(self):
        use_tf_map = True

        def map_fn(elem):
            z, P_z = elem
            x_block = z[: self.n_x]
            eps_blk = z[self.n_x :]
            P_xx = P_z[: self.n_x, : self.n_x]
            P_xeps = P_z[: self.n_x, self.n_x :]
            P_epseps = P_z[self.n_x :, self.n_x :]

            chol_Pxx = tf.linalg.cholesky(P_xx)
            sigmas_x = self._sigma_points_x(x_block, chol_Pxx)
            sigmas_joint = tf.concat(
                [sigmas_x, tf.tile(eps_blk[tf.newaxis, :], [2 * self.n_x + 1, 1])],
                axis=1,
            )
            gammas = self.observation_map(sigmas_joint)

            y_pred = tf.reduce_sum(self.wm[:, tf.newaxis] * gammas, axis=0)
            dh = gammas - y_pred[tf.newaxis, :]
            dx = sigmas_x - x_block[tf.newaxis, :]
            P_hh = tf.einsum("i,ij,ik->jk", self.wc, dh, dh)
            P_xh = tf.einsum("i,ij,ik->jk", self.wc, dx, dh)

            A1 = tf.linalg.triangular_solve(chol_Pxx, P_xh, lower=True)
            Lh_T = tf.linalg.triangular_solve(tf.transpose(chol_Pxx), A1, lower=False)
            Lh = tf.transpose(Lh_T)

            B = self.linear_observation_map
            cross = Lh @ P_xeps @ tf.transpose(B)
            S_pred = (
                P_hh
                + cross
                + tf.transpose(cross)
                + B @ P_epseps @ tf.transpose(B)
                + self.observation_covariance
            )
            return y_pred, S_pred

        return map_fn, use_tf_map

    def predict_from_state(
        self,
        predicted_state_mean: np.ndarray,
        predicted_state_covariance: np.ndarray,
        steps_ahead: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert steps_ahead >= 0
        z = tf.convert_to_tensor(predicted_state_mean, dtype=self.dtype)
        P_z = tf.convert_to_tensor(predicted_state_covariance, dtype=self.dtype)

        f_obs, _ = self._get_fillna_from_state_function()
        y_pred, S_pred = f_obs((z, P_z))

        if steps_ahead > 0:

            def scan_fn(carry, _):
                z_c, P_z_c, _, _ = carry
                z_n, P_z_n = self._predict_states_one_step(z_c, P_z_c)
                y_n, S_n = f_obs((z_n, P_z_n))
                return z_n, P_z_n, y_n, S_n

            dummy = tf.zeros([steps_ahead], dtype=self.dtype)
            _, _, y_preds, S_preds = tf.scan(
                fn=scan_fn,
                elems=dummy,
                initializer=(z, P_z, y_pred, S_pred),
            )
            y_pred = tf.concat([y_pred[tf.newaxis], y_preds], axis=0)
            S_pred = tf.concat([S_pred[tf.newaxis], S_preds], axis=0)
        else:
            y_pred = y_pred[tf.newaxis]
            S_pred = S_pred[tf.newaxis]
        return y_pred.numpy(), S_pred.numpy()
