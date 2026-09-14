import logging
from itertools import count
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from models.base import FactorModel
from tools.loss_tools import mse_missing
from tools.getters_converters_tools import get_idio
from models.vector_autoregressive import VARAutoencoder, VARLayerClosedForm

logger = logging.getLogger("experiments.deep_learning")

def _column_mean_impute(x: np.ndarray) -> np.ndarray:
    """Return a copy of x with NaN replaced by column means."""
    x_imp = x.copy().astype(float)
    col_means = np.nanmean(x_imp, axis=0)
    nan_rows, nan_cols = np.where(np.isnan(x_imp))
    x_imp[nan_rows, nan_cols] = col_means[nan_cols]
    return x_imp


def _get_model_weights(model):
    """Extract all trainable + non-trainable weights from a Keras model."""
    return [w.numpy() for w in model.weights]


def _set_model_weights(model, weight_list):
    """Restore weights saved by _get_model_weights."""
    for w, val in zip(model.weights, weight_list):
        w.assign(val)


class VARAE(FactorModel):
    """
    Autoencoder with VAR dynamics in the latent space.

    Wraps the existing VARAutoencoder (models/vector_autoregressive.py).
    The encoder maps x (T, N) -> z (T, r); the decoder maps z -> x_recon (T, N).
    VAR(var_order) is fit on the latent factors via closed-form OLS each epoch.

    When nan_max_iterations > 0, missing values are handled via EM-style:
      step 1 (E-style): reconstruct x from current decoder, fill NaN positions.
      step 2 (M-style): retrain the AE on the updated completed data.
    With nan_max_iterations=0, column-mean imputation + masked MSE loss is used.

    When ar_idio = True, then estimate idiosyncratic AR1s at the end.
    """

    def __init__(
        self,
        r: int,
        hidden_structure: Tuple[int] = (16, 4),
        link: str = "relu",
        var_order: int = 1,
        var_loss_weight: float = 1.0,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        nan_max_iterations: int = 0,
        nan_tol: float = 5e-4,
        seeds: Tuple[int] = (1,),
        ar_idio: bool = False,
    ):
        """

        Args:
            r:
            hidden_structure:
            link:
            var_order:
            var_loss_weight:
            epochs:
            learning_rate:
            nan_max_iterations:
            nan_tol:
            seeds:
            ar_idio:
        """
        super().__init__(r)
        self.hidden_structure = hidden_structure
        self.link = link
        self.var_order = var_order
        self.var_loss_weight = var_loss_weight
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.nan_max_iterations = nan_max_iterations
        self.nan_tol = nan_tol
        self.seeds = seeds
        self.ar_idio = ar_idio
        self._var_idios = None
        self._phi_idios = None
        self._autoencoder = None
        self._variables = None

    def _build(self, N: int, seed_start: int) -> None:
        seeds = count(seed_start)
        encoder = keras.Sequential(
            [
                keras.layers.Dense(
                    hid,
                    activation=self.link,
                    input_shape=(N,) if i == 0 else (self.hidden_structure[i-1],),
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=next(seeds)),
                    bias_initializer="zeros",
                ) for i, hid in enumerate(self.hidden_structure)]
            +[
                keras.layers.Dense(self.r,
                                   activation=None,
                                   input_shape=(self.hidden_structure[-1],),
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(seed=next(seeds)),
                                   bias_initializer="zeros",
                                   ),
            ],
            name="encoder",
        )
        reversed_hidden_structure = list(reversed(self.hidden_structure))
        decoder = keras.Sequential(
            [
                keras.layers.Dense(
                    hid,
                    activation=self.link,
                    input_shape=(self.r,) if i == 0 else (reversed_hidden_structure[i - 1],),
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=next(seeds)),
                    bias_initializer="zeros",
                ) for i, hid in enumerate(reversed_hidden_structure)
            ]
            +[
                keras.layers.Dense(N,
                                   activation=None,
                                   input_shape=(reversed_hidden_structure[-1],),
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(
                                       seed=next(seeds)),
                                   bias_initializer="zeros",
                                   ),
            ],
            name="decoder",
        )
        var_layer = VARLayerClosedForm(n_vars=self.r, var_order=self.var_order)
        self._autoencoder = VARAutoencoder(
            encoder=encoder,
            var_layer=var_layer,
            decoder=decoder,
            var_loss_weight=self.var_loss_weight,
            ae_loss=mse_missing,
        )

    def _make_train_step(self, optimizer):
        autoencoder = self._autoencoder

        def step_body(x_t: tf.Tensor, x_tgt: tf.Tensor) -> tf.Tensor:
            with tf.GradientTape() as tape:
                total_loss, _, _ = autoencoder.compute_loss(
                    x_t, x_tgt, with_var_training=True
                )
            grads = tape.gradient(total_loss, autoencoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))
            return total_loss

        @tf.function
        def train_epochs(x_t: tf.Tensor, x_tgt: tf.Tensor, n: tf.Tensor) -> tf.Tensor:
            loss = tf.constant(float("inf"), tf.float32)
            for _ in tf.range(n):  # tf.range, NOT range: AutoGraph -> tf.while_loop
                loss = step_body(x_t, x_tgt)
            return loss

        return train_epochs

    def _train_epochs(
            self, x_imp: np.ndarray, x_target: np.ndarray, train_step
    ) -> float:
        """Run self.epochs gradient steps inside one graph. Returns final loss."""
        x_t = tf.constant(x_imp, dtype=tf.float32)
        x_tgt = tf.constant(x_target, dtype=tf.float32)
        return float(train_step(x_t, x_tgt, tf.constant(self.epochs)))

    def _fit_once(
        self, x: np.ndarray, x_imp: np.ndarray, nan_mask: np.ndarray, N: int, seed_start: int
    ) -> float:
        """Build, train, return final in-sample loss."""
        self._build(N, seed_start)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        optimizer.build(self._autoencoder.trainable_variables)
        train_step = self._make_train_step(optimizer)
        x_imp_local = x_imp.copy()
        if self.nan_max_iterations > 0 and np.any(nan_mask):
            loss, _dif, _iter = float("inf"), self.nan_tol + 1.0, 0
            while _iter < self.nan_max_iterations and _dif > self.nan_tol:
                loss = self._train_epochs(x_imp_local, x, train_step)
                x_t = tf.constant(x_imp_local, dtype=tf.float32)
                x_recon, _, _ = self._autoencoder(x_t, training=False)
                x_recon = x_recon.numpy()
                _dif = np.mean(np.abs(x_imp_local[nan_mask] - x_recon[nan_mask]))
                x_imp_local[nan_mask] = x_recon[nan_mask]
                _iter += 1
        else:
            loss = self._train_epochs(x_imp_local, x, train_step)
        return loss

    def fit(self, data: pd.DataFrame) -> None:
        self._variables = list(data.columns)
        self.mean_data = data.mean().values
        self.sigma_data = data.std().values
        if np.any(self.sigma_data == 0):
            raise ValueError("Some variables have zero variance.")

        x = self._std(data)  # (T, N) standardised
        T, N = x.shape
        nan_mask = np.isnan(x)
        x_imp = _column_mean_impute(x)

        best_loss, best_weights, best_seed = float("inf"), None, None

        n_seeds = len(self.seeds)
        for k, v in enumerate(self.seeds):
            loss = self._fit_once(x, x_imp, nan_mask, N, v)
            logger.info("VARAE init %d/%d: loss=%.6f", k + 1, n_seeds, loss)
            if loss < best_loss:
                best_loss, best_weights, best_seed = loss, _get_model_weights(self._autoencoder), v

        if best_weights is None:
            raise RuntimeError("All seed initialisations produced non-finite loss.")

        if n_seeds > 1:
            self._build(N, best_seed)
            _set_model_weights(self._autoencoder, best_weights)

        if self.ar_idio:
            x_filled = self._nans_iter(x_imp, nan_mask) if np.any(nan_mask) else x_imp
            x_recon = self._autoencoder(tf.constant(x_filled, tf.float32), training=False)[0].numpy()
            eps = x - x_recon
            phi, std_eps, _ = get_idio(eps, ~nan_mask)
            self._phi_idios = phi.diagonal()
            self._var_idios = (1 - self._phi_idios ** 2) * (std_eps ** 2)

        self._fitted = True

    def _nans_iter(self, x_input: np.ndarray, nan_idxs: np.ndarray) -> np.ndarray:
        _dif = self.nan_tol + 1.0
        _iter = 0
        x = x_input.copy()
        while _iter < self.nan_max_iterations and _dif > self.nan_tol:
            x_new = self._autoencoder(x)[0].numpy()
            _dif = np.mean(np.abs(x_new[nan_idxs] - x[nan_idxs]))
            x[nan_idxs] = x_new[nan_idxs]
            _iter += 1
        return x

    def _get_idio_preds(self, x, x_imp, nan_idxs):
        eps = x - x_imp
        n = np.arange(eps.shape[0])[:, None]
        idx = np.where(~nan_idxs, n, 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        last = np.nan_to_num(np.take_along_axis(eps, idx, axis=0), nan=0.0)
        eps_filled = last * (self._phi_idios ** (n - idx))
        return eps_filled

    def get_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[self._variables]
        x_imp = _column_mean_impute(self._std(data))
        nan_idxs = np.isnan(data.values)
        if np.any(nan_idxs):
            x_imp = self._nans_iter(x_imp, nan_idxs)
        z = self._autoencoder.encoder(x_imp, training=False).numpy()  # (T, r)
        cols = [f"f{i + 1}" for i in range(self.r)]
        return pd.DataFrame(z, index=data.index, columns=cols)

    def fill_na(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[self._variables]
        x = self._std(data)
        x_imp = _column_mean_impute(x)
        nan_idxs = np.isnan(data.values)
        if np.any(nan_idxs):
            x_imp = self._nans_iter(x_imp, nan_idxs)
            x_recon_std = self._autoencoder(
                tf.constant(x_imp, dtype=tf.float32), training=False
            )[0].numpy()
            x_imp[nan_idxs] = x_recon_std[nan_idxs]
            if self.ar_idio:
                eps_filled = self._get_idio_preds(x, x_recon_std, nan_idxs)
                x_imp[nan_idxs] += eps_filled[nan_idxs]

        return pd.DataFrame(self._unstd(x_imp), index=data.index, columns=self._variables)

    def predict(self, data: pd.DataFrame, steps_ahead: int) -> pd.DataFrame:
        x = self._std(data)
        nan_x = np.isnan(x)
        z = self.get_factors(data).values  # (T, r)
        # h=0: decode last latent point (standardised space)
        x_imp = self._autoencoder.decoder(
            tf.constant(z, dtype=tf.float32), training=False
        ).numpy()
        if self.ar_idio:
            eps_filled = self._get_idio_preds(x, x_imp, nan_x)
            x_imp[nan_x] += eps_filled[nan_x]
            x_imp[~nan_x] = x[~nan_x]
        rows = [x_imp[-1:]]
        if steps_ahead > 0:
            # h=1..steps_ahead: iterative VAR forecast in latent space then decode
            z_extended = list(z)
            if self.ar_idio:
                eps_extended = list(eps_filled)
            for _ in range(steps_ahead):
                z_seq = tf.constant(np.array(z_extended), dtype=tf.float32)
                z_pred = self._autoencoder.var_layer(z_seq)
                z_extended.append(z_pred[-1].numpy())
                if self.ar_idio:
                    eps_extended.append(eps_extended[-1] * self._phi_idios)
            z_forecast = np.array(z_extended[-steps_ahead:])
            x_forecast_std = self._autoencoder.decoder(
                tf.constant(z_forecast, dtype=tf.float32), training=False
            ).numpy()
            if self.ar_idio:
                x_forecast_std += np.array(eps_extended[-steps_ahead:])
            rows.append(x_forecast_std)
        x_pred_std = np.vstack(rows)  # (steps_ahead + 1, N)
        return pd.DataFrame(
            self._unstd(x_pred_std), index=range(steps_ahead + 1), columns=self._variables
        )


class VARVAE(FactorModel):
    """
    Variational Autoencoder with VAR dynamics in the latent space (VAR-VAE).

    Architecture (similar to Leushuis, 2025):
      - Encoder:  x_t (N,) -> Dense stack -> mu_t (r,), log_var_t (r,)
      - Reparameterization: z_t = mu_t + eps * exp(0.5 * log_var_t), eps ~ N(0,I)
      - VAR prior: mu_hat_t = VAR(var_order)(mu_{1:t})
      - Decoder:  z_t (r,) -> Dense stack -> x_recon_t (N,)

    Objective (bounded -ELBO under the VAR prior):
      Loss = ||y_t - y_hat_t||^2                       (reconstruction, masked)
           + var_loss_weight * (||mu_t - mu_hat_t||^2 + sum_k tr(A_k V_{t-k} A_k'))   (KL mean part: VAR consistency)
           + beta * 1/2 sum_j (exp(lv_j) - lv_j - 1)   (KL variance part, t >= p)
      i.e. the standard -ELBO with KL taken against the VAR prior
      z_t | lags ~ N(mu_hat_t, I); the innovation covariance is normalised to I.
      NB: this deliberately deviates from Eq. (3) of Leushuis (2025), which keeps
      the intractable posterior-gap KL in the maximised objective (evaluated under
      the substitution "true posterior = N(0, I)") and is unbounded: the entropy
      is cancelled by the KL's -log det term, leaving a linear reward for latent
      variance.

    Forecasting (predict):
      Latent moments follow the standard predictive recursion in companion form:
        P_0 = diag(exp(log_var)) of the last p states   (filtering uncertainty)
        m_h = F m_{h-1},  P_h = F P_{h-1} F' + Q_hat    (h = 1..steps_ahead)
      with Q_hat the empirical VAR innovation covariance (OLS residuals of mu).
      At h=1 the propagation term is A diag(exp(log_var_T)) A' — the paper's
      V_hat — plus the innovation injection Q_hat that the paper omits (its
      Eq. (13) implies a noiseless transition; for a stable VAR that makes
      forecast uncertainty vanish with horizon instead of growing to the
      unconditional variance). Observables are decoded by Gaussian cubature /
      MC over N(m_h, P_h).

    Missing data:
      nan_max_iterations=0  -> column-mean imputation + masked reconstruction.
      nan_max_iterations>0  -> EM loop: M-step trains VAE, E-step re-imputes NaN
                               positions from E[decoder(z)].

    Multi-seed initialisation:
      With len(seeds)>1 the model is re-initialised per seed and the init with the
      lowest final training loss is kept.

    Reference:
      Leushuis, R.M. (2025). Probabilistic forecasting with VAR-VAE. Information
      Sciences, 713. https://www.sciencedirect.com/science/article/pii/S0020025525003160
    """

    def __init__(
        self,
        r: int,
        hidden_structure: Tuple[int] = (16, 4),
        link: str = "relu",
        var_order: int = 1,
        var_loss_weight: float = 1.0,
        beta: float = 1.0,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        nan_max_iterations: int = 0,
        nan_tol: float = 5e-4,
        seeds: Tuple[int] = (1,),
        use_cubature: bool = True,
        mc_samples: int = 1000,
    ):
        """

        Args:
            r:
            hidden_structure:
            link:
            var_order:
            var_loss_weight:
            beta:
            epochs:
            learning_rate:
            nan_max_iterations:
            nan_tol:
            seeds:
            use_cubature:
            mc_samples:
        """
        super().__init__(r)
        self.hidden_structure = hidden_structure
        self.link = link
        self.var_order = var_order
        self.var_loss_weight = var_loss_weight
        self.beta = beta
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.nan_max_iterations = nan_max_iterations
        self.nan_tol = nan_tol
        self.seeds = seeds
        self.use_cubature = use_cubature
        self.mc_samples = mc_samples
        self._generator_vae = tf.random.Generator.from_seed(1234)
        self._encoder = None
        self._decoder = None
        self._var_layer = None
        self._variables = None

    def _build(self, N: int, start_seed: int) -> None:
        seeds = count(start_seed)
        inp = keras.Input(shape=(N,))
        h = keras.layers.Dense(self.hidden_structure[0], activation=self.link, bias_initializer="zeros",
                               kernel_initializer=tf.keras.initializers.GlorotNormal(seed=next(seeds)))(inp)
        for c, v in enumerate(self.hidden_structure[1:]):
            h = keras.layers.Dense(v, activation=self.link, bias_initializer="zeros",
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(seed=next(seeds)))(h)

        mu = keras.layers.Dense(self.r, name="mu", bias_initializer="zeros",
                                kernel_initializer=tf.keras.initializers.GlorotNormal(
                                    seed=next(seeds)))(h)
        log_var = keras.layers.Dense(self.r, name="log_var", bias_initializer="zeros",
                                     kernel_initializer=tf.keras.initializers.GlorotNormal(
                                         seed=next(seeds)))(h)
        self._encoder = keras.Model(inp, [mu, log_var], name="encoder")

        reversed_hidden_structure = list(reversed(self.hidden_structure))
        self._decoder = keras.Sequential(
            [
                keras.layers.Dense(
                    hid,
                    activation=self.link,
                    input_shape=(self.r,) if i == 0 else (reversed_hidden_structure[i - 1],),
                    kernel_initializer=tf.keras.initializers.GlorotNormal(
                        seed=next(seeds)),
                    bias_initializer="zeros",
                ) for i, hid in enumerate(reversed_hidden_structure)
            ]
            + [
                keras.layers.Dense(N,
                                   activation=None,
                                   input_shape=(reversed_hidden_structure[-1],),
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(
                                       seed=next(seeds)),
                                   bias_initializer="zeros",
                                   ),
            ],
            name="decoder",
        )

        self._var_layer = VARLayerClosedForm(n_vars=self.r, var_order=self.var_order)

    def _encode(self, x_t: tf.Tensor, training: bool = False):
        return self._encoder(x_t, training=training)

    def _sample(self, mu: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        eps = self._generator_vae.normal(shape=tf.shape(mu))
        return mu + eps * tf.exp(0.5 * log_var)

    def _expected_decode(self, mu, log_var):
        """Compute E[decoder(z)] for diagonal posterior N(mu, diag(exp(log_var))).
        mu, log_var: (T, r) -> (T, N)."""
        r = self.r
        mu = tf.cast(mu, tf.float32)
        std = tf.exp(0.5 * tf.cast(log_var, tf.float32))            # (T, r)
        if self.use_cubature:
            scale = tf.sqrt(tf.cast(r, tf.float32))
            eye = tf.eye(r, dtype=tf.float32)
            shifts = tf.concat([eye, -eye], axis=0) * scale         # (2r, r)
            z = mu[:, None, :] + shifts[None, :, :] * std[:, None, :]  # (T, 2r, r)
            n = 2 * r
        else:
            n = self.mc_samples
            eps = self._generator_vae.normal((tf.shape(mu)[0], n, r))
            z = mu[:, None, :] + eps * std[:, None, :]              # (T, n, r)
        T = tf.shape(z)[0]
        dec = self._decoder(tf.reshape(z, (-1, r)), training=False)  # (T*n, N)
        dec = tf.reshape(dec, (T, n, -1))
        return tf.reduce_mean(dec, axis=1)                          # (T, N)

    def _expected_decode_moment(self, m, S):
        """Single E[decoder(z)] for z ~ N(m, S) (full cov). m:(r,), S:(r,r) -> (N,)."""
        r = self.r
        # L = np.linalg.cholesky(S + 1e-8 * np.eye(r))                # S = L L^T
        L = np.linalg.cholesky(S)
        if self.use_cubature:
            step = np.sqrt(r) * L.T                                 # row i = sqrt(r) L[:,i]^T
            z = np.vstack([m[None, :] + step, m[None, :] - step])   # (2r, r)
        else:
            eps = self._generator_vae.normal((self.mc_samples, r)).numpy()
            z = m[None, :] + eps @ L.T                              # (M, r): cov = L L^T = S
        dec = self._decoder(tf.constant(z.astype(np.float32)), training=False).numpy()
        return dec.mean(axis=0)                                     # (N,)

    def _make_train_step(self, trainable_vars, optimizer):
        p = self.var_order

        def step_body(x_t: tf.Tensor, x_tgt: tf.Tensor) -> tf.Tensor:
            with tf.GradientTape() as tape:
                mu, log_var = self._encode(x_t, training=True)
                z = self._sample(mu, log_var)
                self._var_layer.update_weights_closed_form(tf.stop_gradient(mu))
                z_pred = self._var_layer(mu)
                x_recon = self._decoder(z, training=True)
                # ===== -ELBO under the VAR prior z_t | lags ~ N(mu_hat_t, I) =====
                # (the bounded objective below is the standard -ELBO with the KL
                # taken against the VAR prior, innovation covariance normalised
                # to I.)
                # Term 1 — reconstruction  ||y_t - y_hat_t||^2  (masked, scalar):
                recon_loss = mse_missing(x_tgt, x_recon)
                # Term 2 — KL mean part: VAR consistency  ||mu_t - mu_hat_t||^2
                var_consistency = tf.reduce_mean(
                    tf.reduce_mean(tf.square(mu[p:] - z_pred[p:]), axis=1)
                )
                # Term 2b — KL mean part, propagated lag variance.
                #   E_q||z_t - sum_k A_k z_{t-k}||^2
                #     = ||mu_t - mu_hat_t||^2 + tr(V_t) + sum_k tr(A_k V_{t-k} A_k')
                #   The last piece is dropped if the conditioning lags are
                #   evaluated at their posterior means, as z_pred above does.
                #   For diagonal V_{t-k} = diag(v_{t-k}),
                #     tr(A_k V_{t-k} A_k') = sum_j v_{t-k,j} ||A_k[:,j]||^2,
                #   and row (k-1)r+j of `coefficients` IS A_k[:,j] (because
                #   z_pred = X_lags @ coefficients), so those column norms are
                #   the row sums of squares. build_lagged_matrix's default
                #   start=1 yields [v_{t-1},...,v_{t-p}], aligned with the rows.
                #   Shares var_loss_weight with Term 2: both come from the same
                #   1/(2 sigma^2) quadratic form. The 1/r matches the per-factor
                #   mean used in var_consistency.
                #   NB this is the term Leushuis (2025) Eq. (3) carries with the
                #   opposite sign; as a penalty it is bounded below by 0.
                v_lagged = self._var_layer.build_lagged_matrix(tf.exp(log_var))
                col_sq = tf.reduce_sum(
                    tf.square(tf.stop_gradient(self._var_layer.coefficients)),
                    axis=1,
                )
                tr_prop = tf.reduce_mean(
                    tf.reduce_mean(v_lagged * col_sq[tf.newaxis, :], axis=1)[p:]
                )
                # Term 3 — KL variance part: 1/2 sum_j (exp(log_var) - log_var - 1)
                #   +exp(log_var): variance penalty (linear, bounds the objective)
                #   -log_var     : entropy (rewards variance only logarithmically)
                #   Minimised at log_var = 0, i.e. unit posterior variance.
                kl_var = 0.5 * tf.reduce_mean(
                    tf.reduce_sum(tf.exp(log_var[p:]) - log_var[p:] - 1.0, axis=1)
                )
                total_loss = (
                        recon_loss
                        + self.var_loss_weight * (var_consistency + tr_prop)
                        + self.beta * kl_var
                )
            grads = tape.gradient(total_loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            return total_loss

        @tf.function
        def train_epochs(x_t: tf.Tensor, x_tgt: tf.Tensor, n: tf.Tensor) -> tf.Tensor:
            loss = tf.constant(float("inf"), tf.float32)
            for _ in tf.range(n):
                loss = step_body(x_t, x_tgt)
            return loss

        return train_epochs

    def _train_epochs(self, x_imp, x_target, train_step):
        x_t = tf.constant(x_imp, dtype=tf.float32)
        x_tgt = tf.constant(x_target, dtype=tf.float32)
        return float(train_step(x_t, x_tgt, tf.constant(self.epochs)))

    def _get_trainable_vars(self):
        return (
            self._encoder.trainable_variables
            + self._decoder.trainable_variables
        )

    def _get_all_weights(self):
        return (
            _get_model_weights(self._encoder)
            + _get_model_weights(self._decoder)
            + [w.numpy() for w in self._var_layer.weights]
        )

    def _set_all_weights(self, weight_list):
        enc_n = len(self._encoder.weights)
        dec_n = len(self._decoder.weights)
        _set_model_weights(self._encoder, weight_list[:enc_n])
        _set_model_weights(self._decoder, weight_list[enc_n:enc_n + dec_n])
        for w, v in zip(self._var_layer.weights, weight_list[enc_n + dec_n:]):
            w.assign(v)

    def _fit_once(self, x, x_imp, nan_mask):
        trainable_vars = self._get_trainable_vars()
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        optimizer.build(trainable_vars)
        train_step = self._make_train_step(trainable_vars, optimizer)
        x_imp_local = x_imp.copy()
        if self.nan_max_iterations > 0 and np.any(nan_mask):
            loss, _iter, _diff = float("inf"), 0, self.nan_tol + 1.0
            while _iter < self.nan_max_iterations and _diff > self.nan_tol:
                loss = self._train_epochs(x_imp_local, x, train_step)
                mu, log_var = self._encode(tf.constant(x_imp_local, tf.float32), training=False)
                x_recon = self._expected_decode(mu, log_var).numpy()
                _diff = np.mean(np.abs(x_imp_local[nan_mask] - x_recon[nan_mask]))
                x_imp_local[nan_mask] = x_recon[nan_mask]
                _iter += 1
        else:
            loss = self._train_epochs(x_imp_local, x, train_step)
        return loss

    def fit(self, data: pd.DataFrame) -> None:
        self._variables = list(data.columns)
        data = data[self._variables]
        self.mean_data = data.mean().values
        self.sigma_data = data.std().values
        if np.any(self.sigma_data == 0):
            raise ValueError("Some variables have zero variance.")
        x = self._std(data)
        T, N = x.shape
        nan_mask = np.isnan(x)
        x_imp = _column_mean_impute(x)

        best_loss, best_weights, best_seed = float("inf"), None, None
        n_seeds = len(self.seeds)
        for c, v in enumerate(self.seeds):
            self._build(N, v)
            loss = self._fit_once(x, x_imp, nan_mask)
            logger.info("VARVAE init %d/%d: loss=%.6f", c + 1, n_seeds, loss)
            if loss < best_loss:
                best_loss, best_weights, best_seed = loss, self._get_all_weights(), v

        if best_weights is None:
            raise RuntimeError("All seed initialisations produced non-finite loss.")

        if n_seeds > 1:
            self._build(N, best_seed)
            self._set_all_weights(best_weights)
        self._fitted = True

    def _nans_iter(self, x, nan_idxs):
        _dif, _iter = self.nan_tol + 1.0, 0
        while _iter < self.nan_max_iterations and _dif > self.nan_tol:
            mu, log_var = self._encode(tf.constant(x, tf.float32), training=False)
            x_new = self._expected_decode(mu, log_var).numpy()
            _dif = np.mean(np.abs(x_new[nan_idxs] - x[nan_idxs]))
            x[nan_idxs] = x_new[nan_idxs]
            _iter += 1
        return x

    def _encode_series(self, data):
        data = data[self._variables]
        nan_idxs = np.isnan(data.values)
        x_imp = _column_mean_impute(self._std(data))
        if np.any(nan_idxs):
            x_imp = self._nans_iter(x_imp, nan_idxs)
            mu, log_var = self._encode(tf.constant(x_imp, tf.float32), training=False)
            mu, log_var = mu.numpy(), log_var.numpy()
            x_new = self._expected_decode(mu, log_var).numpy()
            x_imp[nan_idxs] = x_new[nan_idxs]
        else:
            mu, log_var = self._encode(tf.constant(x_imp, tf.float32), training=False)
            mu, log_var = mu.numpy(), log_var.numpy()
        return data, x_imp, mu, log_var

    def get_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        data, _, mu, _ = self._encode_series(data)
        cols = [f"f{i + 1}" for i in range(self.r)]
        return pd.DataFrame(mu, index=data.index, columns=cols)

    def fill_na(self, data: pd.DataFrame) -> pd.DataFrame:
        data, x_std, _, _ = self._encode_series(data)
        return pd.DataFrame(self._unstd(x_std), index=data.index, columns=self._variables)

    def _var_innovation_cov(self, mu: np.ndarray) -> np.ndarray:
        """Empirical innovation covariance Q_hat of the fitted VAR on mu."""
        p = self.var_order
        z_pred = self._var_layer(tf.constant(mu.astype(np.float32))).numpy()
        resid = mu[p:] - z_pred[p:]
        return (resid.T @ resid) / max(resid.shape[0], 1)           # (r, r)

    def _forecast_latent_moments(self, mu, log_var, steps, Q):
        # One-step predictive covariance: P_{h} = F P_{h-1} F' + Q, with
        # P_0 = filtering uncertainty about the last p latent states (posterior
        # variances) and Q the VAR innovation covariance. At h=1 the first term
        # is A diag(exp(log_var_T)) A' — the paper's V_hat — plus the Q it omits.
        p, r = self.var_order, self.r
        d = p * r
        coef = self._var_layer.coefficients.numpy()                 # (p*r, r), row-conv
        F = np.zeros((d, d))
        F[:r, :] = coef.T                                           # z_{t+1} = coef^T @ s_t
        if p > 1:
            F[r:d, :(p - 1) * r] = np.eye((p - 1) * r)              # identity shifts
        Qs = np.zeros((d, d))
        Qs[:r, :r] = Q
        sc = mu[-p:][::-1].reshape(-1).astype(np.float64)          # [z_T; z_{T-1}; ...]
        # P_0: block-diagonal posterior variances of [z_T; z_{T-1}; ...]
        # (mean-field posterior => no cross-lag covariances)
        P = np.diag(np.exp(log_var[-p:][::-1].reshape(-1)).astype(np.float64))
        out = []
        for _ in range(steps):
            sc = F @ sc
            P = F @ P @ F.T + Qs
            out.append((sc[:r].copy(), P[:r, :r].copy()))
        return out

    def predict(self, data: pd.DataFrame, steps_ahead: int) -> pd.DataFrame:
        _, _, mu, log_var = self._encode_series(data)
        x_h0 = self._expected_decode(mu[-1:], log_var[-1:]).numpy()
        rows = [x_h0]
        if steps_ahead > 0:
            Q = self._var_innovation_cov(mu)
            for m_h, S_h in self._forecast_latent_moments(mu, log_var, steps_ahead, Q):
                rows.append(self._expected_decode_moment(m_h, S_h)[None, :])
        x_pred_std = np.vstack(rows)                                # (steps+1, N)
        return pd.DataFrame(
            self._unstd(x_pred_std), index=range(steps_ahead + 1), columns=self._variables
        )