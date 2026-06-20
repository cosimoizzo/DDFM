import logging
from typing import Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.base import FactorModel
from models.state_space.state_space_wrapper import StateSpace, FilterType
from models.vector_autoregressive import VARLayerClosedForm, VARAutoencoder
from tools.loss_tools import mse_missing, np_mse_missing
from tools.getters_converters_tools import (
    get_ssm_from_decoder,
    get_transition_params,
    get_idio,
    get_data_with_lags,
    _get_idio_matrix,
)
from tools.monthly_quarterly_layer import MixedFreqMQLayer

# tf.config.run_functions_eagerly(True)

# Using Marginalized UKF instead of UKF (speed up and efficiency gain)
_USE_M_UKF = True


class DDFM(FactorModel):
    """
    Deep Dynamic Factor Models
    """

    def __init__(
        self,
        lags_input: int = 0,
        structure_encoder: tuple = (16, 4),
        structure_decoder: tuple = None,
        use_bias: bool = True,
        factor_order: int = 2,
        var_loss_weight: float = 0.0,
        seed: int = 3,
        batch_norm: bool = True,
        link: str = "relu",
        learning_rate: float = 0.005,
        optimizer: str = "Adam",
        decay_learning_rate: bool = True,
        clipnorm: Optional[float] = None,
        epochs: int = 150,
        batch_size: int = 250,
        max_iter: int = 200,
        tolerance: float = 0.0005,
        disp: int = 10,
        logger=logging.getLogger("DDFM"),
        dtype: Optional[tf.DType] = tf.float32,
    ):
        """

        Args:
            lags_input: number of lags of the inputs on the encoder (default is 0, i.e. same inputs and outputs)
            structure_encoder: number of layers and neurons for the encoder
            structure_decoder: number of layers and neurons for the decoder (default is None, i.e. asymmetric
                autoencoder with one single linear layer decoder)
            use_bias: whether to use bias term in the last decoder layer
            factor_order: number of lags in the transition equation for the dynamics of the common factors
            var_loss_weight: if > 0, then estimate jointly the var dynamics of the latent factors and add VAR forecast
                error term in the reconstruction loss of the autoencoder
                error term in the reconstruction loss of the autoencoder
            seed: seed to control randomness for replicability
            batch_norm: whether to add batch norm layers into the encoder
            link: the type of link/activation function
            learning_rate: the learning rate for the optimizer
            optimizer: the selected optimizer
            decay_learning_rate: whether to use a decaying learning rate
            clipnorm: whether to clip gradient norm
            epochs: number of epochs between iterations
            batch_size: the size of the batch
            max_iter: maximum number of iterations
            tolerance: the tolerance to stop iterations
            disp: display intermediate results every "disp" iterations
            logger:
            dtype:

        """
        super().__init__(r=structure_encoder[-1])
        # common factors
        self.factor_order = factor_order
        if factor_order not in [1, 2]:
            raise ValueError("factor_order must be 1 or 2")
        self.var_loss_weight = var_loss_weight
        self.lags_input = lags_input
        # autoencoder structure
        self.structure_encoder = structure_encoder
        self.structure_decoder = structure_decoder
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.link = link
        if self.structure_decoder is None:
            self._filter_type = FilterType.KalmanFilter
        else:
            self._filter_type = FilterType.Marginalized_UKF if _USE_M_UKF else FilterType.UnscentedKalmanFilter
        # seed setting
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # learning process
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.disp = disp
        # optimizer
        if decay_learning_rate:
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps=epochs, decay_rate=0.96, staircase=True
            )
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.logger = logger
        self.dtype = dtype
        # initialize relevant attributes
        self.quarterly_start = None
        self._optimizer = None
        self.data = None
        self.variable_order = None
        self.loss_now = None
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.idio_residuals = None
        self.last_neurons = None
        self.factors_ae = None
        self.factors_filtered = None
        self.factors_smoothed = None
        self.state_space = None
        self._latents = {}

    def fit(
        self,
        data: pd.DataFrame,
        build_state_space: bool = True,
        vars_mq_restrictions: List[Any] = None,
    ) -> None:
        """
        Model fitting
        Args:
            data: data for training
            build_state_space: whether to build the final state space representation for model inference
            vars_mq_restrictions: list of quarterly variables where monthly to quarterly aggregation restrictions are
                applied (the quarterly variable is assumed to be a flow variable)

        Returns:
            None, it updates internal attributes and makes the model ready for inference
        """
        self._training_data_set_up(data, vars_mq_restrictions)
        quarterly_start = (
            data.shape[1] - len(vars_mq_restrictions) if vars_mq_restrictions else None
        )
        self.quarterly_start = quarterly_start
        self._build_model()
        self._pre_train()
        self._train()
        if build_state_space:
            self.state_space = self.build_state_space()
            # get filtered factors
            self._latents["filtered"], self._latents["sigma_filtered"] = (
                self.state_space.filter(data.values)
            )
            self._latents["smoothed"], self._latents["sigma_smoothed"] = (
                self.state_space.smooth(data.values)
            )
            self.factors_filtered = self._latents["filtered"][
                :, : self.structure_encoder[-1]
            ]
            self.factors_smoothed = self._latents["smoothed"][
                :, : self.structure_encoder[-1]
            ]

    def predict_with_covariance(
        self, data: pd.DataFrame, steps_ahead: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prediction step using state-space representation.

        Returns:
            (mean DataFrame of shape (steps_ahead+1, N),
             covariance DataFrame with MultiIndex)
        """
        if self.state_space is None:
            raise ValueError("State space must be built before making inference")
        mean, cov = self.state_space.predict(
            data[self.variable_order].sort_index().values, steps_ahead=steps_ahead
        )
        return self._numpy_to_df_mean_and_cov(mean, cov, steps_ahead)

    def predict(self, data: pd.DataFrame, steps_ahead: int = 1) -> pd.DataFrame:
        """
        Forecast observables at horizons h=0..steps_ahead.
        Row 0 is the reconstruction at the last observed point (h=0).
        Rows 1..steps_ahead are h-step-ahead forecasts.
        Output is in the original (un-standardised) scale.
        """
        mean_df, _ = self.predict_with_covariance(data, steps_ahead)
        return mean_df

    def get_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Return Kalman-smoothed latent factors using the fitted state space.

        Returns pd.DataFrame of shape (T, r) with columns ["f1", ..., "fr"].
        The factors live in their own latent space and are NOT un-standardised.
        """
        if self.state_space is None:
            raise ValueError("State space must be built before calling get_factors.")
        x = data[self.variable_order].sort_index().values
        smoothed, _ = self.state_space.smooth(x)
        f_hat = smoothed[:, : self.r]
        cols = [f"f{i + 1}" for i in range(self.r)]
        return pd.DataFrame(f_hat, index=data.sort_index().index, columns=cols)

    def fill_na(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using kalman smoother.
        """
        if self.state_space is None:
            raise ValueError("State space must be built before calling get_factors.")
        x = data[self.variable_order].values.astype(float)
        x_filled, x_cov = self.state_space.fill_na(x)
        df = pd.DataFrame(x_filled, index=data.index, columns=self.variable_order)
        return df[data.columns]

    def predict_from_states(
        self,
        x_hat_start: np.ndarray,
        sigma_x_hat_start: np.ndarray,
        steps_ahead: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prediction step using state-space representation with state starting values
        Args:
            x_hat_start: starting values for the state mean
            sigma_x_hat_start: starting values for the state variance covariance matrix
            steps_ahead: number of steps ahead

        Returns:
            mean predictions and covariances
        """
        if self.state_space is None:
            raise ValueError("State space must be built before making inference")
        mean, cov = self.state_space.predict_from_state(
            x_hat_start, sigma_x_hat_start, steps_ahead=steps_ahead
        )
        return self._numpy_to_df_mean_and_cov(mean, cov, steps_ahead)

    def build_state_space(self) -> StateSpace:
        """
        Build state space object from the autoencoder (decoder).
            measurement: z_t = H(x_t) + v_t; v_t ∼ N(0, R)
            transition: x_t = F(x_t-1) + w_t; w_t ∼ N(0, Q)
        Returns:
            The state space object
        """
        # extract common factors
        f_t = np.mean(self.factors_ae, axis=0)
        # get params from decoder (measurement equation)
        bs, H = get_ssm_from_decoder(
            self.decoder,
            self.use_bias,
            self.factor_order,
            structure_decoder=self.structure_decoder,
        )
        # get transition equation params
        lags_needed = (
            len(self.decoder.get_layer(index=-1).aggr_restr)
            if self.quarterly_start is not None
            else None
        )
        F, Q, x_t = get_transition_params(
            f_t,
            self.idio_residuals,
            factor_order=self.factor_order,
            bool_no_miss=self._bool_no_miss,
            extended_factor_lags=(
                max(0, lags_needed - self.factor_order)
                if lags_needed is not None
                else 0
            ),
            quarterly_start=self.quarterly_start,
            transition_as_keras_model=self._filter_type
            == FilterType.UnscentedKalmanFilter,
            dtype=self.dtype,
        )
        self._latents["ae_states"] = x_t
        R = (
            np.eye(self.idio_residuals.shape[1]) * 1e-5
            if self.dtype == tf.float32
            else np.eye(self.idio_residuals.shape[1]) * 1e-10
        )
        measurement = {
            "observation_map": H,
            "observation_covariance": R,
            "observation_offsets": bs,
        }
        if self._filter_type == FilterType.Marginalized_UKF:
            measurement["linear_observation_map"] = _get_idio_matrix(self.data.shape[1], self.decoder.get_layer(index=-1))
        transition = {"transition_map": F, "transition_covariance": Q}
        return StateSpace(
            transition,
            measurement,
            mean_y=self.mean_data,
            sigma_y=self.sigma_data,
            filter_type=self._filter_type,
            x0=np.nanmean(x_t, axis=1),
            P0=np.diag(np.nanvar(x_t, axis=1)),
            dtype=self.dtype,
        )

    def _init_optimizer(self):
        if self.optimizer == "SGD":
            self._optimizer = keras.optimizers.SGD(
                learning_rate=self.learning_rate, clipnorm=self.clipnorm
            )
        elif self.optimizer == "Adam":
            self._optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate, clipnorm=self.clipnorm
            )
        else:
            raise KeyError("Optimizer must be SGD or Adam")

    def _training_data_set_up(
        self, data: pd.DataFrame, vars_mq_restrictions: List[str] = None
    ) -> None:
        if vars_mq_restrictions is not None:
            vars_order = [
                v for v in data.columns if v not in vars_mq_restrictions
            ] + vars_mq_restrictions
            data = data[vars_order]
        data.sort_index(inplace=True)
        self.variable_order = data.columns
        self.mean_data = data.mean().values
        self.sigma_data = data.std().values
        if np.any(self.sigma_data == 0):
            raise ValueError("Some variables have zero variance.")
        self.data = (data - self.mean_data) / self.sigma_data
        # keep track of missing
        self._bool_miss = self.data.isnull()[self.lags_input :].values
        self._bool_no_miss = ~self._bool_miss
        # create two copies of the original data which will be modified during training
        # (missing data imputation and idiosyncratic one side filtering)
        self._data_imputed, self._data_mod = self.data.copy(), self.data.copy()
        self._target = self.data[self.lags_input :].values
        self._target_tf = tf.convert_to_tensor(self._target, dtype=self.dtype)

    def _build_model(self) -> None:
        # encoder
        inputs = keras.Input(
            shape=(int((self.lags_input + 1) * self.data.shape[1]),), dtype=self.dtype
        )
        len_encoder = len(self.structure_encoder)
        if len_encoder > 1:
            encoded = layers.Dense(
                self.structure_encoder[0],
                activation=self.link,
                bias_initializer="zeros",
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=self.seed),
                dtype=self.dtype,
            )(inputs)
            for c, j in enumerate(self.structure_encoder[1:]):
                if self.batch_norm:
                    encoded = layers.BatchNormalization()(encoded)
                encoded = layers.Dense(
                    j,
                    activation=(
                        self.link
                        if self.structure_decoder is None or c != len_encoder - 2
                        else None
                    ),
                    kernel_initializer=tf.keras.initializers.GlorotNormal(
                        seed=self.seed + c + 1
                    ),
                    bias_initializer="zeros",
                    dtype=self.dtype,
                )(encoded)
        else:
            encoded = layers.Dense(
                self.structure_encoder[0],
                bias_initializer="zeros",
                kernel_initializer=tf.keras.initializers.GlorotNormal(
                    seed=self.seed + 1
                ),
                dtype=self.dtype,
            )(inputs)

        self.encoder = keras.Model(inputs, encoded)
        # decoder
        latent_inputs = keras.Input(
            shape=(self.structure_encoder[-1],), dtype=self.dtype
        )
        if self.structure_decoder:
            decoded = layers.Dense(
                self.structure_decoder[0],
                activation=self.link,
                kernel_initializer=tf.keras.initializers.GlorotNormal(
                    seed=self.seed + len_encoder + 1
                ),
                bias_initializer="zeros",
                dtype=self.dtype,
            )(latent_inputs)
            for c, j in enumerate(self.structure_decoder[1:]):
                decoded = layers.Dense(
                    j,
                    activation=self.link,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(
                        seed=self.seed + len_encoder + 2 + c
                    ),
                    bias_initializer="zeros",
                    dtype=self.dtype,
                )(decoded)
            output = layers.Dense(
                self.data.shape[1],
                bias_initializer="zeros",
                kernel_initializer=tf.keras.initializers.GlorotNormal(
                    seed=self.seed + len_encoder + 2 + len(self.structure_decoder)
                ),
                use_bias=self.use_bias,
                dtype=self.dtype,
            )(decoded)
        else:
            output = layers.Dense(
                self.data.shape[1],
                bias_initializer="zeros",
                kernel_initializer=tf.keras.initializers.GlorotNormal(
                    seed=self.seed + len_encoder + 1
                ),
                use_bias=self.use_bias,
                dtype=self.dtype,
            )(latent_inputs)
        # If quarterly variables present, then add MM restrictions
        if self.quarterly_start is not None:
            output = MixedFreqMQLayer(
                self.data.shape[1], self.quarterly_start, dtype=self.dtype
            )(output)
        self.decoder = keras.Model(latent_inputs, output)
        outputs_ = self.decoder(self.encoder(inputs))
        # autoencoder
        self.autoencoder = keras.Model(inputs, outputs_)
        if self.var_loss_weight > 0:
            self.var_layer = VARLayerClosedForm(
                n_vars=self.structure_encoder[-1],
                var_order=self.factor_order,
                dtype=self.dtype,
            )
            self.var_autoencoder = VARAutoencoder(
                self.encoder,
                self.var_layer,
                self.decoder,
                ae_loss=mse_missing,
                var_loss_weight=self.var_loss_weight,
            )

    def _build_inputs(self, interpolate: bool = True) -> None:
        self._data_tmp = get_data_with_lags(
            interpolate=interpolate, data_raw=self._data_mod, lags_input=self.lags_input
        )

    def _pre_train(self, min_obs: int = 50, mult_epoch_pre: int = 1) -> None:
        self._init_optimizer()
        self.autoencoder.compile(optimizer=self._optimizer, loss="mse")
        # build inputs without interpolation
        self._build_inputs(interpolate=False)
        # check number of observations, and if not enough then interpolate
        if self._data_tmp.dropna().shape[0] >= min_obs:
            inpt_pre_train = self._data_tmp.dropna().values
        else:
            self._build_inputs()
            inpt_pre_train = self._data_tmp.dropna().values
        oupt_pre_train = self._data_tmp.dropna()[self.variable_order].values
        self.autoencoder.fit(
            inpt_pre_train,
            oupt_pre_train,
            epochs=self.epochs * mult_epoch_pre,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=False,
        )

    def _train(self) -> None:
        """
        Algorithm 1 of the paper.
        """
        # re-compile the autoencoder to re-init the optimizer
        self._init_optimizer()
        self.autoencoder.compile(optimizer=self._optimizer, loss=mse_missing)
        # construct initial input data
        self._build_inputs()
        # if jointly estimate var
        if self.var_loss_weight > 0:
            factors = self.encoder.predict(self._data_tmp.values)
            self.var_layer.update_weights_closed_form(factors)
        prediction_iter = self.autoencoder.predict(self._data_tmp.values)
        # update missing
        self._data_imputed.values[self.lags_input :][self._bool_miss] = prediction_iter[
            self._bool_miss
        ]
        # idiosyncratic term
        self.idio_residuals = (
            self._data_imputed.values[self.lags_input :] - prediction_iter
        )
        # start MCMC iterations
        self.i_iter = 0
        not_converged = True
        T, D = self._data_tmp.shape[0], self.data.shape[1]
        fit_method = (
            self._fit_method_var_ae(T, D)
            if self.var_loss_weight > 0
            else self._fit_method_ae(T, D)
        )
        while not_converged and self.i_iter < self.max_iter:
            # get idio distribution
            phi, std_eps, cond_mean = get_idio(
                self.idio_residuals,
                self._bool_no_miss,
                quarterly_start=self.quarterly_start,
            )
            # subtract conditional AR-idio mean from x
            self._data_mod[self.lags_input + 1 :] = (
                self._data_imputed[self.lags_input + 1 :] - cond_mean[:-1]
            )
            # for first observations set to 0 the idio
            self._data_mod[: self.lags_input + 1] = self._data_imputed[
                : self.lags_input + 1
            ]
            # gen data_tmp from _data_mod
            self._build_inputs(interpolate=False)
            # gen MC samples for idio (dims = Sim x T * D)
            idio_residuals_sims = self.rng.multivariate_normal(
                np.zeros_like(std_eps), np.diag(std_eps**2), self.epochs * T
            )
            # memory intensive, could be converted to a loop if memory is an issue - or reduce epochs
            x_sim_noisy = np.tile(self._data_tmp.values, (self.epochs, 1))
            # Column order: [y_t, y_{t-1}, ..., y_{t-lags}]
            x_sim_noisy[:, :D] -= idio_residuals_sims
            x_sim_noisy = tf.convert_to_tensor(x_sim_noisy, dtype=self.dtype)
            fit_method(x_sim_noisy)
            # update factors: average over all predictions from the MC samples
            factors_ae_sims = self.encoder(x_sim_noisy)
            prediction_iter = tf.reduce_mean(
                tf.reshape(self.decoder(factors_ae_sims), (self.epochs, T, D)), axis=0
            ).numpy()
            if self.var_loss_weight > 0:
                z_latent = tf.reduce_mean(
                    tf.reshape(
                        factors_ae_sims, (self.epochs, T, self.var_layer.n_vars)
                    ),
                    axis=0,
                )
                self.var_layer.update_weights_closed_form(z_latent)
                z_pred = self.var_layer(z_latent)
                var_loss = (
                    self.var_autoencoder.var_loss_weight
                    * self.var_autoencoder.var_loss(
                        z_latent[self.factor_order :], z_pred[self.factor_order :]
                    )
                )
            else:
                var_loss = 0
            self.loss_now = (
                np_mse_missing(self._target, prediction_iter, self._bool_no_miss)
                + var_loss
            )
            # check convergence
            if self.i_iter > 1:
                delta = (
                    2
                    * np.abs(self.loss_now - loss_prev)
                    / (np.abs(self.loss_now) + np.abs(loss_prev) + 1e-12)
                )
                if self.i_iter % self.disp == 0:
                    self.logger.info(
                        f"iteration: {self.i_iter} - new loss: {self.loss_now} - delta: {delta}"
                    )
                if delta < self.tolerance:
                    not_converged = False
                    self.logger.info(
                        f"Convergence achieved in {self.i_iter} iterations - new loss: {self.loss_now} - delta: {delta} < {self.tolerance}"
                    )
            # update missings
            self._data_imputed.values[self.lags_input :][self._bool_miss] = (
                prediction_iter[self._bool_miss]
            )
            # update idio
            self.idio_residuals = (
                self._data_imputed.values[self.lags_input :] - prediction_iter
            )
            loss_prev = self.loss_now
            self.i_iter += 1

        self.factors_ae = factors_ae_sims.numpy().reshape(self.epochs, T, -1)
        # get last neurons (making difference between nonlinear and linear decoder)
        if self.structure_decoder is None or len(self.decoder.layers) == 1:
            self.last_neurons = self.factors_ae
        else:
            decoder_for_last_neuron = keras.Model(
                self.decoder.input,
                self.decoder.get_layer(self.decoder.layers[-2].name).output,
            )
            self.last_neurons = (
                decoder_for_last_neuron(self.encoder(x_sim_noisy))
                .numpy()
                .reshape(self.epochs, T, -1)
            )

        if not_converged:
            self.logger.info(
                "Convergence not achieved within the maximum number of iteration!"
            )

    def _numpy_to_df_mean_and_cov(
        self, mean: np.ndarray, cov: np.ndarray, steps_ahead: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_mean = pd.DataFrame(
            mean, index=range(steps_ahead + 1), columns=self.variable_order
        )
        index = pd.MultiIndex.from_product(
            [range(steps_ahead + 1), self.variable_order], names=["Horizon", "Variable"]
        )
        df_cov = pd.DataFrame(
            cov.reshape((steps_ahead + 1) * mean.shape[1], mean.shape[1]),
            index=index,
            columns=self.variable_order,
        )
        return df_mean, df_cov

    def _fit_method_ae(
        self, T: int, D: int
    ) -> tf.types.experimental.PolymorphicFunction:
        epochs = tf.constant(self.epochs, dtype=tf.int32)
        batch_size = tf.constant(self.batch_size, dtype=tf.int32)

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, D * (self.lags_input + 1)], dtype=self.dtype)
            ]
        )
        def train_all_epochs(x_sim_noisy: tf.Tensor):
            vars_ = self.autoencoder.trainable_variables
            for e in tf.range(epochs):
                x_sim = tf.slice(x_sim_noisy, [e * T, 0], [T, -1])
                for i in tf.range(0, T, batch_size):
                    size_i = batch_size if batch_size <= T - i else T - i
                    with tf.GradientTape() as tape:
                        prediction = self.autoencoder(
                            tf.slice(x_sim, [i, 0], [size_i, -1]), training=True
                        )
                        loss_value = mse_missing(
                            tf.slice(self._target_tf, [i, 0], [size_i, -1]), prediction
                        )
                    grads = tape.gradient(loss_value, vars_)
                    self._optimizer.apply_gradients(zip(grads, vars_))

        return train_all_epochs

    def _fit_method_var_ae(
        self, T: int, D: int
    ) -> tf.types.experimental.PolymorphicFunction:
        epochs = tf.constant(self.epochs, dtype=tf.int32)
        batch_size = tf.constant(self.batch_size, dtype=tf.int32)

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, D * (self.lags_input + 1)], dtype=self.dtype)
            ]
        )
        def train_all_epochs(x_sim_noisy: tf.Tensor):
            vars_ = (
                self.autoencoder.trainable_variables
            )  # var dynamics are updated in closed form
            for e in tf.range(epochs):
                x_sim = tf.slice(x_sim_noisy, [e * T, 0], [T, -1])
                for i in tf.range(0, T, batch_size):
                    size_i = batch_size if batch_size <= T - i else T - i
                    with tf.GradientTape() as tape:
                        total_loss, recon_loss, var_loss = (
                            self.var_autoencoder.compute_loss(
                                tf.slice(x_sim, [i, 0], [size_i, -1]),
                                tf.slice(self._target_tf, [i, 0], [size_i, -1]),
                                with_var_training=False,
                            )
                        )
                    grads = tape.gradient(total_loss, vars_)
                    self._optimizer.apply_gradients(zip(grads, vars_))

        return train_all_epochs
