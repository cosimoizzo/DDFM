import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.state_space import StateSpace
from tools.loss_tools import mse_missing, convergence_checker
from tools.getters_converters_tools import convert_decoder_to_numpy, get_transition_params, get_idio, get_data_with_lags

# tf.config.run_functions_eagerly(True)


class DDFM:
    """
    Deep Dynamic Factor Models
    """

    def __init__(self, lags_input: int = 0, structure_encoder: tuple = (16, 4),
                 structure_decoder: tuple = None, use_bias: bool = True, factor_oder: int = 2, seed: int = 3,
                 batch_norm: bool = True, link: str = 'relu', learning_rate: float = 0.005,
                 optimizer: str = 'Adam', decay_learning_rate: bool = True,
                 epochs: int = 100, batch_size: int = 250, max_iter: int = 200, tolerance: float = 0.0005,
                 disp: int = 10,
                 logger = logging.getLogger('DDFM')):
        """

        Args:
            lags_input: number of lags of the inputs on the encoder (default is 0, i.e. same inputs and outputs)
            structure_encoder: number of layers and neurons for the encoder
            structure_decoder: number of layers and neurons for the decoder (default is None, i.e. asymmetric
                autoencoder with one single decoder linear layer)
            use_bias: whether to use bias term in the last decoder layer
            factor_oder: number of lags in the transition equation for the dynamics of the common factors
            seed: seed to control randomness for replicability
            batch_norm: whether to add batch norm layers into the encoder
            link: the type of link/activation function
            learning_rate: the learning rate for the optimizer
            optimizer: the selected optimizer
            decay_learning_rate: whether to use a decaying learning rate
            epochs: number of epochs between MCMC iterations
            batch_size: the size of the batch
            max_iter: maximum number of iterations for the MCMC
            tolerance: the tolerance to stop iterations
            disp: display intermediate results every "disp" iterations of MCMC

        """
        # common factors
        self.factor_oder = factor_oder
        if factor_oder not in [1, 2]:
            raise ValueError('factor_oder must be 1 or 2')
        self.lags_input = lags_input
        # autoencoder structure
        self.structure_encoder = structure_encoder
        self.structure_decoder = structure_decoder
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.link = link
        if self.structure_decoder is None:
            self.filter_type = "KalmanFilter"
        else:
            self.filter_type = "ToBeDefined"
        # seed setting
        self.rng = np.random.RandomState(seed)
        self.initializer = tf.keras.initializers.GlorotNormal(seed=seed)
        # learning process
        self.batch_size = batch_size
        self.epoch = epochs
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.disp = disp
        # optimizer
        if decay_learning_rate:
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps=epochs,
                                                                           decay_rate=0.96, staircase=True)
        if optimizer == 'SGD':
            self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'Adam':
            self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            raise KeyError("Optimizer must be SGD or Adam")
        self.logger = logger
        # initialize relevant attributes
        self.data = None
        self.variable_order = None
        self.mean_data = None
        self.sigma_data = None
        self.loss_now = None
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.eps = None
        self.last_neurons = None
        self.factors_ae = None
        self.factors_filtered = None
        self.factors_smoothed = None
        self.state_space = None
        self._latents = {}

    def fit(self, data: pd.DataFrame, build_state_space: bool = False) -> None:
        """
        Model fitting
        Args:
            data: data for training
            build_state_space: whether to build the final state space representation for model inference

        Returns:
            None, it updates internal attributes and makes the model ready for inference
        """
        self._training_data_set_up(data)
        self._build_model()
        self._pre_train()
        self._train()
        if build_state_space:
            self.state_space = self.build_state_space()
            # get filtered factors
            self._latents["filtered"], self._latents["sigma_filtered"] = self.state_space.filter(self.data.values)
            self._latents["smoothed"], self._latents["sigma_smoothed"] = self.state_space.smooth(self.data.values)
            self.factors_filtered = self._latents["filtered"][:, :self.structure_encoder[-1]]
            self.factors_smoothed = self._latents["smoothed"][:, :self.structure_encoder[-1]]

    def predict(self, data: pd.DataFrame, steps_ahead: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prediction step using state-space representation
        Args:
            data: observable data
            steps_ahead: number of steps ahead

        Returns:
            mean predictions and covariances
        """
        if self.state_space is None:
            raise ValueError("State space must be built before making inference")
        mean, cov = self.state_space.predict(data[self.variable_order].sort_index().values, steps_ahead=steps_ahead)
        return self._numpy_to_df_mean_and_cov(mean, cov, steps_ahead)

    def predict_from_states(self, x_hat_start: np.ndarray, sigma_x_hat_start: np.ndarray, steps_ahead: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        mean, cov = self.state_space.predict_from_state(x_hat_start, sigma_x_hat_start, steps_ahead=steps_ahead)
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
        # idio components
        eps_t = self.eps
        # get params from decoder (measurement equation)
        bs, H = convert_decoder_to_numpy(self.decoder, self.use_bias, self.factor_oder,
                                         structure_decoder=self.structure_decoder)
        # get transition equation params
        F, Q, mu_0, sigma_0, x_t = get_transition_params(f_t, eps_t, factor_oder=self.factor_oder,
                                                         bool_no_miss=self._bool_no_miss)
        self._latents["ae_states"] = x_t
        R = np.eye(eps_t.shape[1]) * 1e-15
        measurement = {"observation_matrices": H, "observation_covariance": R,
                       "observation_offsets": bs}
        transition = {"transition_matrices": F, "transition_covariance": Q}
        return StateSpace(transition, measurement,
                          mean_y=self.mean_data,
                          sigma_y=self.sigma_data,
                          filter_type=self.filter_type)

    def _training_data_set_up(self, data: pd.DataFrame):
        data.sort_index(inplace=True)
        self.variable_order = data.columns
        self.mean_data = data.mean().values
        self.sigma_data = data.std().values
        self.data = (data - self.mean_data) / self.sigma_data
        # keep track of missing
        self._bool_miss = self.data.isnull()[self.lags_input:].values
        self._bool_no_miss = self._bool_miss == False
        # create two copies of the original data which will be modified during training
        self._data_mod_only_miss, self._data_mod = self.data.copy(), self.data.copy()
        self._target = self.data[self.lags_input:].values

    def _build_model(self) -> None:
        """
        Build the keras model
        """
        # encoder
        inputs_ = keras.Input(shape=(int((self.lags_input + 1) * self.data.shape[1]),))
        if len(self.structure_encoder) > 1:
            encoded = layers.Dense(self.structure_encoder[0], activation=self.link,
                                   bias_initializer='zeros', kernel_initializer=self.initializer)(inputs_)
            for j in self.structure_encoder[1:]:
                if self.batch_norm:
                    encoded = layers.BatchNormalization()(encoded)
                encoded = layers.Dense(j, activation=self.link,
                                       kernel_initializer=self.initializer,
                                       bias_initializer='zeros')(encoded)
        else:
            encoded = layers.Dense(self.structure_encoder[0], bias_initializer='zeros',
                                   kernel_initializer=self.initializer)(inputs_)

        self.encoder = keras.Model(inputs_, encoded)
        # decoder
        latent_inputs = keras.Input(shape=(self.structure_encoder[-1],))
        if self.structure_decoder:
            decoded = layers.Dense(self.structure_decoder[0], activation=self.link,
                                   kernel_initializer=self.initializer,
                                   bias_initializer='zeros')(latent_inputs)
            for j in self.structure_decoder[1:]:
                decoded = layers.Dense(j, activation=self.link, kernel_initializer=self.initializer,
                                       bias_initializer='zeros')(decoded)
            output_ = layers.Dense(self.data.shape[1], bias_initializer='zeros',
                                   kernel_initializer=self.initializer, use_bias=self.use_bias)(decoded)
        else:
            output_ = layers.Dense(self.data.shape[1], bias_initializer='zeros',
                                   kernel_initializer=self.initializer, use_bias=self.use_bias)(latent_inputs)
        self.decoder = keras.Model(latent_inputs, output_)
        outputs_ = self.decoder(self.encoder(inputs_))
        # autoencoder
        self.autoencoder = keras.Model(inputs_, outputs_)

    def _build_inputs(self, interpolate: bool = True) -> None:
        self._data_tmp = get_data_with_lags(interpolate=interpolate, data_raw=self._data_mod, lags_input=self.lags_input)

    def _pre_train(self, min_obs: int = 50, mult_epoch_pre: int = 1) -> None:
        """
        Carry out pre-training of the model.
        """
        # build inputs without interpolation
        self._build_inputs(interpolate=False)
        # check number of observations, and if not enough then interpolate
        if len(self._data_tmp.dropna()) >= min_obs:
            inpt_pre_train = self._data_tmp.dropna().values
            self.autoencoder.compile(optimizer=self.optimizer, loss='mse')
        else:
            self._build_inputs()
            inpt_pre_train = self._data_tmp.dropna().values
            self.autoencoder.compile(optimizer=self.optimizer, loss=mse_missing)
        # get target
        oupt_pre_train = self._data_tmp.dropna()[self.variable_order].values
        # fit (pre-train) autoencoder
        self.autoencoder.fit(inpt_pre_train, oupt_pre_train, epochs=self.epoch * mult_epoch_pre,
                             batch_size=self.batch_size,
                             verbose=0)

    def _train(self) -> None:
        """
        Algorithm 1 of the paper.
        """
        # re-compile the autoencoder to re-init the optimizer and possibly change the objective
        self.autoencoder.compile(optimizer=self.optimizer, loss=mse_missing)
        # construct initial input data
        self._build_inputs()
        # make prediction
        prediction_iter = self.autoencoder.predict(self._data_tmp.values)
        # update missings
        self._data_mod_only_miss.values[self.lags_input:][self._bool_miss] = prediction_iter[self._bool_miss]
        # idiosyncratic term
        self.eps = self._data_tmp[self.variable_order].values - prediction_iter
        self._i_iter = 0
        not_converged = True
        T, D = self._data_tmp.shape[0], self.data.shape[1]
        # start MCMC
        while not_converged and self._i_iter < self.max_iter:
            # get idio distribution
            phi, mu_eps, std_eps = get_idio(self.eps, self._bool_no_miss)
            # subtract conditional AR-idio mean from x
            self._data_mod[self.lags_input + 1:] = self._data_mod_only_miss[self.lags_input + 1:] - self.eps[:-1, :] @ phi
            # for first observations set to 0 the idio
            self._data_mod[:self.lags_input + 1] = self._data_mod_only_miss[:self.lags_input + 1]
            # gen data_tmp from data_mod
            self._build_inputs(interpolate=False)
            # gen MC samples for idio (dims = Sim x T * D)
            eps_draws = self.rng.multivariate_normal(mu_eps, np.diag(std_eps), self.epoch * T)
            x_sim_noisy = np.concatenate([self._data_tmp.copy()] * self.epoch, axis=0)
            x_sim_noisy[:, :D] -= eps_draws
            for e in range(self.epoch):
                for i in range(0, T, self.batch_size):
                    x_sim = x_sim_noisy[e*T:(e+1)*T]
                    self.autoencoder.train_on_batch(x_sim[i:i+self.batch_size], self._target[i:i+self.batch_size])
            # update factors: average over all predictions from the MC samples
            factors_ae_sims = self.encoder(x_sim_noisy).numpy()
            self.factors_ae = factors_ae_sims.reshape(self.epoch, T, -1)
            # check convergence
            decoded_sims = self.decoder(factors_ae_sims).numpy().reshape(self.epoch, T, D)
            prediction_iter = np.mean(decoded_sims, axis=0)
            if self._i_iter > 1:
                delta, self.loss_now = convergence_checker(prediction_prev_iter, prediction_iter, self._target)
                if self._i_iter % self.disp == 0:
                    self.logger.info(f'iteration: {self._i_iter} - new loss: {self.loss_now} - delta: {delta}')
                if delta < self.tolerance:
                    not_converged = False
                    self.logger.info(f'Convergence achieved in {self._i_iter} iterations - new loss: {self.loss_now} - delta: {delta} < {self.tolerance}')
            # store previous prediction to monitor convergence
            prediction_prev_iter = prediction_iter.copy()
            # update missings
            self._data_mod_only_miss.values[self.lags_input:][self._bool_miss] = prediction_iter[self._bool_miss]
            # update idio
            self.eps = self._data_mod_only_miss.values[self.lags_input:] - prediction_iter
            self._i_iter += 1

        # get last neurons (making difference between nonlinear and linear decoder)
        if self.structure_decoder is None:
            self.last_neurons = self.factors_ae
        else:
            decoder_for_last_neuron = keras.Model(self.decoder.input,
                                                  self.decoder.get_layer(self.decoder.layers[-2].name).output)
            self.last_neurons = decoder_for_last_neuron(self.encoder(x_sim_noisy)).numpy().reshape(self.epoch, T, -1)

        if not_converged:
            self.logger.info("Convergence not achieved within the maximum number of iteration!")

    def _numpy_to_df_mean_and_cov(self, mean, cov, steps_ahead):
        df_mean = pd.DataFrame(mean, index=range(steps_ahead + 1), columns=self.variable_order)
        index = pd.MultiIndex.from_product([range(steps_ahead + 1), self.variable_order],
                                           names=["Horizon", "Variable"])
        df_cov = pd.DataFrame(cov.reshape(steps_ahead * mean.shape[1]), index=index, columns=self.variable_order)
        return df_mean, df_cov


if __name__ == "__main__":
    # import tensorflow as tf
    import sklearn  # maybe we can remove this dependency
    import random
    from timeit import default_timer as timer
    from datetime import timedelta
    import logging
    logging.basicConfig(level=logging.INFO)

    random.seed(0)
    np.random.seed(0)
    print('tf version: ', tf.__version__)
    print('sklearn version: ', sklearn.__version__)
    obs_dim = 20
    scale_idio = 0.2
    scale_idio_error = 0.01
    latent_dim = 4
    rho = 0.5
    t_obs = 100
    non_lin_factors = True
    # simulate factors
    f = np.random.multivariate_normal(np.zeros(latent_dim), np.identity(latent_dim), t_obs)
    if non_lin_factors:
        f = np.hstack((f, f ** 2, f ** 3))
    # simulate idios
    v_t = scale_idio * np.random.multivariate_normal(np.zeros(obs_dim), np.identity(obs_dim), t_obs)
    eps = np.zeros_like(v_t)
    for t in range(t_obs):
        if t > 0:
            eps[t, :] = eps[t - 1, :] @ np.diag(rho * np.ones(obs_dim)) + v_t[t, :]
        else:
            eps[t, :] = v_t[t, :]
    # gen observables
    x = f @ np.random.rand(f.shape[1], obs_dim) + eps
    # fit models
    n_lags = 0
    start = timer()
    ddfm = DDFM(lags_input=n_lags,
                structure_encoder=(f.shape[1] * 6, f.shape[1] * 3, f.shape[1]),
                max_iter=100)
    ddfm.fit(pd.DataFrame(x))
    stop = timer()
    print('Elapsed time: ', timedelta(seconds=stop - start))
    # evaluate (Stock and Watson (2002a) and Doz, Giannone, and Reichlin (2006) and use the trace R2)
    f_hat = np.mean(ddfm.factors_ae, axis=0)
    precision_score = np.trace(
        f[n_lags:].T @ f_hat @ np.linalg.pinv(f_hat.T @ f_hat) @ f_hat.T @ f[n_lags:]) / np.trace(
        f[n_lags:].T @ f[n_lags:])
    print('Precision score non filtered: ', precision_score)
