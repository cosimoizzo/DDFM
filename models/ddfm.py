import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers
from models.state_space import StateSpace
from tools.loss_tools import mse_missing, convergence_checker
from tools.getters_converters_tools import convert_decoder_to_numpy, get_transition_params, get_idio


# tf.config.run_functions_eagerly(True)


class DDFM:
    """
    A class implementing Deep Dynamic Factor Models.
    """

    def __init__(self, data: pd.DataFrame, lags_input: int = 0, structure_encoder: tuple = (16, 4),
                 structure_decoder: tuple = None, use_bias: bool = True, factor_oder: int = 2, seed: int = 3,
                 batch_norm: bool = True, link: str = 'relu', learning_rate: float = 0.005,
                 optimizer: str = 'Adam', decay_learning_rate: bool = True,
                 epochs: int = 100, batch_size: int = 100, max_iter=200, tolerance: float = 0.0005,
                 disp: int = 10):
        """

        Args:
            data: input data used for model training
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
        super().__init__()
        # common factors
        self.factor_oder = factor_oder
        if factor_oder not in [1, 2]:
            raise ValueError('factor_oder must be 1 or 2')
        # z is the observable
        print("@Info - Note: Sorting data.")
        data.sort_index(inplace=True)
        self.mean_z = data.mean().values
        self.sigma_z = data.std().values
        self.data = (data - self.mean_z) / self.sigma_z
        # keep track of the missings locations
        self.bool_miss = self.data.isnull()[lags_input:].values
        self.bool_no_miss = self.bool_miss == False
        # create copies of the original data (needed for training and pre-training)
        self.data_mod_only_miss, self.data_mod, self.data_tmp = self.data.copy(), self.data.copy(), self.data.copy()
        self.z_actual = self.data[lags_input:].values
        # autoencoder structure
        self.lags_input = lags_input
        self.structure_encoder = structure_encoder
        self.structure_decoder = structure_decoder
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.link = link
        # self.start_quarterly = start_quarterly
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
        # attributes to be populated
        self.loss_now = None
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.eps = None
        self.factors = None
        self.last_neurons = None
        self.factors_filtered = None
        self.state_space = None
        self.state_space_dict = dict()
        self.latents = dict()

    def build_inputs(self, interpolate: bool = True) -> None:
        """
        Method to build the inputs of the model from the dataset.
        Args:
            interpolate: whether to interpolate or not the missing values

        Returns:
            None, it updates the data attributes of the class
        """

        # create dict with variables and their lagged values
        new_dict = {}
        for col_name in self.data_mod:
            new_dict[col_name] = self.data_mod[col_name]
            # create lagged Series
            for lag in range(self.lags_input):
                new_dict['%s_lag%d' % (col_name, lag + 1)] = self.data_mod[col_name].shift(lag + 1)
        # convert to dataframe
        self.data_tmp = pd.DataFrame(new_dict, index=self.data_mod.index)
        # drop initial nans
        self.data_tmp = self.data_tmp[self.lags_input:]
        # interpolate
        if interpolate and self.data_tmp.isna().sum().sum() > 0:
            # self._x_.interpolate(method='spline', limit_direction='forward', inplace=True, order=3)
            self.data_tmp.interpolate(method='spline', limit_direction='both', inplace=True, order=3)

    def build_model(self) -> None:
        """
        Method to build the keras model.
        Returns:
            None, it updates the attributes related to the autoencoder.
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

    def pre_train(self, min_obs: int = 50, mult_epoch_pre: int = 1) -> None:
        """
        Method to carry out pre-training of the model.
        Args:
            min_obs: minimum number of observations for pre-training with no interpolation for missings
            mult_epoch_pre: coefficient to be multiplied to number of epochs to deliver to the total number of epochs
                for pre-training

        Returns:
            None, it updates the autoencoders attributes
        """
        # build inputs without interpolation
        self.build_inputs(interpolate=False)
        # check number of observations, and if not enough then interpolate
        if len(self.data_tmp.dropna()) >= min_obs:
            inpt_pre_train = self.data_tmp.dropna().values
            self.autoencoder.compile(optimizer=self.optimizer, loss='mse')
        else:
            self.build_inputs()
            inpt_pre_train = self.data_tmp.dropna().values
            self.autoencoder.compile(optimizer=self.optimizer, loss=mse_missing)
        # build output
        oupt_pre_train = self.data_tmp.dropna()[self.data.columns].values
        # fit (pre-train) autoencoder
        self.autoencoder.fit(inpt_pre_train, oupt_pre_train, epochs=self.epoch * mult_epoch_pre,
                             batch_size=self.batch_size,
                             verbose=0)

    def train(self) -> None:
        """
        Method to train a Deep Dynamic Factor Model (see Algorithm 1 from the paper.)
        Returns:
            None, it updates attributes.
        """
        # re-compile the autoencoder to re-init the optimizer and possibly change the objective
        self.autoencoder.compile(optimizer=self.optimizer, loss=mse_missing)
        # construct initial input data
        self.build_inputs()
        # make prediction
        prediction_iter = self.autoencoder.predict(self.data_tmp.values)
        # update missings
        self.data_mod_only_miss.values[self.lags_input:][self.bool_miss] = prediction_iter[self.bool_miss]
        # get idio
        self.eps = self.data_tmp[self.data.columns].values - prediction_iter
        # init counters
        iter = 0
        not_converged = True
        # start MCMC
        while not_converged and iter < self.max_iter:
            # get idio distr
            phi, mu_eps, std_eps = get_idio(self.eps, self.bool_no_miss)
            # subtract conditional AR-idio mean from x
            self.data_mod[self.lags_input + 1:] = self.data_mod_only_miss[self.lags_input + 1:] - self.eps[:-1, :] @ phi
            # for first observations set to 0 the idio
            self.data_mod[:self.lags_input + 1] = self.data_mod_only_miss[:self.lags_input + 1]
            # gen data_tmp from filtered inputs (self.data_mod above)
            self.build_inputs()
            # gen MC samples for idio (dims = Sim x T x D)
            eps_draws = self.rng.multivariate_normal(mu_eps, np.diag(std_eps), (self.epoch, self.data_tmp.shape[0]))
            # init noisy inputs (dims = Sim x T x D_with_lags)
            x_sim_den = np.zeros((eps_draws.shape[0], eps_draws.shape[1], eps_draws.shape[2] * (self.lags_input + 1)))
            # loop over them (MC step)
            for i in range(self.epoch):
                x_sim_den[i, :, :] = self.data_tmp.copy()
                # corrupt input data, only current observations
                x_sim_den[i, :, :eps_draws[i, :, :].shape[1]] = x_sim_den[i, :,
                                                                :eps_draws[i, :, :].shape[1]] - eps_draws[i, :, :]
                # fit autoencoder
                self.autoencoder.fit(x_sim_den[i, :, :], self.z_actual, epochs=1, batch_size=self.batch_size, verbose=0)
            # update factors: average over all predictions from the MC samples
            self.factors = np.array([self.encoder(x_sim_den[i, :, :]) for i in range(x_sim_den.shape[0])])
            # check convergence
            prediction_iter = np.mean(np.array([self.decoder(self.factors[i, :, :]) for i in range(self.factors.shape[0]
                                                                                                   )]), axis=0)
            if iter > 1:
                delta, self.loss_now = convergence_checker(prediction_prev_iter, prediction_iter, self.z_actual)
                if iter % self.disp == 0:
                    print(f'@Info: iteration: {iter} - new loss: {self.loss_now} - delta: {delta}')
                if delta < self.tolerance:
                    not_converged = False
                    print(f'@Info: Convergence achieved in {iter} iterations - new loss: {self.loss_now} - delta: {delta} < {self.tolerance}')
            # store previous prediction to monitor convergence
            prediction_prev_iter = prediction_iter.copy()
            # update missings
            self.data_mod_only_miss.values[self.lags_input:][self.bool_miss] = prediction_iter[self.bool_miss]
            # update idio
            self.eps = self.data_mod_only_miss.values[self.lags_input:] - prediction_iter
            iter += 1

        # get last neurons (making difference between nonlinear and linear decoder)
        if self.structure_decoder is None:
            self.last_neurons = self.factors
        else:
            decoder_for_last_neuron = keras.Model(self.decoder.input,
                                                  self.decoder.get_layer(self.decoder.layers[-2].name).output)
            self.last_neurons = np.array([decoder_for_last_neuron(self.encoder(x_sim_den[i, :, :])) for i in
                                          range(x_sim_den.shape[0])])

        if not_converged:
            print("@Info: Convergence not achieved within the maximum number of iteration!")

    def build_state_space(self) -> None:
        """
        Method to build the state space model from the autoencoder (decoder).
            measurement: z_t = H x_t + v_t; v_t ∼ N(0, R)
            transition: x_t = F x_t-1 + w_t; w_t ∼ N(0, Q)
        Returns:
            None, it updates the class attributes.
        """
        # extract common factors
        f_t = np.mean(self.factors, axis=0)
        # idio components
        eps_t = self.eps
        # get params from decoder (measurement equation)
        bs, H = convert_decoder_to_numpy(self.decoder, self.use_bias, self.factor_oder,
                                         structure_decoder=self.structure_decoder)
        # modify mean with the bias term
        self.mean_z = self.mean_z + bs * self.sigma_z
        # get transition equation params
        F, Q, mu_0, sigma_0, x_t = get_transition_params(f_t, eps_t, factor_oder=self.factor_oder,
                                                         bool_no_miss=self.bool_no_miss)
        # insert in dictionary
        self.state_space_dict["transition"] = dict()
        self.state_space_dict["transition"]["F"] = F
        self.state_space_dict["transition"]["Q"] = Q
        self.state_space_dict["transition"]["mu_0"] = mu_0
        self.state_space_dict["transition"]["Σ_0"] = sigma_0
        self.latents["ae_states"] = x_t
        # we set this to a small number, but we could cross-validate to control the signal-to-noise ratio
        R = np.eye(eps_t.shape[1]) * 1e-15
        # H = None
        self.state_space_dict["measurement"] = dict()
        self.state_space_dict["measurement"]["H"] = H
        self.state_space_dict["measurement"]["R"] = R
        self.state_space = StateSpace(self.mean_z, self.sigma_z,
                                      self.state_space_dict["transition"], self.state_space_dict["measurement"],
                                      filter_type=self.filter_type)

    def fit(self, build_state_space: bool = False):
        """
        Method to fit the Deep Dynamic Factor Model.
        Returns:
            None, it updates the class attributes.
        """
        self.build_model()
        self.pre_train()
        self.train()
        if build_state_space:
            self.build_state_space()
            # get filtered factors
            self.latents["filtered"], self.latents["sigma_kf"] = self.filter(self.data.values)
            self.factors_filtered = self.latents["filtered"][:, 1:self.structure_encoder[-1] + 1]

    def filter(self, z_t: np.ndarray, standardize: bool = False) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Method to carry out the filtering in state-space.
        Args:
            z_t: observable realised values
            standardize: whether to standardize the inputs or not

        Returns:
            filtered states and variance-covariance matrix
        """
        return self.state_space.filter(z_t, standardize=standardize)

    def predict(self, x_hat_start: np.ndarray, sigma_x_hat_start: np.ndarray, steps_ahead: int = 1) -> dict:
        """
        Method to carry out the prediction in state-space.
        Args:
            x_hat_start: starting values for the state mean
            sigma_x_hat_start: starting values for the state variance covariance matrix
            steps_ahead: number of steps ahead

        Returns:
            A dictionary with predicted states and measurement mean and variances.
        """
        return self.state_space.predict(x_hat_start, sigma_x_hat_start, steps_ahead=steps_ahead)


if __name__ == "__main__":
    # import tensorflow as tf
    import sklearn  # maybe we can remove this dependency
    import random
    from timeit import default_timer as timer
    from datetime import timedelta

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
    ddfm = DDFM(pd.DataFrame(x), lags_input=n_lags,
                structure_encoder=(f.shape[1] * 6, f.shape[1] * 3, f.shape[1]),
                max_iter=100)
    ddfm.fit()
    stop = timer()
    print('Elapsed time: ', timedelta(seconds=stop - start))
    # evaluate (Stock and Watson (2002a) and Doz, Giannone, and Reichlin (2006) and use the trace R2)
    f_hat = np.mean(ddfm.factors, axis=0)
    precision_score = np.trace(
        f[n_lags:].T @ f_hat @ np.linalg.pinv(f_hat.T @ f_hat) @ f_hat.T @ f[n_lags:]) / np.trace(
        f[n_lags:].T @ f[n_lags:])
    print('Precision score non filtered: ', precision_score)
