from typing import Tuple
import numpy as np
from pykalman import KalmanFilter
from pykalman.standard import _last_dims, _filter_predict, _filter_correct, _smooth


def _filter(transition_matrices, observation_matrices, transition_covariance,
            observation_covariance, transition_offsets, observation_offsets,
            initial_state_mean, initial_state_covariance, observations):
    """
    Modified version of the PyKalman _filter.
    Changes: I modify the original version to deal with missing observations following Shumway and Stoffer (2000,1982).
    TODO: consider discussing with PyKalman community about integration.
    """
    n_timesteps = observations.shape[0]
    n_dim_state = len(initial_state_mean)
    n_dim_obs = observations.shape[1]

    predicted_state_means = np.zeros((n_timesteps, n_dim_state))
    predicted_state_covariances = np.zeros(
        (n_timesteps, n_dim_state, n_dim_state)
    )
    kalman_gains = np.zeros((n_timesteps, n_dim_state, n_dim_obs))
    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros(
        (n_timesteps, n_dim_state, n_dim_state)
    )

    for t in range(n_timesteps):
        if t == 0:
            predicted_state_means[t] = initial_state_mean
            predicted_state_covariances[t] = initial_state_covariance
        else:
            transition_matrix = _last_dims(transition_matrices, t - 1)
            transition_covariance = _last_dims(transition_covariance, t - 1)
            transition_offset = _last_dims(transition_offsets, t - 1, ndims=1)
            predicted_state_means[t], predicted_state_covariances[t] = (
                _filter_predict(
                    transition_matrix,
                    transition_covariance,
                    transition_offset,
                    filtered_state_means[t - 1],
                    filtered_state_covariances[t - 1]
                )
            )

        observation_matrix = _last_dims(observation_matrices, t).copy()
        observation_covariance = _last_dims(observation_covariance, t).copy()
        observation_offset = _last_dims(observation_offsets, t, ndims=1).copy()

        # modification: look for missing values and follow the approach of Shumway and Stoffer (2000,1982)
        observation_t_mod = observations[t].copy()
        nan_mask = np.isnan(observation_t_mod)
        if np.any(nan_mask):
            observation_offset[nan_mask] = 0
            observation_matrix[nan_mask, :] = 0
            observation_covariance[nan_mask, :] = 0
            observation_covariance[:, nan_mask] = 0
            observation_covariance[nan_mask, nan_mask] = np.diag(observation_covariance)[nan_mask]
            observation_t_mod[nan_mask] = 0

        (kalman_gains[t], filtered_state_means[t],
         filtered_state_covariances[t]) = (
            _filter_correct(observation_matrix,
                            observation_covariance,
                            observation_offset,
                            predicted_state_means[t],
                            predicted_state_covariances[t],
                            observation_t_mod
                            )
        )

    return (predicted_state_means, predicted_state_covariances,
            kalman_gains, filtered_state_means,
            filtered_state_covariances)


class KalmanFilterMod(KalmanFilter):
    def filter(self, X):
        """
        Modified version of the PyKalman filter.
        Changes: I modify the _filter function to deal with missing data.
        TODO: consider discussing with PyKalman community about integration.
        """
        Z = self._parse_observations(X)

        (transition_matrices, transition_offsets, transition_covariance,
         observation_matrices, observation_offsets, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        (_, _, _, filtered_state_means,
         filtered_state_covariances) = (
            _filter(
                transition_matrices, observation_matrices,
                transition_covariance, observation_covariance,
                transition_offsets, observation_offsets,
                initial_state_mean, initial_state_covariance,
                Z
            )
        )
        return (filtered_state_means, filtered_state_covariances)

    def smooth(self, X):
        """
        Modified version of the PyKalman smoother.
        Changes: I modify the _filter function to deal with missing data.
        TODO: consider discussing with PyKalman community about integration.
        """
        Z = self._parse_observations(X)

        (
            transition_matrices,
            transition_offsets,
            transition_covariance,
            observation_matrices,
            observation_offsets,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        (
            predicted_state_means,
            predicted_state_covariances,
            _,
            filtered_state_means,
            filtered_state_covariances,
        ) = _filter(
            transition_matrices,
            observation_matrices,
            transition_covariance,
            observation_covariance,
            transition_offsets,
            observation_offsets,
            initial_state_mean,
            initial_state_covariance,
            Z,
        )
        (smoothed_state_means, smoothed_state_covariances) = _smooth(
            transition_matrices,
            filtered_state_means,
            filtered_state_covariances,
            predicted_state_means,
            predicted_state_covariances,
        )[:2]
        return (smoothed_state_means, smoothed_state_covariances)


class StateSpace:
    """
    State-space model wrapper around modified PyKalman.
    """

    def __init__(self, mean_z: np.ndarray, sigma_z: np.ndarray, transition_params: dict, measurement_params: dict,
                 filter_type: str = "KalmanFilter"):
        """
        The init method will build the state space model according to the selected filter.
        Args:
            mean_z: mean of the measurement variable
            sigma_z: standard deviation of the measurement variable
            transition_params: parameters of the transition equation
            measurement_params: parameters of the measurement equation
            filter_type: the type of filter selected

        """
        super().__init__()
        self.mean_z = mean_z
        self.sigma_z = sigma_z
        # init parameters of the state space to None
        self.H = None
        self.R = None
        self.F = None
        self.Q = None
        self.ssm_repr = None
        if filter_type == "KalmanFilter":
            # build a linear gaussian state-space model
            self.build_lgssm(transition_params, measurement_params)
        else:
            raise NotImplementedError("Only KalmanFilter is implemented at the moment.")

    def build_lgssm(self, transition_params: dict, measurement_params: dict) -> None:
        """
        This method builds a linear gaussian state space model of the following form:
            measurement: z_t = H x_t + v_t; v_t ∼ N(0, R)
            transition: x_t = F x_t-1 + w_t; w_t ∼ N(0, Q)
        Args:
            transition_params: parameters of the transition equation
            measurement_params: parameters of the measurement equation

        Returns:
            None, it updates the class attributes.
        """

        self.H, self.R = measurement_params["H"], measurement_params["R"]
        self.F, self.Q = transition_params["F"], transition_params["Q"]
        self.ssm_repr = KalmanFilterMod(transition_matrices=self.F, observation_matrices=self.H,
                                              transition_covariance=self.Q, observation_covariance=self.R)

    def predict(self, x_hat_start: np.ndarray, sigma_x_hat_start: np.ndarray, steps_ahead: int = 1) -> dict:
        raise NotImplementedError("TODO")

    def filter(self, z: np.ndarray, standardize: bool = False) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        State Space filtering step

        Args:
            z: observable realised values
            standardize: whether to standardize the inputs or not

        Returns:
            filtered states and variance-covariance matrix
        """
        z_cpy = z.copy()
        if standardize:
            z_cpy = (z_cpy - self.mean_z) / self.sigma_z
        # make dimensions consistent
        if z_cpy.ndim == 1:
            z_cpy = np.reshape(z_cpy, (1, z_cpy.shape[0]))
        filtered_state_means, filtered_state_covariances = self.ssm_repr.filter(z_cpy)
        return filtered_state_means, filtered_state_covariances

    def smooth(self, z: np.ndarray, standardize: bool = False) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        State Space smoothing step

        Args:
            z: observable realised values
            standardize: whether to standardize the inputs or not

        Returns:
            smoothed states and variance-covariance matrix
        """
        z_cpy = z.copy()
        if standardize:
            z_cpy = (z_cpy - self.mean_z) / self.sigma_z
        # make dimensions consistent
        if z_cpy.ndim == 1:
            z_cpy = np.reshape(z_cpy, (1, z_cpy.shape[0]))
        smoothed_state_means, smoothed_state_covariances = self.ssm_repr.smooth(z_cpy)
        return smoothed_state_means, smoothed_state_covariances
