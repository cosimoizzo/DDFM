import numpy as np
from pykalman import KalmanFilter
from pykalman.standard import _last_dims, _filter_predict, _filter_correct, _smooth


def _filter(
    transition_matrices,
    observation_matrices,
    transition_covariance,
    observation_covariance,
    transition_offsets,
    observation_offsets,
    initial_state_mean,
    initial_state_covariance,
    observations,
):
    """
    Modified version of the PyKalman (https://pypi.org/project/pykalman/) _filter.
    Changes: I modify the original version to deal with missing observations following Shumway and Stoffer (2000,1982).
    TODO: consider discussing with PyKalman community about integration.
    """
    n_timesteps = observations.shape[0]
    n_dim_state = len(initial_state_mean)
    n_dim_obs = observations.shape[1]

    predicted_state_means = np.zeros((n_timesteps, n_dim_state))
    predicted_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    kalman_gains = np.zeros((n_timesteps, n_dim_state, n_dim_obs))
    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

    for t in range(n_timesteps):
        if t == 0:
            predicted_state_means[t] = initial_state_mean
            predicted_state_covariances[t] = initial_state_covariance
        else:
            transition_matrix = _last_dims(transition_matrices, t - 1)
            transition_covariance = _last_dims(transition_covariance, t - 1)
            transition_offset = _last_dims(transition_offsets, t - 1, ndims=1)
            predicted_state_means[t], predicted_state_covariances[t] = _filter_predict(
                transition_matrix,
                transition_covariance,
                transition_offset,
                filtered_state_means[t - 1],
                filtered_state_covariances[t - 1],
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
            observation_covariance[nan_mask, nan_mask] = np.diag(
                observation_covariance
            )[nan_mask]
            observation_t_mod[nan_mask] = 0

        (kalman_gains[t], filtered_state_means[t], filtered_state_covariances[t]) = (
            _filter_correct(
                observation_matrix,
                observation_covariance,
                observation_offset,
                predicted_state_means[t],
                predicted_state_covariances[t],
                observation_t_mod,
            )
        )

    return (
        predicted_state_means,
        predicted_state_covariances,
        kalman_gains,
        filtered_state_means,
        filtered_state_covariances,
    )


class KFMod(KalmanFilter):
    def filter(self, x: np.ndarray):
        """
        Modified version of the PyKalman (https://pypi.org/project/pykalman/) filter.
        Changes: I modify the _filter function to deal with missing data.
        TODO: consider discussing with PyKalman community about integration.
        """
        Z = self._parse_observations(x)

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

        (_, _, _, filtered_state_means, filtered_state_covariances) = _filter(
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
        return (filtered_state_means, filtered_state_covariances)

    def smooth(self, x: np.ndarray):
        """
        Modified version of the PyKalman (https://pypi.org/project/pykalman/) smoother.
        Changes: I modify the _filter function to deal with missing data.
        TODO: consider discussing with PyKalman community about integration.
        """
        Z = self._parse_observations(x)

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

    def predict(self, x: np.ndarray, steps_ahead: int):
        """
        Predict observables, from 0 (fill missing) to steps_ahead forecasting horizon
        """
        smoothed_state_means, smoothed_state_covariances = self.smooth(x)
        predicted_state_mean = smoothed_state_means[-1]
        predicted_state_covariance = smoothed_state_covariances[-1]
        return self.predict_from_state(
            predicted_state_mean, predicted_state_covariance, steps_ahead
        )

    def predict_from_state(
        self,
        predicted_state_mean: np.ndarray,
        predicted_state_covariance: np.ndarray,
        steps_ahead: int,
    ):
        """
        Predict observables, from 0 (fill missing) to steps_ahead forecasting horizon using starting value from state
        distribution at time 0.
        """
        assert steps_ahead >= 0, "Steps Ahead must be positive."
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
        transition_matrix = _last_dims(transition_matrices, -1)
        transition_covariance = _last_dims(transition_covariance, -1)
        transition_offset = _last_dims(transition_offsets, -1, ndims=1)
        observation_matrix = _last_dims(observation_matrices, -1)
        observation_covariance = _last_dims(observation_covariance, -1)
        observation_offset = _last_dims(observation_offsets, -1, ndims=1)
        predicted_observation_mean = np.zeros(
            (steps_ahead + 1, observation_offset.shape[0])
        )
        predicted_observation_covariance = np.zeros(
            (steps_ahead + 1, observation_offset.shape[0], observation_offset.shape[0])
        )
        for i in range(steps_ahead + 1):
            predicted_observation_mean[i] = (
                np.dot(observation_matrix, predicted_state_mean) + observation_offset
            )
            predicted_observation_covariance[i] = (
                np.dot(
                    observation_matrix,
                    np.dot(predicted_state_covariance, observation_matrix.T),
                )
                + observation_covariance
            )
            predicted_state_mean, predicted_state_covariance = _filter_predict(
                transition_matrix,
                transition_covariance,
                transition_offset,
                predicted_state_mean,
                predicted_state_covariance,
            )
        return predicted_observation_mean, predicted_observation_covariance

    def fill_na(self, x: np.ndarray):
        """
        Fill missing values in the observables
        """
        mean, cov = self.predict(x, steps_ahead=0)
        return mean[0, ...], cov[0, ...]
