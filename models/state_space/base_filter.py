from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseFilter(ABC):
    @abstractmethod
    def smooth(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def filter(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

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
