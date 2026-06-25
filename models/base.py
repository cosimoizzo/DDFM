from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class FactorModel(ABC):
    """
    Abstract base class for all factor models.

    All models share the same interface:
    - fit(data): train the model on observed data, may contain NaN
    - get_factors(data): return estimated latent factors as a DataFrame (T, r)
    - fill_na(data): impute missing values using the fitted model
    - predict(data, steps_ahead): forecast data at horizons h=0..steps_ahead

    All methods accept and return pd.DataFrame.

    Standardisation convention (consistent with DDFM._training_data_set_up):
        During fit(), mean_data and sigma_data are computed from the training
        data (NaN-aware, ddof=1) and stored. All subclasses must standardise
        their input via _std() before model fitting and invert via _unstd()
        when returning observable-space outputs (fill_na, predict).
        get_factors() returns latent factors which live in their own space and
        are NOT un-standardised.

    predict() convention (consistent with DDFM):
        Returns a DataFrame of shape (steps_ahead + 1, N) where:
          row 0 = model reconstruction at the last observed point (h=0)
          row h = h-step-ahead forecast (h = 1 .. steps_ahead)
        Index is range(steps_ahead + 1); columns are the observable variables.
    """

    def __init__(self, r: Optional[int] = None):
        self.r = r
        self._fitted = False
        self.mean_data: Optional[np.ndarray] = None
        self.sigma_data: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Standardisation helpers
    # ------------------------------------------------------------------

    def _std(self, data: pd.DataFrame) -> np.ndarray:
        """Standardise data using training mean and std. Returns numpy array."""
        return (data.values - self.mean_data) / self.sigma_data

    def _unstd(self, x: np.ndarray) -> np.ndarray:
        """Reverse standardisation. Returns numpy array in original scale."""
        return x * self.sigma_data + self.mean_data

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        Train the model on observed data.

        Must store mean_data and sigma_data from the training data and
        raise ValueError if any variable has zero variance.

        Args:
            data : pd.DataFrame, shape (T, N)
                Observed panel; may contain NaN for missing values.

        Returns:

        """
        ...

    @abstractmethod
    def get_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Return estimated latent factors using the fitted model.

        Args:
            data: pd.DataFrame, shape (T, N)

        Returns:
            pd.DataFrame, shape (T, r), columns ["f1", "f2", ..., "fr"]
        """
        ...

    @abstractmethod
    def fill_na(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in data using the fitted model.
        Output is in the original (un-standardised) scale.

        Args:
            data: pd.DataFrame, shape (T, N)

        Returns:
            pd.DataFrame, shape (T, N), same columns and index as input, no NaN
        """
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame, steps_ahead: int) -> pd.DataFrame:
        """
        Forecast data at horizons h=0 to h=steps_ahead.
        Output is in the original (un-standardised) scale.

        Args:
            data: pd.DataFrame, shape (T, N)
                Observed panel (may contain NaN).
            steps_ahead:

        Returns:
            pd.DataFrame, shape (steps_ahead + 1, N)
                Row 0 is the model's reconstruction at the last observed point.
                Rows 1...steps_ahead are the forecasts at each horizon.
                Index is range(steps_ahead + 1); columns match data.columns.

        """
        ...
