from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from statsmodels.multivariate.pca import PCA as SM_PCA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg

from models.base import FactorModel


def _fit_var_or_ar(f_hat: np.ndarray, lags: int):
    """
    Fit VAR(lags) if f_hat has >= 2 columns, else AR(lags) on the single column.
    Returns a result object with a .forecast(y, steps) method.
    """
    if f_hat.ndim == 1:
        f_hat = f_hat.reshape(-1, 1)
    if f_hat.shape[1] >= 2:
        return VAR(f_hat).fit(maxlags=lags, ic=None)
    else:
        return _ARResultWrapper(f_hat[:, 0], lags)


class _ARResultWrapper:
    """Wraps AutoReg result to mimic the VAR result .forecast(y, steps) API."""

    def __init__(self, y: np.ndarray, lags: int):
        self._lags = lags
        self._ar_result = AutoReg(y, lags=lags).fit()

    def forecast(self, y: np.ndarray, steps: int) -> np.ndarray:
        y1 = y[:, 0] if y.ndim == 2 else y
        return self._ar_result.apply(y1).forecast(steps=steps).reshape(-1, 1)


class VARPCA(FactorModel):
    """
    Two-stage factor model:
      1. PCA extracts r latent factors from standardised x.
      2. VAR(p) models the temporal dynamics of those factors.

    Input data are standardised (zero mean, unit variance) using training
    statistics before PCA fitting. Outputs of fill_na and predict are
    returned in the original scale.

    When squared_pc=True, squared principal component is used: PC is applied on the input variables and their
        squared values (Bai and Ng, 2007).

    If missing data are present, then they are handled via the EM algorithm built into statsmodels PCA (missing='fill-em').
    """

    def __init__(
        self,
        r: int,
        var_lags: int = 1,
        squared_pc: bool = False,
        gls: bool = False,
        tol_em: float = 5e-8,
        max_em_iter: int = 100,
    ):
        super().__init__(r)
        self.var_lags = var_lags
        self.squared_pc = squared_pc
        self.gls = gls
        self.tol_em = tol_em
        self.max_em_iter = max_em_iter
        self._pca = None
        self.cols = None
        self._cols_linear = None
        self._var_result = None

    def _add_squared(self, data: pd.DataFrame):
        self._cols_linear = list(data.columns)
        cols_non_lin = [f"sq_{col}" for col in self._cols_linear]
        df_squared = pd.DataFrame(np.square(data.values), columns=cols_non_lin, index=data.index)
        return pd.concat([data, df_squared], axis=1)

    def _unstd_linear(self, x: np.ndarray) -> np.ndarray:
        return x * self.sigma_data[:len(self._cols_linear)] + self.mean_data[:len(self._cols_linear)]

    def _std_linear(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_data[:len(self._cols_linear)]) / self.sigma_data[:len(self._cols_linear)]

    def fit(self, data: pd.DataFrame) -> None:
        df = self._add_squared(data) if self.squared_pc else data.copy()
        self.cols = list(df)
        self.mean_data = df.mean().values
        self.sigma_data = df.std().values
        if np.any(self.sigma_data == 0):
            raise ValueError("Some variables have zero variance.")

        has_nan = np.any(np.isnan(df.values))

        if has_nan:
            self._pca = SM_PCA(
                self._std(df),
                ncomp=self.r,
                missing="fill-em",
                standardize=False,
                demean=False,
                normalize=False,
                gls=self.gls,
                tol_em=self.tol_em,
                max_em_iter=self.max_em_iter,
            )
        else:
            self._pca = SM_PCA(
                self._std(df),
                ncomp=self.r,
                missing=None,
                standardize=False,
                demean=False,
                normalize=False,
                gls=self.gls,
            )

        f_hat = self._pca.factors  # (T, r)
        self._var_result = _fit_var_or_ar(f_hat, self.var_lags)

        self._fitted = True

    def get_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.squared_pc:
            x = self._std(self._add_squared(data.interpolate(method="linear", limit_direction="both"))[self.cols])
            nan_idxs = np.isnan(data[self._cols_linear].values)
            nan_idxs = np.concatenate((nan_idxs, nan_idxs), axis=1)
        else:
            x = self._std(data[self.cols].interpolate(method="linear", limit_direction="both"))
            nan_idxs = np.isnan(data[self.cols].values)
        f_hat = x @ self._pca.loadings  # (T, r)
        if np.sum(nan_idxs)>0:
            f_hat = self._em_fillna(x, f_hat, nan_idxs)
        cols = [f"f{i + 1}" for i in range(self.r)]
        return pd.DataFrame(f_hat, index=data.index, columns=cols)

    def _em_fillna(self, x: np.ndarray, f_hat: np.ndarray, nan_idxs: np.ndarray) -> np.ndarray:
        x_rec_prev = x.copy()
        x_rec_now = x.copy()
        x_rec_now[nan_idxs] = (f_hat @ self._pca.loadings.T)[nan_idxs]
        _iter = 0
        while np.mean(np.abs(x_rec_prev[nan_idxs] - x_rec_now[nan_idxs])) > self.tol_em and _iter < self.max_em_iter:
            x_rec_prev = x_rec_now.copy()
            f_hat = x_rec_now @ self._pca.loadings
            x_rec_now[nan_idxs] = (f_hat @ self._pca.loadings.T)[nan_idxs]
            _iter += 1
        f_hat = x_rec_now @ self._pca.loadings
        return f_hat

    def fill_na(self, data: pd.DataFrame) -> pd.DataFrame:
        f_hat = self.get_factors(data).values
        x_recon_std = f_hat @ self._pca.loadings[:len(self._cols_linear),:].T if self.squared_pc else f_hat @ self._pca.loadings.T
        x_std = self._std_linear(data[self._cols_linear].values) if self.squared_pc else self._std(data[self.cols])
        mask = np.isnan(x_std)
        x_std[mask] = x_recon_std[mask]
        return pd.DataFrame(self._unstd_linear(x_std) if self.squared_pc else self._unstd(x_std),
                            index=data.index,
                            columns=self._cols_linear if self.squared_pc else self.cols)

    def predict(self, data: pd.DataFrame, steps_ahead: int) -> pd.DataFrame:
        f_hat = self.get_factors(data).values  # (T, r)
        # h=0: reconstruction of last point in standardised space
        pc_mapping = self._pca.loadings[:len(self._cols_linear),:].T if self.squared_pc else self._pca.loadings.T
        x_h0_std = (f_hat[-1:] @ pc_mapping)
        # h=1..steps_ahead: VAR forecast then decode
        forecast_f = self._var_result.forecast(f_hat, steps=steps_ahead)
        x_forecast_std = (forecast_f @ pc_mapping)
        x_pred_std = np.vstack([x_h0_std, x_forecast_std])  # (steps_ahead + 1, N)
        return pd.DataFrame(
            self._unstd_linear(x_pred_std) if self.squared_pc else self._unstd(x_pred_std),
            index=range(steps_ahead + 1),
            columns=self._cols_linear if self.squared_pc else self.cols,
        )


class DFM(FactorModel):
    """
    Dynamic Factor Model using statsmodels DynamicFactorMQ.

    Input data are standardised before passing to DynamicFactorMQ.
    Outputs of fill_na and predict are returned in the original scale.
    The Kalman filter handles missing values natively.
    """

    def __init__(self, r: int, factor_order: int = 1, k_endog_monthly: Optional[int] = None):
        """

        Args:
            r: number of factors
            factor_order: number of lags
            k_endog_monthly: number of monthly data series, first monthly then quarterly

        """
        super().__init__(r)
        self.factor_order = factor_order
        self.cols = None
        self.k_endog_monthly = k_endog_monthly
        self._model = None
        self._result = None

    def fit(self, data: pd.DataFrame) -> None:
        self.cols = list(data.columns)
        self.mean_data = data.mean().values
        self.sigma_data = data.std().values
        if np.any(self.sigma_data == 0):
            raise ValueError("Some variables have zero variance.")
        data_std = pd.DataFrame(self._std(data), index=data.index, columns=self.cols)
        self._model = DynamicFactorMQ(
            data_std,
            factors=1,
            factor_orders=self.factor_order,
            factor_multiplicities=self.r,
            idiosyncratic_ar1=True,
            k_endog_monthly=self.k_endog_monthly,
        )
        self._result = self._model.fit(disp=False)
        self._fitted = True

    def _std_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a standardised DataFrame with the same index/columns."""
        return pd.DataFrame(self._std(data[self.cols]), index=data.index, columns=self.cols)

    def get_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        result = self._result.apply(self._std_df(data), refit=False)
        smoothed = result.factors["smoothed"]
        smoothed.index = data.index
        smoothed.columns = [f"f{i + 1}" for i in range(self.r)]
        return smoothed

    def fill_na(self, data: pd.DataFrame) -> pd.DataFrame:
        result = self._result.apply(self._std_df(data), refit=False)
        result = result.get_prediction(information_set="smoothed")
        fitted_std = result.predicted_mean  # in standardised space
        data_std = self._std_df(data)
        filled_std = data_std.where(data_std.notna(), fitted_std).values
        return pd.DataFrame(
            self._unstd(filled_std), index=data.index, columns=self.cols
        )

    def predict(self, data: pd.DataFrame, steps_ahead: int) -> pd.DataFrame:
        result = self._result.apply(self._std_df(data), refit=False)
        # h=0: fitted value at last observed point (standardised space)
        x_h0_std = result.get_prediction(information_set="smoothed").predicted_mean.iloc[[-1]].values
        # h=1..steps_ahead: forecast (standardised space)
        x_forecast_std = result.forecast(steps=steps_ahead).values
        x_pred_std = np.vstack([x_h0_std, x_forecast_std])
        return pd.DataFrame(
            self._unstd(x_pred_std),
            index=range(steps_ahead + 1),
            columns=self.cols,
        )

    def predict_one_step_ahead(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        One-step-ahead predicted observables for the whole series in a single
        Kalman pass: row t = E[y_t | y_{0:t-1}], in the original scale.

        The Kalman filter's fitted values are exactly the in-sample one-step-ahead
        forecasts, so a single `apply` over [history; future] yields all rolling
        1-step OOS forecasts at once — O(T) versus O(T^2) re-filtering per step.
        """
        result = self._result.apply(self._std_df(data), refit=False)
        fitted_std = result.fittedvalues.values  # one-step-ahead, standardised
        return pd.DataFrame(
            self._unstd(fitted_std), index=data.index, columns=self.cols,
        )
