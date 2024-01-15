import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, PredefinedSplit
import numpy as np
from models.ddfm import DDFM
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator
from tools.getters_converters_tools import get_transition_params


class Ddfm_simple(BaseEstimator):
    """
    A wrapper around the DDFM object to provide a simplified interface to users which does inheritance from
    sklearn BaseEstimator.
    """

    def __init__(self, lags_input: int = 0,
                 structure_encoder: tuple = (16, 4),
                 symmetric_decoder: bool = False,
                 factor_oder: int = 2,
                 link: str = "relu",
                 seed: int = 3,
                 disp: int = 100,
                 n_steps_ahead: int = 1):
        self.lags_input = lags_input
        self.structure_encoder = structure_encoder
        self.symmetric_decoder = symmetric_decoder
        self.factor_oder = factor_oder
        self.disp = disp
        self.link = link
        self.seed = seed
        self.n_steps_ahead = n_steps_ahead
        self.model = None

    def fit(self, x, y=0):
        if self.symmetric_decoder:
            structure_decoder = None
        else:
            structure_decoder = self.structure_encoder[::-1][1:]
        self.model = DDFM(x, lags_input=self.lags_input,
                          structure_encoder=self.structure_encoder,
                          structure_decoder=structure_decoder,
                          seed=self.seed,
                          link=self.link,
                          factor_oder=self.factor_oder,
                          disp=self.disp)
        self.model.fit()
        ## transition matrix
        # extract common factors
        f_t = np.mean(self.model.factors, axis=0)
        # idio components
        eps_t = self.model.eps
        # get transition equation params
        F, _, _, _, _ = get_transition_params(f_t, eps_t, factor_oder=self.model.factor_oder,
                                              bool_no_miss=self.model.bool_no_miss)
        self.model.F = F
        return self

    def predict(self, x, y=0, n_steps_ahead: int = None):
        """
        Prediction method for point-estimate up to and including steps_ahead.
        Here, we make a simplification without the use of filters.
        Steps:
        1 - build inputs
        2 - encode data and construct idiosyncratic components
        3 - predict latent states
        4 - decode data and add idiosyncratic part
        """
        if n_steps_ahead is None:
            steps_ahead = self.n_steps_ahead
        else:
            steps_ahead = n_steps_ahead
        x_std = (x - self.model.mean_z) / self.model.sigma_z
        f_t = self.model.encoder.predict(self.model.build_inputs(interpolate=True, data_raw=x_std).values)
        eps_t = x_std[self.lags_input:].values - self.model.decoder.predict(f_t)
        if self.model.factor_oder == 2:
            x_t = np.vstack((f_t[1:, :].T, f_t[:-1, :].T, eps_t[1:, :].T))
        elif self.model.factor_oder == 1:
            x_t = np.vstack((f_t.T, eps_t.T))
        else:
            raise NotImplementedError("Only VAR(2) or VAR(1) for common factors at the moment.")
        x_t[np.isnan(x_t)] = 0
        # H x T x N
        preds = np.ones((steps_ahead, x_t.shape[1], x.shape[1])) * np.nan
        F = self.model.F
        factor_pred = F.dot(x_t)
        preds[0, :, :] = self._get_predictions_from_factors(factor_pred.T)
        for t in range(1, steps_ahead):
            factor_pred = F.dot(factor_pred)
            preds[t, :, :] = self._get_predictions_from_factors(factor_pred.T)

        return np.squeeze(preds)

    def _get_predictions_from_factors(self, factors):
        out_common = self.model.decoder.predict(factors[:, :self.structure_encoder[-1]])
        out_idio = factors[:, -self.model.data.shape[1]:]
        out = out_common + out_idio
        out = out * self.model.sigma_z + self.model.mean_z
        return out


class Validate:
    """
    Validate models with n-steps ahead prediction
    """

    def __init__(self, x,
                 cv_type="hold_out_last_n",
                 n_jobs=1,
                 refit=True,
                 n_splits=2,
                 max_train_size=None,
                 test_size=10,
                 verbose=1,
                 n_steps_ahead=1
                 ):
        self.x = x.copy()
        self.y = x.copy()
        self.cv_type = cv_type
        self.n_jobs = n_jobs
        self.refit = refit
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.verbose = verbose
        self.n_steps_ahead = n_steps_ahead

    def score(self, y_true, y_pred):
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        if self.n_steps_ahead == 1:
            y_true_adj = np.concatenate([y_true[1:, :], np.nan * np.ones((1, y_true.shape[1]))]) # one step ahead target
            # to align dimensions
            if y_true_adj.shape[0] > y_pred.shape[0]:
                y_true_adj = y_true_adj[-y_pred.shape[0]:, :]
        else:
            y_true_adj = np.ones_like(y_pred) * np.nan
            for j in range(self.n_steps_ahead):
                thisy = np.concatenate([y_true[j + 1:, :], np.nan * np.ones((j+1, y_true.shape[1]))])
                # to align dimensions
                if thisy.shape[0] > y_pred.shape[1]:
                    thisy = thisy[-y_pred.shape[1]:, :]
                y_true_adj[j, :, :] = thisy
        #mean_squared_error = np.nanmean((y_true_adj - y_pred) ** 2)
        mean_absolute_error = np.nanmean(np.abs(y_pred - y_true_adj) / (1e-6 + np.abs(y_true_adj)))
        #print("mean_absolute_percentage_error", mean_absolute_error)
        return mean_absolute_error

    def grid_search_cross_validate(self, model, hyper_parameters):
        if self.cv_type == "tssplit":
            cv = TimeSeriesSplit(max_train_size=self.max_train_size,
                                 n_splits=self.n_splits, test_size=self.test_size)
        elif self.cv_type == "hold_out_last_n":
            select_test = np.ones(self.x.shape[0])
            select_test[:-self.test_size] = -1
            cv = PredefinedSplit(list(select_test))
        else:
            raise ValueError(f"Method {self.cv_type} not available!")
        clf = GridSearchCV(model, hyper_parameters, cv=cv, n_jobs=self.n_jobs, refit=self.refit,
                           scoring=make_scorer(self.score, greater_is_better=False),
                           verbose=self.verbose)
        return clf.fit(self.x, self.y)
