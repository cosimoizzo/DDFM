from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, PredefinedSplit
import numpy as np
from models.ddfm import DDFM
from sklearn.base import BaseEstimator
from tools.getters_converters_tools import get_transition_params


class DdfmCv(BaseEstimator):
    """
    A wrapper around the DDFM object to provide a simplified interface to users which does inheritance from
    sklearn BaseEstimator.
    """

    def __init__(self, lags_input: int = 0,
                 structure_encoder: tuple = (16, 4),
                 structure_decoder: tuple = None,
                 factor_oder: int = 2,
                 link: str = "relu",
                 disp: int = 100):
        self.lags_input = lags_input
        self.structure_encoder = structure_encoder
        self.structure_decoder = structure_decoder
        self.factor_oder = factor_oder
        self.disp = disp
        self.link = link

    def fit(self, x, y=0):
        self.model = DDFM(x, lags_input=self.lags_input,
                          structure_encoder=self.structure_encoder,
                          structure_decoder=self.structure_decoder,
                          link=self.link,
                          factor_oder=self.factor_oder,
                          disp=self.disp)
        self.model.fit()
        return self

    def predict(self, x, y=0, steps_ahead: int = 1):
        """
        Prediction method for point-estimate.
        Steps:
        1 - build inputs
        2 - encode data and construct idiosyncratic components
        3 - predict latent states
        4 - decode data and add idiosyncratic part
        """
        x_std = (x - self.model.mean_z) / self.model.sigma_z
        f_t = self.model.encoder(self.model.build_inputs(x_std))
        eps_t = x_std[self.lags_input:] - self.model.decoder(f_t)
        F, _, _, _, x_t = get_transition_params(f_t, eps_t,
                                                factor_oder=self.factor_oder,
                                                bool_no_miss=~np.isnan(x_std[self.lags_input:]))
        factor_pred = F.dot(x_t)
        for t in range(1, steps_ahead):
            factor_pred = F.dot(factor_pred)

        out = (self.model.decoder(factor_pred[:self.structure_encoder[-1]]) + factor_pred[-x_std.shape[1]:])[-1, :]

        return out * self.model.sigma_z + self.model.mean_z


class Validate:
    """
    Validate models with 1-step ahead prediction
    """

    def __init__(self, x, cv_type="hold_out_last_n_obs", n_jobs=1,
                 refit=True, validation_portion=0.1, random_state=3, verbose=1):
        self.x = x[:-1]
        self.y = x[1:]
        self.cv_type = cv_type
        self.n_jobs = n_jobs
        self.refit = refit
        self.n_splits_obs = int(validation_portion * x.shape[0])
        self.random_state = random_state
        self.verbose = verbose

    def grid_search_cross_validate(self, model, hyper_parameters):
        if self.cv_type == "standard":
            cv = KFold(n_splits=self.n_splits_obs, random_state=self.random_state)
        elif self.cv_type == "tssplit":
            cv = TimeSeriesSplit(n_splits=self.n_splits_obs)
        elif self.cv_type == "hold_out_last_n_obs":
            select_test = np.ones(self.x.shape[0])
            select_test[:-self.n_splits_obs] = -1
            cv = PredefinedSplit(list(select_test))
        else:
            raise ValueError("method ", self.cv_type, " not available!")
        clf = GridSearchCV(model, hyper_parameters, cv=cv, n_jobs=self.n_jobs, refit=self.refit,
                           # scoring=make_scorer(self.score, greater_is_better=False),
                           scoring='neg_mean_squared_error',
                           verbose=self.verbose)
        return clf.fit(self.x, self.y)
