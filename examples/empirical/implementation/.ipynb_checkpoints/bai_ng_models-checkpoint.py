import pandas as pd
import statsmodels.api as sm

import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from numpy.linalg import pinv, eig

from pandas.tseries.offsets import BDay, Day, MonthEnd, DateOffset

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLars, Lars, LassoLarsCV
from sklearn.model_selection import TimeSeriesSplit, LeaveOneOut, GridSearchCV

import sys
parentdir = "/Users/paoloandreini/Desktop/github_repo/DDFM_correction_paper/DDFM/"
sys.path.insert(0, parentdir)
print("check and modify parentdir if erro import module")
from tools.bai_ng_utils import preprocess_data, rename_col_factors

class TargetedPredictiors:
    def __init__(
        self,
        X,
        y_,
        thresh_tstat=None,
        lag_x=None,
        target_freq=None,
        n_targeted_predictors=None
    ):
        import statsmodels.api as sm
        self.X = X
        self.y_ = y_
        if thresh_tstat is None:
            self.thresh_tstat = 1.65
        else:
            self.thresh_tstat=thresh_tstat
        if lag_x is None:
            self.lag_x =0
        if target_freq is None:
            self.target_freq="Q"
        else:
            self.target_freq=target_freq

        self.n_targeted_predictors = n_targeted_predictors
    
    def fit(self):
        if self.lag_x > 0:
            self.X = self.X_.shift(self.lag_x)
            self.X = self.X.dropna()
            self.y_ = self.y_.loc[self.X.index]
        if self.target_freq=="Q":
            self.freq=3
        else:
            self.freq=1
        
        self.y_ = self.y_.dropna()
        self.X = self.X.loc[self.y_.index]

        var_to_keep_all = []
        for i in self.X.columns:
            y_lags = pd.concat([self.y_.shift(i*self.freq) for i in range(1,4)], axis=1)
            y_lags.columns = [f"{self.y_.name}_lag{i}" for i in range(1,4)]
            x_ = pd.concat([y_lags, self.X[i]], axis=1)

            x_ = x_.dropna()
            y_reg = self.y_.loc[x_.index]

            model = sm.OLS(y_reg, x_)
            results = model.fit()
            tval = results.tvalues.abs()[i]

            if tval >= self.thresh_tstat:
                var_to_keep_all.append([i, tval])

        var_to_keep_all = pd.DataFrame(var_to_keep_all, columns=["var_name", "tval"])
        var_to_keep_all =var_to_keep_all.set_index("var_name").sort_values(by="tval", ascending=False)
        self.var_to_keep_all = var_to_keep_all.index
        
        # select vars
        if self.n_targeted_predictors is None:
            self.targeted_predictors = self.var_to_keep_all
        else:
            self.targeted_predictors = self.var_to_keep_all[:self.n_targeted_predictors]
            
class BaiNgModels:
    def __init__(self, data, transform_code_final,
                staz=True, standardize=True, target_name=None,
                start_date=pd.Timestamp("1990-01-01"),
                # models
                n_factors=None,
                model_name = None,
                lars_coef_number=None,
                target_freq=None,
                thresh_tstat= 1.96,
                n_targeted_predictors=None,
                lags_y = 5, # 5 is actually 4
                lags_f = 7, # 7 is actually 6):
                *args,
                **kwargs,
                ):
        
        if target_name is None:
            # set it as GDP
            self.target_name = "GDPC1"
            self.target_freq = "Q"
        else:
            self.target_name = target_name
            self.target_freq = target_freq
        if model_name is None:
            self.model_name = "PC"
            
        self.staz=staz
        self.standardize=standardize
        self.model_name=model_name
        self.start_date=start_date

        # pre-process data: staz and stand 
        self.data, self.mean, self.sigma = preprocess_data(data, transform_code_final, self.target_name, self.model_name, self.staz, self.standardize, self.start_date)
        self.X_all = self.data.drop(self.target_name, axis=1)
        self.y_all= self.data[self.target_name]

#         if n_targeted_predictors is None:
#             n_targeted_predictors=5
        self.n_targeted_predictors = n_targeted_predictors

        # input models
        if model_name is None:
            # PC, SPC, PC2, - see paper bai-ng LA, LASPC, LAPC
            self.model_name = "PC"
        # LARS
        if self.model_name == "LA5":
            self.lars_coef_number = 5
        elif self.model_name == "LAPC" or self.model_name == "LASPC":
            self.lars_coef_number = 30
        else:
            self.lars_coef_number=None
            
        # Targeted PCA - uses all no limits
        self.thresh_tstat=thresh_tstat
        if n_factors is None:
            n_factors=1
        self.n_factors=n_factors  

        if target_freq is None:
            self.target_freq="Q"
        else:
            self.target_freq=target_freq
        self.lags_y=lags_y
        self.lags_f=lags_f
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.target_freq=="Q":
            self.freq=3
        else:
            self.freq=1
        
    def fit(self, X, y):
        if "LA" in self.model_name:
            lars = Lars(n_nonzero_coefs=self.lars_coef_number, fit_intercept=False)
            lars_fit = lars.fit(X, y)
            var_keep = lars_fit.feature_names_in_[lars_fit.active_].tolist()
            self.X = X[var_keep]
            
            self.var_keep_lars = var_keep 
            # add lag y 
            self.lars_fit = lars_fit
            
            # fit regression
            y_lags = pd.concat([y.shift(i*self.freq) for i in range(1,self.lags_y)], axis=1)
            y_lags.columns = [f"{y.name}_lag{i}" for i in range(1,self.lags_y)]
            
            X_reg = pd.concat([y_lags, self.X], axis=1).dropna()
            y_reg = y.ffill().loc[X_reg.index]

            model = sm.OLS(y_reg, X_reg).fit()
            self.model = model
            # return model
            
        # if contains PCA we use the PCA otherwise LARS
        if "PC" in self.model_name:   
            if "TPC" in self.model_name:
                targeted_pred = TargetedPredictiors(X, 
                                                     y,
                                                     thresh_tstat=self.thresh_tstat,
                                                     lag_x=None, # already lagged
                                                     target_freq=self.target_freq,
                                                     n_targeted_predictors=self.n_targeted_predictors
                                                    )
                targeted_pred.fit()
                self.var_keep_tp = targeted_pred.targeted_predictors
                if len(self.var_keep_tp) < 10:
                    print(f"{y.name} decrease tstat value")
                    targeted_pred = TargetedPredictiors(X, 
                                                     y,
                                                     thresh_tstat=self.thresh_tstat/3,
                                                     lag_x=None, # already lagged
                                                     target_freq="M",
                                                     n_targeted_predictors=self.n_targeted_predictors
                                                    )
                    targeted_pred.fit()
                    self.var_keep_tp = targeted_pred.targeted_predictors
                    if len(self.var_keep_tp) < 10:
                        print(f"{y.name} Random variable selection")
                        vars_idx = np.random.randint(1,125,30).tolist()
                        self.var_keep_tp = X.columns[vars_idx]
                X = X[self.var_keep_tp]
                
            # fit PCA
            d, v = eig(X.cov())
            eigvect = v[:, :self.n_factors]
            self.eigvect = eigvect
            self.factors = X @ self.eigvect
            self.factors.columns = rename_col_factors(self.factors.columns)

            if self.model_name == "PC2" or self.model_name == "TPC2":
                factors_squared = self.factors**2
                factors_squared.columns = factors_squared.columns+"_squared"
                self.factors = pd.concat([self.factors, factors_squared], axis=1)
                
            # fit regression 
            factors_reg = self.factors.copy(deep=True)
            factors_reg.columns = factors_reg.columns+"_lag0"
            for i in  range(1,self.lags_f):
                f_lag = self.factors.shift(i)
                f_lag.columns = f_lag.columns+"_lag"+str(i)
                factors_reg[f_lag.columns]=f_lag
                
            y_lags = pd.concat([y.shift(i*self.freq) for i in range(1,self.lags_y)], axis=1)
            y_lags.columns = [f"{y.name}_lag{i}" for i in range(1,self.lags_y)]
            
            X_reg = pd.concat([y_lags, factors_reg], axis=1).dropna()
            y_reg = y.ffill().loc[X_reg.index]
            # select number of lags:
            res_=[]
            for i in  range(1,self.lags_f):
                for j in range(1,self.lags_y):
                    name_f = list(factors_reg.filter(like=f'lag{i}').columns)#f"f1_lag{lag}" for lag in range(i)]
                    name_y = [f"{y.name}_lag{lag}" for lag in range(1,j)]

                    cols_select = name_f+name_y
                    
                    X_ = X_reg[cols_select].dropna()
                    y_ = y_reg.loc[X_.index]
                    
                    model = sm.OLS(y_, X_)
                    results = model.fit()
                    res_.append([i, j, results.aic])
            res_ = pd.DataFrame(res_, columns=["lag_f", "lag_y", "aic"])
            res_ = res_.sort_values(by="aic", ascending=True)
            
            # refit - need t0 add 1
            best_lag_f = res_.iloc[0]["lag_f"].astype(int)+1
            best_lag_y = res_.iloc[0]["lag_y"].astype(int)+1
            
            name_f = [list(self.factors.columns+"_lag"+str(lag)) for lag in range(best_lag_f)]
            self.name_f = sum(name_f, [])
            self.name_y = [f"{y.name}_lag{lag}" for lag in range(1,best_lag_y)]
            self.cols_select = self.name_f+self.name_y

            X_ = X_reg[self.cols_select].dropna()
            y_ = y_reg.ffill().loc[X_.index]
            model = sm.OLS(y_, X_).fit()
            self.model = model
            # return model
        
        # return eigvect
    def predict(self, X, y):
        if self.model_name == "LA5":
            X_ = X[self.var_keep_lars]
            
            # fit regression
            y_lags = pd.concat([y.shift(i*self.freq) for i in range(1,self.lags_y)], axis=1)
            y_lags.columns = [f"{y.name}_lag{i}" for i in range(1,self.lags_y)]
            
            X_reg = pd.concat([y_lags, X_], axis=1).dropna()
            return self.model.predict(X_reg)
        else:
            # LAPC use this as well
            if "LAPC" in self.model_name:
                X = X[self.var_keep_lars]
            if "TPC" in self.model_name:
                X = X[self.var_keep_tp]
            # fit PCA
            d, v = eig(X.cov())
            eigvect = v[:, :self.n_factors]
            self.eigvect = eigvect
            self.factors_pred = X @ self.eigvect
            self.factors_pred.columns = rename_col_factors(self.factors_pred.columns)

            if self.model_name == "PC2" or self.model_name == "TPC2":
                factors_squared_pred = self.factors_pred**2
                factors_squared_pred.columns = factors_squared_pred.columns+"_squared"
                self.factors_pred = pd.concat([self.factors_pred, factors_squared_pred], axis=1)
                
            # fit regression
            factors_reg = self.factors_pred.copy(deep=True)
            factors_reg.columns = factors_reg.columns+"_lag0"
            for i in  range(1,self.lags_f):
                f_lag = self.factors_pred.shift(i)
                f_lag.columns = f_lag.columns+"_lag"+str(i)
                factors_reg[f_lag.columns]=f_lag
            
            y_lags = pd.concat([y.shift(i*self.freq) for i in range(1,self.lags_y)], axis=1)
            y_lags.columns = [f"{y.name}_lag{i}" for i in range(1,self.lags_y)]
            
            X_reg = pd.concat([y_lags, factors_reg], axis=1).dropna()
            X_reg = X_reg[self.cols_select].dropna()
            return self.model.predict(X_reg)
        