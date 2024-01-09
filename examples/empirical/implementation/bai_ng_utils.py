import pandas as pd
import numpy as np


def transform_variables(data, code):
    """
    Method to transform data to make variables stationary. We forward fill nan.
    """
    data_tr = data.copy(deep=True)
    for i in code.index:
        if code[i] == 1:
            pass
        elif code[i] == 2:
            data_tr.loc[:, i] = data_tr.loc[:, i].ffill().dropna().diff(1)
        elif code[i] == 3:
            data_tr.loc[:, i] = data_tr.loc[:, i].ffill().dropna().diff(1).diff(1)
        elif code[i] == 4:
            data_tr.loc[:, i] = np.log(data_tr.loc[:, i].ffill().dropna())
        elif code[i] == 5:
            data_tr.loc[:, i] = (np.log(data_tr.loc[:, i].ffill().dropna()).diff(1))
        elif code[i] == 6:
            data_tr.loc[:, i] = (np.log(data_tr.loc[:, i].ffill().dropna()).diff(1).diff(1))
        elif code[i] == 7:
            data_tr.loc[:, i] = data_tr.loc[:, i].pct_change()
        else:
            raise ValueError(f"code {i} - {code[i]} not suported")
    return data_tr


def untransform_variables(data_tr, code, fcst_h):
    """
    Method to untransform data from stationary to original. We forward fill nan.

    """
    data_untr = data_tr.copy(deep=True)
    for i in code.index:
        if code[i] == 1 or code[i] == 4:
            pass
        elif code[i] == 2 or code[i] == 5 or code[i] == 7:
            # (cumsum of diff and pct change)
            data_untr.loc[:, i] = data_untr.loc[:, i].fillna(value=0).rolling(fcst_h, min_periods=fcst_h).sum() / fcst_h
        elif code[i] == 3 or code[i] == 6:
            # (y_t+h - y_t)/h - (y_t-y_t-1) pag.309 Bai/Ng 2008: I(2) transform
            data_untr.loc[:, i] = data_untr.loc[:, i].fillna(value=0).rolling(fcst_h,
                                                                              min_periods=fcst_h).sum() / fcst_h - data_untr.loc[
                                                                                                                   :,
                                                                                                                   i].shift(1).fillna(
                value=0)
        else:
            raise ValueError(f"code {i} - {code[i]} not suported")
    return data_untr


def preprocess_data(data, transform_code_final, target_name, model_name, transform,
                    standardize, start_date):
    if transform:
        data = transform_variables(data, transform_code_final)

    if "SPC" in model_name:
        X_squared = data.drop(target_name, axis=1) ** 2
        X_squared.columns = X_squared.columns + "_squared"
        data = pd.concat([data, X_squared], axis=1)

    # use roling mean and std to avoid look ahead bias.
    # do not stardadize the Y
    y = data[target_name]
    X = data.drop(target_name, axis=1)
    mu = X.rolling(10000, min_periods=12).mean().ffill().dropna(how="all", axis=0)
    sigma = X.rolling(10000, min_periods=12).std().ffill().dropna(how="all", axis=0)
    if standardize:
        X_std = (X - mu) / sigma
        X_std = X_std.ffill().dropna(how="all", axis=0).loc[start_date:]
        data_std = pd.concat([y, X_std], axis=1)
        return data_std, mu, sigma
    else:
        return data, mu, sigma


def rename_col_factors(df_cols):
    return [f"f{i + 1}" for i in range(len(df_cols))]
