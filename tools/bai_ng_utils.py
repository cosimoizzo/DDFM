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
        elif code[i] == 4:
            data_tr.loc[:, i] = np.log(data_tr.loc[:, i].ffill().dropna())
        elif code[i] == 5 or code[i] == 6:
            data_tr.loc[:, i] = (np.log(data_tr.loc[:, i].ffill().dropna()).diff(1)) * 100
        elif code[i] == 7:
            data_tr.loc[:, i] = ((data_tr.loc[:, i].ffill().dropna() / data_tr.loc[:, i].ffill().dropna().shift(
                1)) - 1) * 100
        else:
            raise ValueError(f"code {i} - {code[i]} not suported")
    return data_tr


def preprocess_data(data, transform_code_final, target_name, model_name, staz,
                    standardize, start_date):
    if staz:
        data = trasnform_variables(data, transform_code_final)

    if "SPC" in model_name:
        X_squared = data.drop(target_name, axis=1) ** 2
        X_squared.columns = X_squared.columns + "_squared"
        data = pd.concat([data, X_squared], axis=1)

    mu = data.rolling(10000, min_periods=12).mean()
    sigma = data.rolling(10000, min_periods=12).std()
    if standardize:
        data_std = (data - mu) / sigma
        data_std = data_std.ffill().bfill().loc[start_date:]
        return data_std, mu, sigma
    #     self.X = self.data_staz.drop(self.target_name, axis=1).shift(self.lag_x)
    #     self.y = self.data_staz[self.target_name]
    else:
        return data, mu, sigma


def rename_col_factors(df_cols):
    return [f"f{i + 1}" for i in range(len(df_cols))]
