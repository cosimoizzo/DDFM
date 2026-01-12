from typing import Tuple, Union

import numpy as np
import pandas as pd
from tensorflow import keras


def convert_decoder_to_numpy(
    decoder: keras.Model,
    has_bias: bool,
    factor_order: int,
    structure_decoder: tuple = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a keras Model decoder to a numpy object
    Args:
        decoder: decoder, a keras Model
        has_bias: whether there is a bias term
        factor_order: lag order for the common factors
        structure_decoder: the structure of the decoder, None for single layer

    Returns:
        bias and weight terms
    """
    if structure_decoder is None:
        if has_bias:
            ws, bs = decoder.get_layer(index=-1).get_weights()
        else:
            ws = decoder.get_layer(index=-1).get_weights()[0]
            bs = np.zeros(ws.shape[1])
        if factor_order == 2:
            ws = np.hstack(
                (
                    ws.T,  # weight term
                    np.zeros((ws.shape[1], ws.shape[0])),  # make zero lagged values
                    np.identity(ws.shape[1]),  # idio
                )
            )
        elif factor_order == 1:
            ws = np.hstack((ws.T, np.identity(ws.shape[1])))  # weight term  # idio
        else:
            raise NotImplementedError(
                "Only VAR(2) or VAR(1) for common factors at the moment."
            )
    else:
        raise NotImplementedError("Nonlinear decoder not available yet!")

    return bs, ws


def get_transition_params(
    f_t: np.ndarray, eps_t: np.ndarray, factor_order: int, bool_no_miss: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate transition parameters.
    Args:
        f_t: common factors
        eps_t: idiosyncratic terms
        factor_order: lag order for the common factors
        bool_no_miss: array to keep track of non-missing values

    Returns:
        autoregressive matrix, diagonal residual covariance matrix, unconditional mean, unconditional variance,
            latent states
    """
    if factor_order == 2:
        f_past = np.hstack((f_t[1:-1, :], f_t[:-2, :]))
        A_f = np.linalg.lstsq(f_past, f_t[2:, :], rcond=None)[0].T
    elif factor_order == 1:
        f_past = f_t[:-1, :]
        A_f = np.linalg.lstsq(f_past, f_t[1:, :], rcond=None)[0].T
    else:
        raise NotImplementedError(
            "Only VAR(2) or VAR(1) for common factors at the moment."
        )
    # get AR coeffs. from idiosyncratic
    A_eps, _, _ = get_idio(eps_t, bool_no_miss)
    # companion form x_t = [f_t, f_t_1, eps_t]
    if factor_order == 2:
        x_t = np.vstack((f_t[1:, :].T, f_t[:-1, :].T, eps_t[1:, :].T))
        A = np.vstack(
            (
                np.hstack(
                    (A_f, np.zeros((A_f.shape[0], eps_t.shape[1])))
                ),  # VAR factors
                np.hstack(
                    (
                        np.identity(A_f.shape[0]),
                        np.zeros((A_f.shape[0], A_f.shape[0] + eps_t.shape[1])),
                    )
                ),
                np.hstack(
                    (np.zeros((eps_t.shape[1], A_f.shape[1])), A_eps)
                ),  # AR 1 idio
            )
        )
    elif factor_order == 1:
        x_t = np.vstack((f_t.T, eps_t.T))
        A = np.vstack(
            (
                np.hstack(
                    (A_f, np.zeros((A_f.shape[0], eps_t.shape[1])))
                ),  # VAR factors
                np.hstack(
                    (np.zeros((eps_t.shape[1], A_f.shape[1])), A_eps)
                ),  # AR 1 idio
            )
        )
    else:
        raise NotImplementedError(
            "Only VAR(2) or VAR(1) for common factors at the moment."
        )
    # error term matrix
    w_t = x_t[:, 1:] - A @ x_t[:, :-1]
    W = np.diag(np.diag(np.cov(w_t)))
    # Set to unconditional moments of x_0 = [f_0, f_0_1, eps_0]
    mu_0 = np.mean(x_t, axis=1)
    Σ_0 = np.cov(x_t)
    # zero correlation with idiosyncratic and diagonal covariance among them
    Σ_0[: A_f.shape[1], A_f.shape[1] :] = 0
    Σ_0[A_f.shape[1] :, : A_f.shape[1]] = 0
    Σ_0[A_f.shape[1] :, A_f.shape[1] :] = np.diag(
        np.diag(Σ_0[A_f.shape[1] :, A_f.shape[1] :])
    )
    return A, W, mu_0, Σ_0, x_t


def get_idio(
    eps: np.ndarray,
    idx_no_missings: np.ndarray,
    min_obs: int = 5,
    force_zero_mean: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute statistics from idiosyncratic terms: AR(1), mean, stds
    Args:
        eps: idiosyncratic AR(1)s
        idx_no_missings: array to keep track of non-missing values
        min_obs: minimum number of observations to estimate the statistics
        force_zero_mean: whether to force zero unconditional mean

    Returns:
        autoregressive coefficients, unconditional mean and standard deviation
    """
    # init params statistics
    phi = np.zeros((eps.shape[1], eps.shape[1]))
    mu_eps = np.zeros(eps.shape[1])
    std_eps = np.zeros(eps.shape[1])
    # loop over idios
    for j in range(eps.shape[1]):
        to_select = idx_no_missings[:, j]  # ~np.isnan(self.z_actual[:, j])
        to_select = np.hstack((np.array([False]), to_select[:-1] * to_select[1:]))
        if np.sum(to_select) >= min_obs:
            this_eps = eps[to_select, j]
        else:
            raise ValueError(
                f"Not enough observation ({min_obs}) to estimate idio AR(1) parameters."
            )
        if not force_zero_mean:
            mu_eps[j] = np.mean(this_eps)
        std_eps[j] = np.std(this_eps, ddof=1)
        cov1_eps = np.cov(this_eps[1:], this_eps[:-1])[0][1]
        phi[j, j] = np.clip(cov1_eps / (std_eps[j] ** 2), -0.99, 0.99)
    return phi, mu_eps, std_eps


def get_data_with_lags(
    interpolate: bool, data_raw: Union[pd.DataFrame, np.ndarray], lags_input: int
) -> pd.DataFrame:
    """
    Modify input data with interpolation and lagged values
    Args:
        interpolate: whether to interpolate or not the missing values
        data_raw: data from which to build inputs to the model
        lags_input: number of lags to add to the input data

    Returns:
        New dataframe with lagged values and interpolated missing values
    """
    df = pd.DataFrame(data_raw).copy()

    if interpolate and df.isna().any().any():
        df = df.interpolate(method="linear", limit_direction="both")

    if lags_input > 0:
        lagged = {
            f"{col}_lag{lag}": df[col].shift(lag)
            for col in df.columns
            for lag in range(1, lags_input + 1)
        }
        # possible starting missing values in lagged to correct for
        df_lagged = pd.DataFrame(lagged)
        df = pd.concat([df, df_lagged], axis=1)
        df = (
            df.iloc[lags_input:].reset_index(drop=True).fillna(value=0)
        )  # fill initial missing values with 0 mean

    return df
