from typing import Tuple, Union, Optional, List

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from tensorflow import keras

from tools.monthly_quarterly_layer import MixedFreqMQLayer


def convert_decoder_to_numpy(
    decoder: keras.Model,
    has_bias: bool,
    factor_order: int,
    structure_decoder: tuple = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a keras Model decoder to a numpy object, accounting for Mixed frequency layer
    Args:
        decoder: decoder, a keras Model
        has_bias: whether there is a bias term
        factor_order: lag order for the common factors
        structure_decoder: the structure of the decoder, None for single layer

    Returns:
        bias and weight terms
    """
    if structure_decoder is None:
        return _convert_linear_decoder_to_numpy(
            decoder=decoder, has_bias=has_bias, factor_order=factor_order
        )
    else:
        raise NotImplementedError("Multilayer decoder not available yet!")


def _convert_linear_decoder_to_numpy(
    decoder: keras.Model,
    has_bias: bool,
    factor_order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a linear decoder to numpy arrays.
    Used to recover the factor loadings, say dimension observable is n, then output is dimension n x k,
        with k being equal to dimension idio and factors plus their lags.
    """
    last_layer = decoder.get_layer(index=-1)
    if isinstance(last_layer, MixedFreqMQLayer):
        last_layer = decoder.get_layer(index=-2)
        mix_freq_layer = decoder.get_layer(index=-1)
    else:
        mix_freq_layer = None
    if (
        last_layer.activation is not None
        and last_layer.activation != keras.activations.linear
    ):
        raise ValueError("Only linear layer supported.")
    if has_bias:
        ws, bs = last_layer.get_weights()
    else:
        ws = last_layer.get_weights()[0]
        bs = np.zeros(ws.shape[1])
    # state = [f_t, f_t-1, ..., eps_t, eps_t-1, ...]
    ws = ws.T
    if mix_freq_layer is not None:
        idio_loadings = np.eye(ws.shape[0])
        ws = np.hstack(
            (
                np.vstack(
                    (
                        np.kron(
                            np.array([1, 0, 0, 0, 0]),
                            ws[: mix_freq_layer.start_quarterly, :],
                        ),
                        np.kron(
                            np.array(mix_freq_layer.aggr_restr),
                            ws[mix_freq_layer.start_quarterly :, :],
                        ),
                    )
                ),
                # TODO: consider adding lags only for idio of quarterly variables
                np.vstack(
                    (
                        np.kron(
                            np.array([1, 0, 0, 0, 0]),
                            idio_loadings[: mix_freq_layer.start_quarterly, :],
                        ),
                        np.kron(
                            np.array(mix_freq_layer.aggr_restr),
                            idio_loadings[mix_freq_layer.start_quarterly :, :],
                        ),
                    )
                ),
            )
        )
    else:
        if factor_order == 1:
            ws = np.hstack((ws, np.eye(ws.shape[0])))  # weight term  # idio
        else:
            ws = np.hstack(
                (
                    np.kron(np.array([1] + [0] * (factor_order - 1)), ws),
                    # only one lag for the idio
                    np.eye(ws.shape[0]),
                )
            )
    return bs, ws


def get_transition_params(
    f_t: np.ndarray,
    eps_t: np.ndarray,
    factor_order: int,
    bool_no_miss: np.ndarray,
    extended_factor_lags: int = 0,
    quarterly_start: List[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate transition parameters.
    Args:
        f_t: common factors
        eps_t: idiosyncratic terms
        factor_order: lag order for the common factors
        bool_no_miss: array to keep track of non-missing values
        extended_factor_lags: how many factor lags to add in the state representation on top of factor_order (hence,
            total number of lags is factor_order + extended_factor_lags)
        quarterly_start: if flow quarterly idio AR1s are present, specify index start

    Returns:
        autoregressive matrix, diagonal residual covariance matrix, unconditional mean, unconditional variance,
            latent states (k x T)
    """
    if factor_order < 1:
        raise ValueError("factor_order must be >= 1")
    T, n_f = f_t.shape
    n_eps = eps_t.shape[1]
    p = factor_order
    X_f = np.hstack([f_t[p - j - 1 : T - j, :] for j in range(p)])
    A_f = np.linalg.lstsq(X_f[:-1], f_t[p:, :], rcond=None)[0].T
    A_eps, _, _ = get_idio(eps_t, bool_no_miss, quarterly_start=quarterly_start)
    # if extended factor lags is larger than zero, then we add lags also to the idiosyncratic
    # TODO: if we add extended lags only to idio quarterly, then this would need to change
    state_dim = (p + extended_factor_lags) * n_f + n_eps * (
        1 + (p + extended_factor_lags - 1) * (extended_factor_lags > 0)
    )
    A = np.zeros((state_dim, state_dim))
    # VAR in companion form plus idios
    A[:n_f, : p * n_f] = A_f
    start_idx_idio = (p + extended_factor_lags) * n_f
    A[
        start_idx_idio : start_idx_idio + n_eps, start_idx_idio : start_idx_idio + n_eps
    ] = A_eps
    for i in range(1, p + extended_factor_lags):
        A[i * n_f : (i + 1) * n_f, (i - 1) * n_f : i * n_f] = np.eye(n_f)
        if extended_factor_lags > 0:
            A[
                start_idx_idio + i * n_eps : start_idx_idio + (i + 1) * n_eps,
                start_idx_idio + (i - 1) * n_eps : start_idx_idio + i * n_eps,
            ] = np.eye(n_eps)
    # companion form x_t = [f_t, f_t_1, ..., eps_t, eps_t-1, ...]
    if extended_factor_lags > 0:
        X_f = np.hstack(
            [
                f_t[p + extended_factor_lags - j - 1 : T - j, :]
                for j in range(p + extended_factor_lags)
            ]
        )
        X_eps = np.hstack(
            [
                eps_t[p + extended_factor_lags - j - 1 : T - j, :]
                for j in range(p + extended_factor_lags)
            ]
        )
        x_t = np.vstack((X_f.T, X_eps.T))
    else:
        x_t = np.vstack((X_f.T, eps_t[p - 1 :].T))
    # error term matrix
    w_t = x_t[:, 1:] - A @ x_t[:, :-1]
    W = np.diag(np.diag(np.cov(w_t)))
    return A, W, x_t


def get_idio(
    eps: np.ndarray,
    idx_no_missings: np.ndarray,
    min_obs: int = 5,
    quarterly_start: Optional[int] = None,
    raise_error_if_few_obs: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute statistics from idiosyncratic terms
    Note: mixed estimation, quarterly in SS while others not.

    Args:
        eps: idiosyncratic zero mean AR(1)s
        idx_no_missings: array to keep track of non-missing values
        min_obs: minimum number of observations to estimate the statistics
        quarterly_start: if flow quarterly idio AR1s are present, specify index start
        raise_error_if_few_obs: whether to raise an error if the number of observations below min_obs or only warning

    Returns:
        autoregressive coefficients, unconditional standard deviation, conditional mean 1 step ahead
    """
    # init params statistics
    phi = np.zeros((eps.shape[1], eps.shape[1]))
    std_eps = np.nanstd(eps, ddof=1, axis=0)
    # loop over idios
    cond_mean = np.zeros_like(eps)
    end_nn_quarterly = quarterly_start if quarterly_start is not None else eps.shape[1]
    for j in range(end_nn_quarterly):
        to_select = np.r_[False, idx_no_missings[:-1, j] & idx_no_missings[1:, j]]
        if np.sum(to_select) >= min_obs:
            this_eps = eps[to_select, j]
            cov_eps = np.cov(this_eps[1:], this_eps[:-1])
            phi[j, j] = np.clip(
                cov_eps[0, 1] / ((cov_eps[0, 0] * cov_eps[1, 1]) ** 0.5), -0.99, 0.99
            )
            cond_mean[:, j] = phi[j, j] * eps[:, j]
        else:
            if raise_error_if_few_obs:
                raise ValueError(
                    f"Not enough observation ({min_obs}) to estimate idio AR(1) parameters."
                )
            else:
                # TODO: replace with warning
                print(
                    f"Not enough observation ({min_obs}) to estimate idio AR(1) parameters, setting to 0."
                )
                phi[j, j] = 0
                cond_mean[:, j] = 0

    for j in range(end_nn_quarterly, eps.shape[1]):
        mod_idio_qmm = QuarterlyAR1(eps[:, j])
        res_idio_qmm = mod_idio_qmm.fit(maxiter=50, return_params=True, disp=False)
        res_idio_qmm = mod_idio_qmm.fit_em(res_idio_qmm, maxiter=50, return_params=True)
        res = mod_idio_qmm._em_expectation_step(res_idio_qmm)
        cond_mean[:, [j]] = (
            res.smoothed_state.T @ res.transition[..., 0] @ res.design[0, ...]
        )
        phi[j, j] = np.clip(res_idio_qmm[0], -0.99, 0.99)
    return phi, std_eps, cond_mean


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
