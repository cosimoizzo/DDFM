import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tools.getters_converters_tools import (
    convert_decoder_to_numpy,
    get_transition_params,
    get_idio,
    get_data_with_lags,
)
from tools.monthly_quarterly_layer import MixedFreqMQLayer


class TestGettersAndConvertersTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls, seed: int = 123):
        cls.rng = np.random.default_rng(seed)

    def test_convert_decoder_to_numpy(self):
        def build_two_layer_model(
            input_dim=3,
            hidden_dim=4,
            output_dim=10,
            use_bias=True,
            use_mixed_freq=False,
        ):
            all_layers = [
                layers.Input(shape=(input_dim,)),
                layers.Dense(hidden_dim, activation="relu", use_bias=True),
                layers.Dense(output_dim, use_bias=use_bias),
            ]
            if use_mixed_freq:
                all_layers.append(MixedFreqMQLayer(output_dim, start_quarterly=8))
            model = keras.Sequential(all_layers)
            return model

        # test factor order 1 and 2 with and without bias, with and without mixed freq
        for use_mixed_freq in [True, False]:
            for factor_order in [1, 2]:
                for use_bias in [True, False]:
                    model = build_two_layer_model(
                        use_bias=use_bias, use_mixed_freq=use_mixed_freq
                    )
                    last_layer = (
                        model.layers[-2] if use_mixed_freq else model.layers[-1]
                    )
                    W = np.array([[j * i for i in range(1, 11)] for j in range(1, 5)])
                    b = np.arange(0, 10, 1) if use_bias else np.array([0.0] * 10)
                    (
                        last_layer.set_weights([W, b])
                        if use_bias
                        else last_layer.set_weights([W])
                    )
                    bs, ws = convert_decoder_to_numpy(
                        decoder=model,
                        has_bias=use_bias,
                        factor_order=factor_order,
                    )
                    # factors, lagged factors, idiosyncratic components
                    if use_mixed_freq:
                        # here also lagged idiosyncratic components
                        loading_idio = np.eye(10)
                        expected_ws = np.hstack(
                            (
                                np.vstack(
                                    (
                                        np.kron(np.array([1, 0, 0, 0, 0]), W.T[:8, :]),
                                        np.kron(np.array([1, 2, 3, 2, 1]), W.T[8:, :]),
                                    )
                                ),
                                np.vstack(
                                    (
                                        np.kron(
                                            np.array([1, 0, 0, 0, 0]),
                                            loading_idio[:8, :],
                                        ),
                                        np.kron(
                                            np.array([1, 2, 3, 2, 1]),
                                            loading_idio[8:, :],
                                        ),
                                    )
                                ),
                            )
                        )
                    else:
                        expected_ws = (
                            np.hstack((W.T, np.identity(W.shape[1])))
                            if factor_order == 1
                            else np.hstack(
                                (
                                    W.T,
                                    np.zeros((W.shape[1], W.shape[0])),
                                    np.identity(W.shape[1]),
                                )
                            )
                        )
                    np.testing.assert_array_equal(bs, b)
                    np.testing.assert_array_equal(ws, expected_ws)

    def test_get_transition_params(self):
        np.random.seed(0)
        T = 50
        n_f = 2
        n_eps = 3
        f_t = np.random.randn(T, n_f)
        eps_t = np.random.randn(T, n_eps)
        bool_no_miss = np.ones_like(eps_t, dtype=bool)
        for extended_lags in [0, 3]:
            for factor_order in [1, 2]:
                A, W, x_t = get_transition_params(
                    f_t,
                    eps_t,
                    factor_order=factor_order,
                    bool_no_miss=bool_no_miss,
                    extended_factor_lags=extended_lags,
                )
                n_f = f_t.shape[1]
                n_eps = eps_t.shape[1]
                state_dim = (factor_order + extended_lags) * n_f + n_eps * (
                    1 + (factor_order + extended_lags - 1) * (extended_lags > 0)
                )
                start_idio = (factor_order + extended_lags) * n_f
                # --- Dimension ---
                self.assertTrue(A.shape == (state_dim, state_dim))
                self.assertTrue(W.shape == (state_dim, state_dim))
                self.assertTrue(
                    x_t.T.shape
                    == (f_t.shape[0] - (factor_order + extended_lags - 1), state_dim)
                )
                # W diagonal
                self.assertTrue(np.allclose(W, np.diag(np.diag(W))))
                # zero entries for the lags in W
                self.assertTrue(
                    np.allclose(
                        W[n_f:start_idio, n_f:start_idio],
                        np.diag(np.zeros((factor_order + extended_lags - 1) * n_f)),
                    )
                )
                if extended_lags > 0:
                    self.assertTrue(
                        np.allclose(
                            W[
                                -n_eps * (factor_order + extended_lags - 1) :,
                                -n_eps * (factor_order + extended_lags - 1) :,
                            ],
                            np.diag(
                                np.zeros((factor_order + extended_lags - 1) * n_eps)
                            ),
                        )
                    )
                # block structure in A
                A_expected = np.zeros((state_dim, state_dim))
                ## VAR loadings
                A_expected[:n_f, : n_f * factor_order] = A[:n_f, : n_f * factor_order]
                ## idio diagonal AR1s
                A_expected[
                    start_idio : start_idio + n_eps, start_idio : start_idio + n_eps
                ] = np.diag(
                    np.diag(
                        A[
                            start_idio : start_idio + n_eps,
                            start_idio : start_idio + n_eps,
                        ]
                    )
                )
                ## identity to off-set lagged values in common factors and idio
                A_expected[n_f:start_idio, : start_idio - n_f] = np.eye(
                    start_idio - n_f
                )
                if extended_lags > 0:
                    A_expected[start_idio + n_eps :, start_idio:-n_eps] = np.eye(
                        n_eps * (factor_order + extended_lags - 1)
                    )
                ##
                self.assertTrue(np.allclose(A, A_expected))
                # x_t is [f_t, f_t-1, ... , e_t]
                self.assertTrue(
                    np.allclose(
                        x_t.T[:, :n_f], f_t[factor_order + extended_lags - 1 :, :]
                    )
                )
                self.assertTrue(
                    np.allclose(
                        x_t.T[:, start_idio : start_idio + n_eps],
                        eps_t[factor_order + extended_lags - 1 :, :],
                    )
                )

    def test_get_idio(self):
        """
        Simulate AR1s and check outputs of get_idio
        """
        # generate mixed frequencies AR(1)s
        T = 205
        n_monthly = 10
        n_quarterly = 2
        n_series = n_monthly + n_quarterly
        quarterly_start = n_monthly
        eps = np.zeros((T, n_series))
        for j in range(n_series):
            for t in range(1, T):
                eps[t, j] = eps[t - 1, j] * 0.5 + 0.1 * self.rng.standard_normal()
        for j in range(quarterly_start, n_series):
            tmp = np.ones(T) * np.nan
            for t in range(1, T):
                if t >= 4 and t % 3 == 0:
                    tmp[t] = (
                        eps[t, j]
                        + 2 * eps[t - 1, j]
                        + 3 * eps[t - 2, j]
                        + 2 * eps[t - 3, j]
                        + eps[t - 4, j]
                    )
            eps[:, j] = tmp
        eps = eps[5:, :]
        # estimate them
        phi, std_eps, cond_mean = get_idio(
            eps,
            ~np.isnan(eps),
            min_obs=20,
            quarterly_start=quarterly_start,
        )
        # check shapes
        self.assertTrue(phi.shape == (n_series, n_series))
        self.assertTrue(std_eps.shape == (n_series,))
        self.assertTrue(cond_mean.shape == eps.shape)
        off_diag = phi - np.diag(np.diag(phi))
        self.assertTrue(np.allclose(off_diag, 0.0))
        # check stationarity and approximate recovery
        for j in range(n_series):
            self.assertTrue(np.isfinite(phi[j, j]))
            self.assertTrue(abs(phi[j, j]) < 0.999)
            self.assertTrue(abs(phi[j, j] - 0.5) < 0.15)
            self.assertAlmostEqual(std_eps[j], np.nanstd(eps[:, j], ddof=1), places=7)
            frac_error = np.nansum((eps[1:, j] - cond_mean[:-1, j]) ** 2) / np.nansum(
                eps[1:, j] ** 2
            )
            self.assertTrue(0 < frac_error < 1)

    def test_get_data_with_lags(self):
        # test no interpolation and no lags
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = get_data_with_lags(
            interpolate=False,
            data_raw=data,
            lags_input=0,
        )

        pd.testing.assert_frame_equal(result, pd.DataFrame(data, dtype=np.float32))
        # test with lags
        data = np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=np.float32).T
        result = get_data_with_lags(
            interpolate=False,
            data_raw=data,
            lags_input=2,
        )
        expected = pd.DataFrame(
            {
                0: [3, 4],
                1: [6, 7],
                "0_lag1": [2, 3],
                "0_lag2": [1, 2],
                "1_lag1": [5, 6],
                "1_lag2": [4, 5],
            },
            dtype=np.float32,
        )
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
