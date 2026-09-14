import unittest

import keras
import numpy as np
import pandas as pd

from models.base import FactorModel
from models.ddfm import DDFM, _USE_M_UKF
from models.state_space.state_space_wrapper import StateSpace
from synthetic_dgp.simulate import SIMULATE, QuarterlyVars, AggregationInstr


class TestDDFMWrapper(unittest.TestCase):
    """
    Exercises the public DDFM object itself (the other classes test _best_model directly):
    seed selection, the FactorModel contract on the wrapper, and attribute forwarding to
    the selected run. Deliberately small: it checks wiring, not estimation quality.
    """

    @classmethod
    def setUpClass(cls):
        cls.sim = SIMULATE(seed=1, n=8, r=2, poly_degree=1)
        cls.x = pd.DataFrame(cls.sim.simulate(60, portion_missings=0.05))
        cls.seeds = (3, 4)
        cls.ddfm = DDFM(
            structure_encoder=(2,),
            factor_order=1,
            use_bias=False,
            max_iter=2,
            epochs=5,
            seed=cls.seeds,
        )
        cls.ddfm.fit(cls.x, build_state_space=True)

    def test_before_fit(self):
        fresh = DDFM(structure_encoder=(2,), seed=3)
        self.assertIsInstance(fresh, FactorModel)
        self.assertFalse(fresh._fitted)
        with self.assertRaises(RuntimeError):
            fresh.predict(self.x, steps_ahead=1)
        with self.assertRaises(AttributeError):
            fresh.state_space  # forwarded attribute, only available after fit

    def test_seed_selection(self):
        self.assertIn(self.ddfm._best_seed, self.seeds)
        self.assertEqual(self.ddfm._best_model.seed, self.ddfm._best_seed)
        self.assertTrue(np.isfinite(self.ddfm._best_model.loss_now))

    def test_factor_model_contract_on_wrapper(self):
        self.assertTrue(self.ddfm._fitted)
        np.testing.assert_array_equal(self.ddfm.mean_data, self.ddfm._best_model.mean_data)
        np.testing.assert_array_equal(self.ddfm.sigma_data, self.ddfm._best_model.sigma_data)
        self.assertEqual(list(self.ddfm.variable_order), list(self.x.columns))
        np.testing.assert_allclose(self.ddfm._unstd(self.ddfm._std(self.x)), self.x.values)

    def test_interface_methods(self):
        n = self.x.shape[1]
        self.assertEqual(self.ddfm.predict(self.x, steps_ahead=2).shape, (3, n))
        self.assertEqual(self.ddfm.predict_one_step_ahead(self.x).shape, self.x.shape)
        self.assertEqual(self.ddfm.get_factors(self.x).shape, (len(self.x), 2))
        filled = self.ddfm.fill_na(self.x)
        self.assertEqual(filled.shape, self.x.shape)
        self.assertFalse(filled.isnull().any().any())

    def test_forwarded_methods_and_attributes(self):
        n = self.x.shape[1]
        mean, cov = self.ddfm.predict_with_covariance(self.x, steps_ahead=2)
        self.assertEqual(mean.shape, (3, n))
        self.assertEqual(cov.shape, (3 * n, n))
        xf, pf = self.ddfm.state_space.filter(self.x.values)
        self.assertEqual(self.ddfm.predict_from_states(xf[-1], pf[-1], 1)[0].shape, (2, n))
        self.assertIsInstance(self.ddfm.build_state_space(), StateSpace)
        self.assertEqual(self.ddfm.factors_filtered.shape, (len(self.x), 2))
        self.assertEqual(self.ddfm.factors_smoothed.shape, (len(self.x), 2))
        self.assertEqual(self.ddfm.last_neurons.shape[1], len(self.x))
        # forwarding never exposes private names of the inner run
        with self.assertRaises(AttributeError):
            self.ddfm._data_tmp

    def test_forwarding_is_the_same_object(self):
        self.assertIs(self.ddfm.state_space, self.ddfm._best_model.state_space)
        pd.testing.assert_frame_equal(
            self.ddfm.predict(self.x, steps_ahead=1),
            self.ddfm._best_model.predict(self.x, steps_ahead=1),
        )


"""
The following tests are run on the _DDFM instance obtained via DDFM, as the methods of DDFM fall back to _DDFM.
"""


class TestDDFM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed = 123
        cls.lags_input = 0
        cls.append_to_msg = ""
        cls.sim = SIMULATE(seed=seed, n=40, r=3, poly_degree=1)
        cls.x = cls.sim.simulate(150, portion_missings=0.0)
        cls.structure_encoder = (3,)

    def test_fit_predict(self):
        """
        Testing true states are recovered with R2 of at least 80%, autoencoder is consistent with decode+encode, a
        state space representation can be built, predict returns correct shapes.
        """
        for jointly_est_var in [True, False]:
            ddfm = self._get_model(self.structure_encoder, jointly_est_var)
            self._single_test_fit(ddfm)
            self._single_test_predict(ddfm)

    def test_replicability(self):
        """
        Testing similar states can be recovered over 2 runs of the same model (R^2 at least 95%).
        """
        ddfm1 = self._get_model(structure_encoder=self.structure_encoder)
        ddfm2 = self._get_model(structure_encoder=self.structure_encoder)
        predict1 = ddfm1.encoder(ddfm1._data_tmp)
        predict2 = ddfm2.encoder(ddfm2._data_tmp)
        r2 = self.sim.evaluate(predict2.numpy(), f_true=predict1.numpy())
        self.assertGreaterEqual(
            r2, 0.95, msg=f"Cannot reproduce states{self.append_to_msg}."
        )

    def _get_model(self, structure_encoder, jointly_est_var=False, seed=3):
        ddfm = DDFM(
            structure_encoder=structure_encoder,
            factor_order=1,
            lags_input=self.lags_input,
            use_bias=False,
            link="relu",
            max_iter=1000,
            var_loss_weight=1 if jointly_est_var else 0,
            seed=seed,
        )
        ddfm.fit(pd.DataFrame(self.x), build_state_space=True)
        return ddfm._best_model

    def _single_test_fit(self, ddfm):
        last_neurons = np.mean(ddfm.last_neurons, axis=0)
        r2 = self.sim.evaluate(last_neurons, f_true=self.sim.f[self.lags_input :])
        self.assertGreaterEqual(
            r2,
            0.8,
            msg=f"r2 should be true and fitted states should be greater than 0.8{self.append_to_msg}",
        )
        predict_from_auto = ddfm.autoencoder(ddfm._data_tmp)
        predict_from_encode_decode = ddfm.decoder(ddfm.encoder(ddfm._data_tmp))
        np.testing.assert_array_almost_equal(
            predict_from_auto,
            predict_from_encode_decode,
            err_msg=f"Autoencoder output different from decode+encode{self.append_to_msg}.",
        )
        self._check_state_space(ddfm)
        np.testing.assert_array_almost_equal(
            ddfm.mean_data,
            ddfm.state_space.mean_y,
            err_msg=f"Mean in ddfm and state space object do not match{self.append_to_msg}.",
        )
        np.testing.assert_array_almost_equal(
            ddfm.sigma_data,
            ddfm.state_space.sigma_y,
            err_msg=f"Vols in ddfm and state space object do not match{self.append_to_msg}.",
        )

    def _check_state_space(self, ddfm):
        msg_if_fail = f"Failed to build state_space properly {self.append_to_msg}"
        self.assertIsInstance(ddfm.state_space, StateSpace, msg=msg_if_fail)
        # check shapes
        self.assertEqual(
            ddfm.state_space.observation_map.shape,
            (self.sim.n, self.sim.n + self.sim.r),
            msg=msg_if_fail,
        )
        np.testing.assert_array_almost_equal(
            ddfm.state_space.observation_map[:, self.sim.r :],
            np.eye(self.sim.n),
            err_msg=msg_if_fail,
        )
        self.assertEqual(
            ddfm.state_space.observation_covariance.shape,
            (self.sim.n, self.sim.n),
            msg=msg_if_fail,
        )
        np.testing.assert_array_almost_equal(
            ddfm.state_space.observation_covariance,
            np.diag(np.diag(ddfm.state_space.observation_covariance)),
            err_msg=msg_if_fail,
        )
        self.assertEqual(
            ddfm.state_space.transition_map.shape,
            (self.sim.n + self.sim.r, self.sim.n + self.sim.r),
            msg=msg_if_fail,
        )
        np.testing.assert_array_almost_equal(
            ddfm.state_space.transition_map[self.sim.r :, self.sim.r :],
            np.diag(
                np.diag(ddfm.state_space.transition_map[self.sim.r :, self.sim.r :])
            ),
            err_msg=msg_if_fail,
        )
        self.assertEqual(
            ddfm.state_space.transition_covariance.shape,
            (self.sim.n + self.sim.r, self.sim.n + self.sim.r),
            msg=msg_if_fail,
        )
        np.testing.assert_array_almost_equal(
            ddfm.state_space.transition_covariance,
            np.diag(np.diag(ddfm.state_space.transition_covariance)),
            err_msg=msg_if_fail,
        )
        # r2 between ssm and encoded factors
        factors_mean = np.mean(ddfm.last_neurons, axis=0)
        # TODO: replace with a get factors method?
        ssm_factors = ddfm.state_space.smooth(self.x)[0][:, : factors_mean.shape[0]]
        r2 = self.sim.evaluate(ssm_factors, f_true=factors_mean)
        self.assertGreaterEqual(
            r2,
            0.8,
            msg=f"r2 between ssm and encoded should be greater than 0.8{self.append_to_msg}",
        )

    def _single_test_predict(self, ddfm):
        # FactorModel interface: predict returns a single DataFrame
        self.assertIsInstance(ddfm, FactorModel)
        pred_df = ddfm.predict(pd.DataFrame(self.x), steps_ahead=2)
        self.assertIsInstance(pred_df, pd.DataFrame)
        self.assertEqual(pred_df.shape[0], 3)
        self.assertEqual(pred_df.shape[1], self.x.shape[1])

        # predict_with_covariance returns (mean, cov) tuple
        mean, covs = ddfm.predict_with_covariance(pd.DataFrame(self.x), steps_ahead=2)
        self.assertEqual(mean.shape[0], 3)
        self.assertEqual(mean.shape[1], self.x.shape[1])
        self.assertEqual(covs.shape[0], 3 * self.x.shape[1])
        self.assertEqual(covs.shape[1], self.x.shape[1])

        # get_factors returns (T, r) DataFrame
        factors = ddfm.get_factors(pd.DataFrame(self.x))
        self.assertIsInstance(factors, pd.DataFrame)
        self.assertEqual(factors.shape[1], self.structure_encoder[-1])
        self.assertEqual(factors.shape[0], self.x.shape[0])

        # fill_na returns (T, N) DataFrame in original scale
        if ddfm.lags_input == 0:
            x_miss = self.x.copy()
            x_miss[0, 0] = np.nan
            filled = ddfm.fill_na(pd.DataFrame(x_miss))
            self.assertIsInstance(filled, pd.DataFrame)
            self.assertFalse(filled.isnull().any().any())


class TestDDFMMonthlyQuarterly(TestDDFM):
    @classmethod
    def setUpClass(cls):
        seed = 1234546
        cls.lags_input = 0
        cls.append_to_msg = " (mixed frequency)"
        cls.idx_quarterly = [i for i in range(35, 40)]
        cls.sim = SIMULATE(seed=seed, n=40, r=3, poly_degree=1)
        cls.x = cls.sim.simulate(
            250,
            portion_missings=0.05,
            quarterly_vars=QuarterlyVars(
                cls.idx_quarterly, aggregation=AggregationInstr.MM
            ),
        )
        cls.structure_encoder = (3,)

    def _check_state_space(self, ddfm):
        msg_if_fail = f"Failed to build state_space properly {self.append_to_msg}"
        self.assertIsInstance(ddfm.state_space, StateSpace, msg=msg_if_fail)
        # check shapes
        self.assertEqual(
            ddfm.state_space.observation_map.shape,
            (self.sim.n, (self.sim.n + self.sim.r) * 5),
            msg=msg_if_fail,
        )
        n_monthly = self.sim.n - len(self.idx_quarterly)
        expected_monthly = np.zeros(
            (n_monthly, (self.sim.n + self.sim.r) * 5 - self.sim.r)
        )
        expected_monthly[:, 4 * self.sim.r : 4 * self.sim.r + n_monthly] = np.eye(
            n_monthly
        )
        np.testing.assert_array_almost_equal(
            ddfm.state_space.observation_map[: -len(self.idx_quarterly), self.sim.r :],
            expected_monthly,
            err_msg=msg_if_fail,
        )
        aggr_weights = np.array([1, 2, 3, 2, 1])
        for j in self.idx_quarterly:
            quarterly_loadings = ddfm.state_space.observation_map[j, :]
            expected_quarterly_loadings = np.zeros_like(quarterly_loadings)
            # common
            for i_f in range(self.sim.r):
                expected_quarterly_loadings[
                    [i * self.sim.r + i_f for i in range(5)]
                ] = (aggr_weights * quarterly_loadings[i_f])
            # idio
            expected_quarterly_loadings[
                [self.sim.r * 5 + self.sim.n * i + j for i in range(5)]
            ] = aggr_weights
            np.testing.assert_array_almost_equal(
                quarterly_loadings, expected_quarterly_loadings, err_msg=msg_if_fail
            )
        self.assertEqual(
            ddfm.state_space.observation_covariance.shape,
            (self.sim.n, self.sim.n),
            msg=msg_if_fail,
        )
        np.testing.assert_array_almost_equal(
            ddfm.state_space.observation_covariance,
            np.diag(np.diag(ddfm.state_space.observation_covariance)),
            err_msg=msg_if_fail,
        )
        self.assertEqual(
            ddfm.state_space.transition_map.shape,
            ((self.sim.n + self.sim.r) * 5, (self.sim.n + self.sim.r) * 5),
            msg=msg_if_fail,
        )
        np.testing.assert_array_almost_equal(
            ddfm.state_space.transition_map[
                self.sim.r : self.sim.r * 5, : self.sim.r * 5
            ],
            np.eye(self.sim.r * 4, self.sim.r * 5),
            err_msg=msg_if_fail,
        )
        np.testing.assert_array_almost_equal(
            ddfm.state_space.transition_map[
                self.sim.r * 5 + self.sim.n :, self.sim.r * 5 :
            ],
            np.eye(self.sim.n * 4, self.sim.n * 5),
            err_msg=msg_if_fail,
        )
        self.assertEqual(
            ddfm.state_space.transition_covariance.shape,
            ((self.sim.n + self.sim.r) * 5, (self.sim.n + self.sim.r) * 5),
            msg=msg_if_fail,
        )
        np.testing.assert_array_almost_equal(
            ddfm.state_space.transition_covariance,
            np.diag(np.diag(ddfm.state_space.transition_covariance)),
            err_msg=msg_if_fail,
        )
        transition_covariance_zeroed = ddfm.state_space.transition_covariance.copy()
        transition_covariance_zeroed[: self.sim.r, : self.sim.r] = 0
        transition_covariance_zeroed[
            self.sim.r * 5 : self.sim.r * 5 + self.sim.n,
            self.sim.r * 5 : self.sim.r * 5 + self.sim.n,
        ] = 0
        np.testing.assert_array_almost_equal(
            transition_covariance_zeroed,
            np.zeros_like(transition_covariance_zeroed),
            err_msg=msg_if_fail,
        )

    def _get_model(self, structure_encoder, jointly_est_var=False, seed=123):
        ddfm = DDFM(
            structure_encoder=structure_encoder,
            factor_order=1,
            lags_input=self.lags_input,
            use_bias=False,
            link="relu",
            max_iter=1000,
            var_loss_weight=1 if jointly_est_var else 0,
            seed=seed,
            clipnorm=5.0,
        )
        df_x = pd.DataFrame(self.x)
        ddfm.fit(df_x, build_state_space=True, vars_mq_restrictions=self.idx_quarterly)
        return ddfm._best_model


class TestDDFMMonthlyQuarterlyNonLinDec(TestDDFM):
    @classmethod
    def setUpClass(cls):
        seed = 1234546
        cls.lags_input = 0
        cls.append_to_msg = " (mixed frequency with nonlinear decoder)"
        cls.idx_quarterly = [i for i in range(35, 40)]
        cls.sim = SIMULATE(seed=seed, n=40, r=3, poly_degree=2, sign_features=3)
        cls.x = cls.sim.simulate(
            250,
            portion_missings=0.05,
            quarterly_vars=QuarterlyVars(
                cls.idx_quarterly, aggregation=AggregationInstr.MM
            ),
        )
        cls.structure_encoder = (cls.sim.f.shape[1], 3 * 2, 3)
        cls.structure_decoder = (3 * 2, cls.sim.f.shape[1])

    def _check_state_space(self, ddfm):
        msg_if_fail = f"Failed to build state_space properly {self.append_to_msg}"
        self.assertIsInstance(ddfm.state_space, StateSpace, msg=msg_if_fail)
        # check shapes
        self.assertIsInstance(
            ddfm.state_space.observation_map,
            keras.Model,
            msg=msg_if_fail,
        )
        self.assertEqual(
            ddfm.state_space.observation_covariance.shape,
            (self.sim.n, self.sim.n),
            msg=msg_if_fail,
        )
        np.testing.assert_array_almost_equal(
            ddfm.state_space.observation_covariance,
            np.diag(np.diag(ddfm.state_space.observation_covariance)),
            err_msg=msg_if_fail,
        )
        if _USE_M_UKF:
            self.assertEqual(
                ddfm.state_space.transition_map.shape,
                ((self.sim.n + self.sim.r) * 5, (self.sim.n + self.sim.r) * 5),
                msg=msg_if_fail,
            )
            np.testing.assert_array_almost_equal(
                ddfm.state_space.transition_map[
                    self.sim.r : self.sim.r * 5, : self.sim.r * 5
                ],
                np.eye(self.sim.r * 4, self.sim.r * 5),
                err_msg=msg_if_fail,
            )
            np.testing.assert_array_almost_equal(
                ddfm.state_space.transition_map[
                    self.sim.r * 5 + self.sim.n :, self.sim.r * 5 :
                ],
                np.eye(self.sim.n * 4, self.sim.n * 5),
                err_msg=msg_if_fail,
            )
        else:
            self.assertIsInstance(
                ddfm.state_space.transition_map,
                keras.Model,
                msg=msg_if_fail,
            )
        self.assertEqual(
            ddfm.state_space.transition_covariance.shape,
            ((self.sim.n + self.sim.r) * 5, (self.sim.n + self.sim.r) * 5),
            msg=msg_if_fail,
        )
        np.testing.assert_array_almost_equal(
            ddfm.state_space.transition_covariance,
            np.diag(np.diag(ddfm.state_space.transition_covariance)),
            err_msg=msg_if_fail,
        )
        transition_covariance_zeroed = ddfm.state_space.transition_covariance.copy()
        transition_covariance_zeroed[: self.sim.r, : self.sim.r] = 0
        transition_covariance_zeroed[
            self.sim.r * 5 : self.sim.r * 5 + self.sim.n,
            self.sim.r * 5 : self.sim.r * 5 + self.sim.n,
        ] = 0
        np.testing.assert_array_almost_equal(
            transition_covariance_zeroed,
            np.zeros_like(transition_covariance_zeroed),
            err_msg=msg_if_fail,
        )

    def _get_model(self, structure_encoder, jointly_est_var=False, seed=123):
        ddfm = DDFM(
            structure_encoder=structure_encoder,
            structure_decoder=self.structure_decoder,
            factor_order=1,
            lags_input=self.lags_input,
            use_bias=False,
            link="relu",
            max_iter=1000,
            var_loss_weight=1 if jointly_est_var else 0,
            seed=seed,
            clipnorm=5.0,
        )
        df_x = pd.DataFrame(self.x)
        ddfm.fit(df_x, build_state_space=True, vars_mq_restrictions=self.idx_quarterly)
        return ddfm._best_model


class TestDDFMFixes(unittest.TestCase):
    """Small, fast fits pinning two fit-time behaviours of DDFM."""

    @staticmethod
    def _model(seed=3):
        return DDFM(
            structure_encoder=(2,), factor_order=1, use_bias=False, max_iter=2, epochs=5, seed=seed
        )

    def test_fit_uses_variable_order_when_quarterly_columns_come_first(self):
        """
        With vars_mq_restrictions the columns are reordered (quarterly last). The
        filtered/smoothed factors stored by fit() must be computed on that order,
        not on the caller's column order.
        """
        sim = SIMULATE(seed=11, n=8, r=2, poly_degree=1)
        x = sim.simulate(90, quarterly_vars=QuarterlyVars([0, 1], aggregation=AggregationInstr.MM))
        df = pd.DataFrame(x, columns=[f"v{i}" for i in range(8)])
        m = self._model()
        m.fit(df, build_state_space=True, vars_mq_restrictions=["v0", "v1"])
        self.assertEqual(list(m.variable_order), ["v2", "v3", "v4", "v5", "v6", "v7", "v0", "v1"])
        expected = m.state_space.filter(df[m.variable_order].values)[0][:, :2]
        np.testing.assert_array_equal(m.factors_filtered, expected)

    def test_one_step_ahead_equals_rolling_forecast(self):
        sim = SIMULATE(seed=5, n=6, r=2, poly_degree=1)
        x = sim.simulate(80, portion_missings=0.05)
        T_train = 70
        m = self._model()
        m.fit(pd.DataFrame(x[:T_train]), build_state_space=True)
        single = m.predict_one_step_ahead(pd.DataFrame(x)).values
        for t in range(T_train, x.shape[0]):
            rolling = m.predict(pd.DataFrame(x[:t]), steps_ahead=1).values[1]
            np.testing.assert_allclose(single[t], rolling, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
