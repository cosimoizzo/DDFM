import unittest

import numpy as np
import pandas as pd

from models.ddfm import DDFM
from models.state_space.state_space_wrapper import StateSpace
from synthetic_dgp.simulate import SIMULATE, QuarterlyVars, AggregationInstr


class TestDDFM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed = 123
        cls.lags_input = 0
        cls.append_to_msg = ""
        cls.sim = SIMULATE(seed=seed, n=40, r=3, poly_degree=1)
        cls.x = cls.sim.simulate(150, portion_missings=0.0)
        r_f_and_nnlinf = cls.sim.f.shape[1]
        cls.structure_encoder = (
            (r_f_and_nnlinf * 6, r_f_and_nnlinf * 4, r_f_and_nnlinf * 2, r_f_and_nnlinf)
            if cls.sim.poly_degree > 1
            else (r_f_and_nnlinf,)
        )

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
        return ddfm

    def _single_test_fit(self, ddfm):
        factors_hat = np.mean(ddfm.factors_ae, axis=0)
        r2 = self.sim.evaluate(factors_hat, f_true=self.sim.f[self.lags_input :])
        self.assertGreaterEqual(
            r2, 0.8, msg=f"r2 should be greater than 0.8{self.append_to_msg}"
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

    def _single_test_predict(self, ddfm):
        mean, covs = ddfm.predict(pd.DataFrame(self.x), steps_ahead=2)
        self.assertEqual(mean.shape[0], 3)
        self.assertEqual(mean.shape[1], self.x.shape[1])
        self.assertEqual(covs.shape[0], 3 * self.x.shape[1])
        self.assertEqual(covs.shape[1], self.x.shape[1])


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
        r_f_and_nnlinf = cls.sim.f.shape[1]
        cls.structure_encoder = (
            (r_f_and_nnlinf * 6, r_f_and_nnlinf * 4, r_f_and_nnlinf * 2, r_f_and_nnlinf)
            if cls.sim.poly_degree > 1
            else (r_f_and_nnlinf,)
        )

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
        return ddfm


if __name__ == "__main__":
    unittest.main()
