import unittest

import numpy as np
import pandas as pd

from models.ddfm import DDFM
from models.state_space import StateSpace
from synthetic_dgp.simulate import SIMULATE


class TestDDFM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed = 123
        cls.lags_input = 0
        cls.sim = SIMULATE(seed=seed, n=40, r=3, poly_degree=1)
        cls.x = cls.sim.simulate(150, portion_missings=0.2)

    def test_fit(self):
        r_f_and_nnlinf = self.sim.f.shape[1]
        structure_encoder = (
            (r_f_and_nnlinf * 6, r_f_and_nnlinf * 4, r_f_and_nnlinf * 2, r_f_and_nnlinf)
            if self.sim.poly_degree > 1
            else (r_f_and_nnlinf,)
        )
        for jointly_est_var in [True, False]:
            ddfm = self._get_model(structure_encoder, jointly_est_var)
            self._single_test_fit(ddfm)

    def _get_model(self, structure_encoder, jointly_est_var=False):
        ddfm = DDFM(
            structure_encoder=structure_encoder,
            factor_order=1,
            lags_input=self.lags_input,
            use_bias=False,
            link="relu",
            max_iter=1000,
            jointly_est_var=jointly_est_var,
        )
        ddfm.fit(pd.DataFrame(self.x), build_state_space=True)
        return ddfm

    def _single_test_fit(self, ddfm):
        factors_hat = np.mean(ddfm.factors_ae, axis=0)
        r2 = self.sim.evaluate(factors_hat, f_true=self.sim.f[self.lags_input :])
        self.assertGreaterEqual(r2, 0.8, msg="r2 should be greater than 0.8")
        predict_from_auto = ddfm.autoencoder(ddfm._data_tmp)
        predict_from_encode_decode = ddfm.decoder(ddfm.encoder(ddfm._data_tmp))
        np.testing.assert_array_almost_equal(
            predict_from_auto,
            predict_from_encode_decode,
            err_msg="Autoencoder output different from decode+encode.",
        )
        self.assertIsInstance(
            ddfm.state_space, StateSpace, msg="Failed to build state_space"
        )


if __name__ == "__main__":
    unittest.main()
