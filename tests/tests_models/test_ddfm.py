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
        r_f_and_nnlinf = cls.sim.f.shape[1]
        cls.structure_encoder = (
            (r_f_and_nnlinf * 6, r_f_and_nnlinf * 4, r_f_and_nnlinf * 2, r_f_and_nnlinf)
            if cls.sim.poly_degree > 1
            else (r_f_and_nnlinf,)
        )

    def test_fit(self):
        """
        Testing true states are recovered with R2 of at least 80%, autoencoder is consistent with decode+encode, a
        state space representation can be built.
        """
        for jointly_est_var in [True, False]:
            ddfm = self._get_model(self.structure_encoder, jointly_est_var)
            self._single_test_fit(ddfm)

    def test_replicability(self):
        """
        Testing similar states can be recovered over 2 runs of the same model (R^2 at least 95%).
        """
        ddfm1 = self._get_model(structure_encoder=self.structure_encoder)
        ddfm2 = self._get_model(structure_encoder=self.structure_encoder)
        predict1 = ddfm1.encoder(ddfm1._data_tmp)
        predict2 = ddfm2.encoder(ddfm2._data_tmp)
        r2 = self.sim.evaluate(predict2.numpy(), f_true=predict1.numpy())
        self.assertGreaterEqual(r2, 0.95, msg="Cannot reproduce states.")

    def _get_model(self, structure_encoder, jointly_est_var=False, seed=3):
        ddfm = DDFM(
            structure_encoder=structure_encoder,
            factor_order=1,
            lags_input=self.lags_input,
            use_bias=False,
            link="relu",
            max_iter=1000,
            var_loss_weight= 1 if jointly_est_var else 0,
            seed=seed,
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
