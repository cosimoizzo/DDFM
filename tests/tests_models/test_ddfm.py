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
        cls.sim = SIMULATE(seed=seed, n=40, r=3, poly_degree=2)
        cls.x = cls.sim.simulate(150, portion_missings=0.2)
        r_f_and_nnlinf = cls.sim.f.shape[1]
        structure_encoder = (r_f_and_nnlinf * 6, r_f_and_nnlinf * 4, r_f_and_nnlinf * 2, r_f_and_nnlinf)
        cls.ddfm = DDFM(structure_encoder=structure_encoder, factor_order=1,
                        use_bias=False, link='relu', max_iter=1000)
        cls.ddfm.fit(pd.DataFrame(cls.x), build_state_space=True)

    def test_fit(self):
        factors_hat = np.mean(self.ddfm.factors_ae, axis=0)
        r2 = self.sim.evaluate(factors_hat, f_true=self.sim.f)
        self.assertGreaterEqual(r2, 0.8)

    def test_state_space(self):
        self.assertIsInstance(self.ddfm.state_space, StateSpace)


if __name__ == '__main__':
    unittest.main()