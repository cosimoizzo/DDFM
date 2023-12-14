import unittest
import tensorflow as tf
import numpy as np
from tools.monthly_quarterly_layer import MixedFreqMQLayer


class TestMixedFreqMQLayer(unittest.TestCase):
    """
    A class containing test for MixedFreqMQLayer.
    """

    def setUp(self) -> None:
        n_obs = 10
        n_vars = 4
        start_quarterly = 2
        g1 = tf.random.Generator.from_seed(1)
        self.x = g1.uniform((n_obs, n_vars))
        x_numpy = self.x.numpy()
        self.mixed_freq_mq_layer = MixedFreqMQLayer(input_dim=n_vars,
                                                    start_quarterly=start_quarterly)
        self.out_expected = x_numpy.copy()
        for var in range(start_quarterly, n_vars):
            for j in range(4, n_obs):
                self.out_expected[j, var] = (x_numpy[j, var]
                                             + 2 * x_numpy[j - 1, var]
                                             + 3 * x_numpy[j - 2, var]
                                             + 2 * x_numpy[j - 3, var]
                                             + 1 * x_numpy[j - 4, var])
            j = 3
            self.out_expected[j, var] = (x_numpy[j, var]
                                         + 2 * x_numpy[j - 1, var]
                                         + 3 * x_numpy[j - 2, var]
                                         + 2 * x_numpy[j - 3, var])
            j = 2
            self.out_expected[j, var] = (x_numpy[j, var]
                                         + 2 * x_numpy[j - 1, var]
                                         + 3 * x_numpy[j - 2, var])
            j = 1
            self.out_expected[j, var] = (x_numpy[j, var]
                                         + 2 * x_numpy[j - 1, var])

    def tearDown(self) -> None:
        del self.x, self.mixed_freq_mq_layer, self.out_expected

    def test_mq_layer(self):
        output_layer = self.mixed_freq_mq_layer(self.x)
        self.assertAlmostEquals(np.sum(np.abs(self.out_expected - output_layer.numpy())), 0, places=5)


if __name__ == '__main__':
    unittest.main()
