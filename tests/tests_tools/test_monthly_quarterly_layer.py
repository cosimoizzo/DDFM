import unittest
import tensorflow as tf
import numpy as np
from tools.monthly_quarterly_layer import MixedFreqMQLayer


class TestMixedFreqMQLayer(unittest.TestCase):
    """
    A class containing test for MixedFreqMQLayer.
    """

    def setUp(self) -> None:
        n_obs = 10000
        start_quarterly = 2
        n_vars = 4
        g1 = tf.random.Generator.from_seed(1)
        self.x = g1.normal(shape=(n_obs, n_vars), mean=0, stddev=1)
        x_numpy = self.x.numpy()
        self.start_quarterly = start_quarterly
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
        self.assertAlmostEquals(np.mean(np.abs(self.out_expected - output_layer.numpy())), 0, places=6)

    def test_mq_layer_plus_dense(self):
        # define linear model with mixed frequency output layer
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.x.shape[1], bias_initializer='zeros',
                                  use_bias=False),
            MixedFreqMQLayer(input_dim=self.out_expected.shape[1],
                             start_quarterly=self.start_quarterly)
        ])
        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.SGD(learning_rate=0.005))
        model.fit(self.x, self.out_expected, epochs=3000,
                  batch_size=self.x.shape[0])
        weights = model.get_weights()
        # True value of the parameter of the first layer is identity matrix, checking mean absolute error below 0.05.
        self.assertLessEqual(np.mean(np.abs(weights[0] - np.eye(self.x.shape[1]))), 0.05,
                             msg=f"value: {weights[0]}")


if __name__ == '__main__':
    unittest.main()
