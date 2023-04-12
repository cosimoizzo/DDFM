import unittest
import numpy as np
import tensorflow as tf
from tools.loss_tools import mse_missing, convergence_checker


class TestDDFM(unittest.TestCase):
    """
    A class containing tests for methods in tools.loss_tools.
    """

    def setUp(self) -> None:
        self.y_actual = np.zeros((10, 2))
        self.y_actual[2:5, :] = np.nan
        self.y_predicted = np.zeros_like(self.y_actual)

    def tearDown(self) -> None:
        del self.y_actual, self.y_predicted

    def test_mse_missing(self):
        self.assertEqual(
            np.sum(mse_missing(tf.convert_to_tensor(self.y_actual), tf.convert_to_tensor(self.y_predicted)).numpy()), 0)
        self.assertEqual(
            np.sum(mse_missing(tf.convert_to_tensor(self.y_actual), tf.convert_to_tensor(self.y_predicted)).numpy()), 0)

    def test_convergence_checker(self):
        dif, loss = convergence_checker(self.y_predicted + 0.001, self.y_predicted + 0.001, self.y_actual)
        self.assertGreater(loss, 0)
        self.assertEqual(dif, 0)


if __name__ == '__main__':
    unittest.main()
