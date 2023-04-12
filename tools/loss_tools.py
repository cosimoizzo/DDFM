import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error as mse


@tf.function
def mse_missing(y_actual: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
    """
    Custom loss (mse) for missing data.
    """
    mask = tf.where(tf.math.is_nan(y_actual), tf.zeros_like(y_actual), tf.ones_like(y_actual))
    y_actual_ = tf.where(tf.math.is_nan(y_actual), tf.zeros_like(y_actual), y_actual)
    y_predicted_ = tf.multiply(y_predicted, mask)
    return keras.losses.mean_squared_error(y_actual_, y_predicted_)


def convergence_checker(y_prev, y_now, y_actual):
    # TODO: Consider converting all to tensorflow
    loss_minus = mse(y_prev[~np.isnan(y_actual)], y_actual[~np.isnan(y_actual)])
    loss = mse(y_now[~np.isnan(y_actual)], y_actual[~np.isnan(y_actual)])
    return np.abs(loss - loss_minus) / loss_minus, loss
