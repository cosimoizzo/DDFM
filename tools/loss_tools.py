import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse


@tf.function
def mse_missing(y_actual: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
    """
    Custom loss (mse) for missing data.
    """
    mask = tf.cast(~tf.math.is_nan(y_actual), y_predicted.dtype)
    y_actual = tf.where(mask > 0, y_actual, tf.zeros_like(y_actual))
    sq_error = tf.square(y_actual - y_predicted) * mask
    return tf.reduce_sum(sq_error, axis=-1) / tf.maximum(
        tf.reduce_sum(mask, axis=-1), 1.0
    )


def convergence_checker(y_prev, y_now, y_actual):
    loss_minus = mse(y_prev[~np.isnan(y_actual)], y_actual[~np.isnan(y_actual)])
    loss = mse(y_now[~np.isnan(y_actual)], y_actual[~np.isnan(y_actual)])
    return np.abs(loss - loss_minus) / loss_minus, loss
