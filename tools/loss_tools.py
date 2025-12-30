import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse


@tf.function
def mse_missing(y_actual: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
    """
    Custom loss (mse) for missing data.
    """
    y_actual = tf.cast(y_actual, y_predicted.dtype)
    mask = tf.cast(~tf.math.is_nan(y_actual), y_predicted.dtype)
    y_actual = tf.where(mask > 0, y_actual, tf.zeros_like(y_actual))
    sq_error = tf.square(y_actual - y_predicted) * mask
    return tf.reduce_sum(sq_error, axis=-1) / tf.maximum(
        tf.reduce_sum(mask, axis=-1), 1.0
    )


def np_mse_missing(y_actual: np.ndarray, y_now: np.ndarray, mask: np.ndarray) -> float:
    return mse(y_now[mask], y_actual[mask])
