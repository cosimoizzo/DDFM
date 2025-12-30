import tensorflow as tf
from tensorflow import keras


class MixedFreqMQLayer(keras.layers.Layer):
    """
    A layer implementing monthly quarterly mixed frequency aggregation following Mariano and Murasawa (2003).
    """

    def __init__(self, input_dim: int, start_quarterly: int):
        super().__init__()
        mm_mq = tf.concat([tf.concat([tf.eye(start_quarterly),
                                      tf.zeros((5 * input_dim - start_quarterly, start_quarterly))], axis=0),
                           # monthly to quarterly aggregation
                           tf.concat([0 * tf.eye(start_quarterly, num_columns=input_dim - start_quarterly),
                                      1 * tf.eye(input_dim, num_columns=input_dim - start_quarterly),
                                      2 * tf.eye(input_dim, num_columns=input_dim - start_quarterly),
                                      3 * tf.eye(input_dim, num_columns=input_dim - start_quarterly),
                                      2 * tf.eye(input_dim, num_columns=input_dim - start_quarterly),
                                      1 * tf.eye(input_dim - start_quarterly, num_columns=input_dim - start_quarterly),
                                      ],
                                     axis=0)],
                          axis=1)
        self.w = self.add_weight(
            shape=(input_dim * 5, input_dim),
            initializer=lambda shape, dtype: tf.cast(mm_mq, dtype),
            trainable=False
        )

    def call(self, inputs):
        return custom_op(inputs, self.w)


@tf.custom_gradient
def custom_op(x, weights):
    x_ext = get_input(x)
    result = tf.matmul(x_ext, weights)

    def custom_grad(upstream):
        inputs_gradient = tf.matmul(get_input(upstream), weights)
        # weights_gradient = tf.matmul(tf.transpose(get_input(x)), get_input(upstream))
        return inputs_gradient, None #weights_gradient

    return result, custom_grad


@tf.function
def get_input(inputs):
    num_lags = 5
    lags = [inputs]
    for lag in range(1, num_lags):
        rolled = tf.roll(inputs, shift=lag, axis=0)
        mask = tf.concat([
            tf.zeros((lag, tf.shape(inputs)[1]), dtype=inputs.dtype),
            tf.ones((tf.shape(inputs)[0] - lag, tf.shape(inputs)[1]), dtype=inputs.dtype)
        ], axis=0)
        lags.append(rolled * mask)
    return tf.concat(lags, axis=1)
