import tensorflow as tf
from tensorflow import keras


class MixedFreqMQLayer(keras.layers.Layer):
    """
    A layer implementing monthly quarterly mixed frequency aggregation following Mariano and Murasawa (2003).
    """

    def __init__(self, input_dim: int, start_quarterly: int):
        super().__init__()
        self.start_quarterly = start_quarterly
        self.aggr_restr = [1, 2, 3, 2, 1]
        n_q = input_dim - start_quarterly
        # at lag 0: same for all
        blocks = [tf.eye(input_dim)]
        # then only quarterly
        for a in self.aggr_restr[1:]:
            block = tf.concat(
                [
                    tf.zeros((start_quarterly, input_dim)),  # monthly to zero
                    tf.concat(
                        [tf.zeros((n_q, start_quarterly)), a * tf.eye(n_q)], axis=1
                    ),
                ],
                axis=0,
            )
            blocks.append(block)
        mm_mq = tf.concat(blocks, axis=0)
        self.w = self.add_weight(
            shape=(input_dim * 5, input_dim),
            initializer=tf.keras.initializers.Constant(mm_mq),
            trainable=False,
        )

    def call(self, inputs):
        x_ext = get_input(inputs)
        y = tf.matmul(x_ext, self.w)
        return y


def get_input(inputs):
    num_lags = 5
    lags = [inputs]
    for lag in range(1, num_lags):
        rolled = tf.roll(inputs, shift=lag, axis=0)
        mask = tf.concat(
            [
                tf.zeros((lag, tf.shape(inputs)[1]), dtype=inputs.dtype),
                tf.ones(
                    (tf.shape(inputs)[0] - lag, tf.shape(inputs)[1]), dtype=inputs.dtype
                ),
            ],
            axis=0,
        )
        lags.append(tf.multiply(rolled, mask))
    return tf.concat(lags, axis=1)
