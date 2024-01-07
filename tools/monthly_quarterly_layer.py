import tensorflow as tf
from tensorflow import keras


class MixedFreqMQLayer(keras.layers.Layer):
    """
    A layer implementing monthly quarterly mixed frequency aggregation.
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
            initializer=tf.constant_initializer(mm_mq.numpy()),
            trainable=False
        )
        self.b = self.add_weight(shape=(input_dim,), initializer="zeros", trainable=False)

    def call(self, inputs):
        return custom_op(inputs, self.w)


@tf.custom_gradient
def custom_op(x, weights):
    x_ext = get_input(x)
    result = tf.matmul(x_ext, weights)

    def custom_grad(upstream):
        inputs_gradient = tf.matmul(get_input(upstream), weights)
        weights_gradient = tf.matmul(tf.transpose(get_input(x)), get_input(upstream))
        return inputs_gradient, weights_gradient

    return result, custom_grad


@tf.function
def get_input(inputs):
    inputs_and_lags = tf.concat([inputs,
                                 tf.multiply(tf.roll(inputs, shift=1, axis=0, name=None), tf.concat(
                                     [tf.zeros((1, inputs.shape[1])),
                                      tf.ones((inputs.shape[0] - 1, inputs.shape[1]))], axis=0)),
                                 tf.multiply(tf.roll(inputs, shift=2, axis=0, name=None), tf.concat(
                                     [tf.zeros((2, inputs.shape[1])),
                                      tf.ones((inputs.shape[0] - 2, inputs.shape[1]))], axis=0)),
                                 tf.multiply(tf.roll(inputs, shift=3, axis=0, name=None), tf.concat(
                                     [tf.zeros((3, inputs.shape[1])),
                                      tf.ones((inputs.shape[0] - 3, inputs.shape[1]))], axis=0)),
                                 tf.multiply(tf.roll(inputs, shift=4, axis=0, name=None), tf.concat(
                                     [tf.zeros((4, inputs.shape[1])),
                                      tf.ones((inputs.shape[0] - 4, inputs.shape[1]))], axis=0)),
                                 ],
                                axis=1)
    return inputs_and_lags
