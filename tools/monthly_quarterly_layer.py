import tensorflow as tf
from tensorflow import keras


class MixedFreqMQLayer(keras.layers.Layer):
    """
    A layer implementing monthly quarterly mixed frequency aggregation.
    """
    def __init__(self, input_dim: int, start_quarterly: int):
        super().__init__()
        mm_mq = tf.concat([tf.concat([tf.eye(start_quarterly),
                                      tf.zeros((5 * input_dim-start_quarterly, start_quarterly))], axis=0),
                           # monthly to quarterly aggregation
                           tf.concat([0 * tf.eye(start_quarterly, num_columns=input_dim-start_quarterly),
                                      1 * tf.eye(input_dim, num_columns=input_dim-start_quarterly),
                                      2 * tf.eye(input_dim, num_columns=input_dim-start_quarterly),
                                      3 * tf.eye(input_dim, num_columns=input_dim-start_quarterly),
                                      2 * tf.eye(input_dim, num_columns=input_dim-start_quarterly),
                                      1 * tf.eye(input_dim-start_quarterly, num_columns=input_dim-start_quarterly),
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
        return tf.matmul(inputs_and_lags, self.w) + self.b
