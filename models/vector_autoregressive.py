import tensorflow as tf
from tensorflow.keras.layers import Layer


class VARLayerClosedForm(Layer):
    """
    Vector Autoregressive (VAR) Layer with closed form VAR dynamics estimation
    """

    def __init__(self, n_vars: int, var_order: int = 1, dtype: tf.DType=tf.float32, **kwargs):
        super(VARLayerClosedForm, self).__init__(dtype=dtype, **kwargs)
        self.n_vars = n_vars
        self.var_order = var_order
        self.coefficients = None
        self.coefficients = self.add_weight(
            shape=(self.var_order * self.n_vars, self.n_vars),
            initializer="zeros",
            trainable=False,
            name="VAR_coefficients",
            dtype=dtype,
        )

    def call(self, inputs, return_valid_only=False):
        # output is inputs_hat
        X_lags = self.build_lagged_matrix(inputs)
        outputs = tf.matmul(X_lags, self.coefficients)
        if return_valid_only:
            return outputs[self.var_order :]
        return outputs

    def update_weights_closed_form(self, x):
        """
        OLS for VAR dynamics
        """
        X_lags = self.build_lagged_matrix(x)[self.var_order :]
        X = x[self.var_order :]
        W_hat = tf.linalg.lstsq(X_lags, X, fast=True)
        self.coefficients.assign(W_hat)

    def get_config(self):
        config = super(VARLayerClosedForm, self).get_config()
        config.update({"var_order": self.var_order})
        return config

    def build_lagged_matrix(self, x):
        lags = []
        for lag in range(self.var_order):
            rolled = tf.roll(x, shift=lag + 1, axis=0)
            mask = tf.concat(
                [
                    tf.zeros((lag + 1, tf.shape(x)[1]), dtype=x.dtype),
                    tf.ones(
                        (tf.shape(x)[0] - (lag + 1), tf.shape(x)[1]), dtype=x.dtype
                    ),
                ],
                axis=0,
            )
            lags.append(rolled * mask)
        return tf.concat(lags, axis=1)


class VARAutoencoder(tf.keras.Model):
    def __init__(
        self,
        encoder: tf.keras.Model,
        var_layer: VARLayerClosedForm,
        decoder: tf.keras.Model,
        var_loss_weight: float = 1.0,
        var_loss=None,
        ae_loss=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.var_layer = var_layer
        self.decoder = decoder
        self.var_order = self.var_layer.var_order
        self.var_loss_weight = var_loss_weight
        self.var_loss = var_loss if var_loss is not None else tf.keras.losses.MeanSquaredError(dtype=var_layer.dtype)
        self.ae_loss = ae_loss if ae_loss is not None else tf.keras.losses.MeanSquaredError(dtype=var_layer.dtype)

    def call(self, x, training=False):
        z_latent = self.encoder(x)
        if training:
            self.var_layer.update_weights_closed_form(z_latent.numpy())
        z_pred = self.var_layer(z_latent)
        x_recon = self.decoder(z_latent)
        return x_recon, z_latent, z_pred

    def compute_loss(self, x_input, x_target, with_var_training: bool = True):
        x_recon, z_latent, z_pred = self.call(x_input, training=with_var_training)

        recon_loss = self.ae_loss(x_target[self.var_order :], x_recon[self.var_order :])

        var_loss = self.var_loss(z_latent[self.var_order :], z_pred[self.var_order :])

        total_loss = recon_loss + self.var_loss_weight * var_loss
        return total_loss, recon_loss, var_loss
