import unittest
import numpy as np
import tensorflow as tf
from statsmodels.tsa.api import VAR

from models.vector_autoregressive import VARLayerClosedForm, VARAutoencoder


class TestVARLayerClosedForm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(0)

    def test_against_stats_models_var1(self):
        # Simulate VAR(1) process
        T = 300
        r = 3
        p = 1
        true_coef = np.array([[[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.2, 0.0, 0.3]]])
        Z = np.zeros((T, r))
        noise = self.rng.normal(size=(T, r))
        for t in range(T):
            Z[t] = Z[t - 1] @ true_coef[0].T + noise[t]
        model = VAR(Z)
        results = model.fit(maxlags=p, trend="n")
        sm_coef = results.coefs
        Z_tf = tf.convert_to_tensor(Z, dtype=tf.float32)
        var_layer = VARLayerClosedForm(n_vars=r, var_order=p)
        var_layer.update_weights_closed_form(Z_tf)
        # compare coefficients
        varlayer_coef = var_layer.coefficients.numpy()
        np.testing.assert_allclose(
            varlayer_coef.T, sm_coef.squeeze(), rtol=1e-5, atol=1e-5
        )
        # compare predictions
        np.testing.assert_allclose(
            var_layer(Z_tf, return_valid_only=True).numpy().squeeze(),
            results.fittedvalues,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_against_stats_models_var2(self):
        # Simulate VAR(1) process
        T = 300
        r = 2
        p = 2
        true_coef = np.array(
            [
                [[0.5, 0.1], [0.0, 0.4]],
                [[-0.3, 0.0], [0.1, -0.7]],
            ]
        )
        Z = np.zeros((T, r))
        noise = self.rng.normal(size=(T, r))
        for t in range(T):
            Z[t] = Z[t - 1] @ true_coef[0].T + Z[t - 2] @ true_coef[1].T + noise[t]
        model = VAR(Z)
        results = model.fit(maxlags=p, trend="n")
        sm_coef = results.coefs
        Z_tf = tf.convert_to_tensor(Z, dtype=tf.float32)
        var_layer = VARLayerClosedForm(n_vars=r, var_order=p)
        var_layer.update_weights_closed_form(Z_tf)
        # compare coefficients
        varlayer_coef = var_layer.coefficients.numpy()
        # lag 1
        np.testing.assert_allclose(
            varlayer_coef.T[:, :r], sm_coef[0].squeeze(), rtol=1e-5, atol=1e-5
        )
        # lag 2
        np.testing.assert_allclose(
            varlayer_coef.T[:, r:], sm_coef[1].squeeze(), rtol=1e-5, atol=1e-5
        )
        # compare predictions
        np.testing.assert_allclose(
            var_layer(Z_tf, return_valid_only=True).numpy().squeeze(),
            results.fittedvalues,
            rtol=1e-5,
            atol=1e-5,
        )


class TestVARAutoencoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(0)

    def test_var_1_autoencoder(self):
        T = 300
        r = 1
        p = 1
        d = 3
        true_autoregcoef = np.array([[[0.5]]])
        true_coef = np.array([[[0.5], [0.8], [0.1]]])
        z = np.zeros((T, r))
        noise = self.rng.normal(size=(T, r))
        for t in range(T):
            z[t] = z[t - 1] @ true_autoregcoef[0].T + noise[t]
        x = z @ true_coef[0].T
        # setting up autoencoder
        inputs = tf.keras.Input(shape=(d,))
        encoded = tf.keras.layers.Dense(
            r, activation=None, kernel_initializer="ones", bias_initializer="zeros"
        )(inputs)
        encoder = tf.keras.Model(inputs, encoded)
        decoded = tf.keras.layers.Dense(
            d, activation=None, kernel_initializer="ones", bias_initializer="zeros"
        )(encoded)
        decoder = tf.keras.Model(encoded, decoded)
        var_layer = VARLayerClosedForm(n_vars=r, var_order=p)
        var_autoencoder = VARAutoencoder(encoder, var_layer, decoder)
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(0.01)
        epochs = 300
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                total_loss, recon_loss, var_loss = var_autoencoder.compute_loss(
                    x_tf, x_tf
                )
            grads = tape.gradient(
                total_loss, encoder.trainable_variables + decoder.trainable_variables
            )
            optimizer.apply_gradients(
                zip(grads, encoder.trainable_variables + decoder.trainable_variables)
            )
            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1}, Total Loss: {total_loss.numpy():.4f}, "
                    f"Recon Loss: {recon_loss.numpy():.4f}, VAR Loss: {var_loss.numpy():.4f}"
                )
        x_recon, z_latent, z_pred = var_autoencoder(x_tf)
        z_latent = z_latent.numpy()
        corr = np.corrcoef(z_latent.T, z.T)[0, 1]
        self.assertGreaterEqual(corr, 0.95, msg="Factor space not recovered")
        autocov = np.cov(z_latent[:-1].T, z_latent[1:].T)
        autcoef = autocov[0, 1] / autocov[0, 0]
        self.assertAlmostEqual(
            autcoef,
            var_layer.coefficients.numpy()[0, 0],
            places=3,
            msg="Autoregressive does not match calculated ones.",
        )


if __name__ == "__main__":
    unittest.main()
