#!/usr/bin/env python3

from tensorflow.keras.layers import Lambda, Input, Dense, Layer
from tensorflow.keras.models import Model
import random as rn
import tensorflow as tf
import numpy as np

rn.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=123)
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50


inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean_layer = Dense(latent_dim, name='z_mean')(x)
z_log_var_layer = Dense(latent_dim, name='z_log_var')(x)


z_layer = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean_layer, z_log_var_layer])

encoder = Model(inputs, [z_mean_layer, z_log_var_layer, z_layer], name='encoder')

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
decoder_x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(decoder_x)

decoder = Model(latent_inputs, outputs, name='decoder')
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, [z_mean_layer, z_log_var_layer, outputs], name='vae_mlp')

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(128)

epochs = 50

for epoch in range(epochs):
    
    loss_metric = tf.keras.metrics.Mean()
    
    for x_batch_train in train_dataset:
        
        with tf.GradientTape() as tape:
            
            z_mean, z_log_var, reconstructed = vae(x_batch_train)
            
            mse_loss = mse_loss_fn(x_batch_train, reconstructed)
            mse_loss *= original_dim

            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            
            loss = tf.reduce_mean(mse_loss + kl_loss)

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        loss_metric(loss)

    if not epoch % 10:
        
        print('Epoch %s: mean loss = %s' % (epoch, loss_metric.result()))


z_mean_test, _, _ = encoder(x_test)

plt.figure(figsize=(12, 10))
plt.scatter(z_mean_test[:, 0], z_mean_test[:, 1], c=y_test)
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()

n = 30
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
start_range = digit_size // 2
end_range = (n - 1) * digit_size + start_range + 1
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.imshow(figure, cmap='Greys_r')
plt.show()