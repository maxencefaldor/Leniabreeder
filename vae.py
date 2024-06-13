from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class Encoder(nn.Module):
	latent_size: int
	features: int

	@nn.compact
	def __call__(self, x):
		x = nn.relu(nn.Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x))  # (16, 16, self.features,)
		x = nn.relu(nn.Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x))  # (8, 8, self.features,)
		x = nn.relu(nn.Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x))  # (4, 4, self.features,)
		x = x.reshape(*x.shape[:-3], -1)  # (4 * 4 * self.features,)
		x = nn.Dense(features=4 * 4 * self.features)(x)  # (4 * 4 * self.features,)
		mean = nn.Dense(features=self.latent_size)(x)
		logvar = nn.Dense(features=self.latent_size)(x)
		return mean, logvar

class Decoder(nn.Module):
	img_shape: Tuple[int, int, int]
	features: int

	@nn.compact
	def __call__(self, z):
		z = nn.relu(nn.Dense(features=4 * 4 * self.features)(z))
		z = nn.relu(nn.Dense(features=4 * 4 * self.features)(z))
		z = z.reshape(*z.shape[:-1], 4, 4, self.features)
		z = nn.relu(nn.ConvTranspose(features=self.features, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z))
		z = nn.relu(nn.ConvTranspose(features=self.features, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z))
		z = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z)
		return z

class VAE(nn.Module):
	img_shape: Tuple[int, int, int]
	latent_size: int
	features: int

	def setup(self):
		self.encoder = Encoder(latent_size=self.latent_size, features=self.features)
		self.decoder = Decoder(img_shape=self.img_shape, features=self.features)

	def reparameterize(self, random_key, mean, logvar):
		eps = jax.random.normal(random_key, shape=mean.shape)
		return eps * jnp.exp(logvar * .5) + mean

	def encode(self, x, random_key):
		mean, logvar = self.encoder(x)
		return self.reparameterize(random_key, mean, logvar)

	def encode_with_mean_logvar(self, x, random_key):
		mean, logvar = self.encoder(x)
		return self.reparameterize(random_key, mean, logvar), mean, logvar

	def decode(self, z, random_key):
		return self.decoder(z)

	def generate(self, z, random_key):
		return nn.sigmoid(self.decoder(z))

	def __call__(self, x, random_key):
		z, mean, logvar = self.encode_with_mean_logvar(x, random_key)
		logits = self.decode(z, random_key)
		return logits, mean, logvar

@jax.jit
def kl_divergence(mean, logvar):
	return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.jit
def binary_cross_entropy_with_logits(logits, labels):
	logits = nn.log_sigmoid(logits)
	return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))

@jax.jit
def loss(logits, targets, mean, logvar):
	bce_loss = binary_cross_entropy_with_logits(logits, targets).mean()
	kld_loss = kl_divergence(mean, logvar).mean()
	return bce_loss + kld_loss
