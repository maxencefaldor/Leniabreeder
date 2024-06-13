from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qdax.types import Descriptor, Fitness, Genotype, Observation, RNGKey


class UnstructuredRepertoire(flax.struct.PyTreeNode):
	"""
	Class for the unstructured repertoire in Map Elites.

	Args:
		genotypes: a PyTree containing all the genotypes in the repertoire ordered
			by the centroids. Each leaf has a shape (num_centroids, num_features). The
			PyTree can be a simple Jax array or a more complex nested structure such
			as to represent parameters of neural network in Flax.
		fitnesses: an array that contains the fitness of solutions in each cell of the
			repertoire, ordered by centroids. The array shape is (num_centroids,).
		descriptors: an array that contains the descriptors of solutions in each cell
			of the repertoire, ordered by centroids. The array shape
			is (num_centroids, num_descriptors).
		observations: observations that the genotype gathered in the environment.
	"""

	genotypes: Genotype
	fitnesses: Fitness
	descriptors: Descriptor
	observations: Observation
	size: int = flax.struct.field(pytree_node=False)

	@jax.jit
	def add(
		self,
		batch_of_genotypes: Genotype,
		batch_of_descriptors: Descriptor,
		batch_of_fitnesses: Fitness,
		batch_of_observations: Observation,
	) -> UnstructuredRepertoire:
		"""Adds a batch of genotypes to the repertoire.

		Args:
			batch_of_genotypes: genotypes of the individuals to be considered
				for addition in the repertoire.
			batch_of_descriptors: associated descriptors.
			batch_of_fitnesses: associated fitness.
			batch_of_observations: associated observations.

		Returns:
			A new unstructured repertoire where the relevant individuals have been
			added.
		"""
		# Concatenate everything
		genotypes = jax.tree.map(
			lambda x, y: jnp.concatenate([x, y], axis=0),
			self.genotypes,
			batch_of_genotypes,
		)
		descriptors = jnp.concatenate([self.descriptors, batch_of_descriptors], axis=0)
		fitnesses = jnp.concatenate([self.fitnesses, batch_of_fitnesses], axis=0)
		observations = jax.tree.map(
			lambda x, y: jnp.concatenate([x, y], axis=0),
			self.observations,
			batch_of_observations,
		)

		is_empty = fitnesses == -jnp.inf

		# Fitter
		fitter = fitnesses[:, None] <= fitnesses[None, :]
		fitter = jnp.where(is_empty[None, :], False, fitter)  # empty individuals can not be fitter
		fitter = jnp.fill_diagonal(fitter, False, inplace=False)  # an individual can not be fitter than itself

		# Distance to k-fitter-nearest neighbors
		distance = jnp.linalg.norm(descriptors[:, None, :] - descriptors[None, :, :], axis=-1)
		distance = jnp.where(fitter, distance, jnp.inf)
		values, indices = jax.vmap(partial(jax.lax.top_k, k=3))(-distance)
		distance = jnp.mean(-values, where=jnp.take_along_axis(fitter, indices, axis=1), axis=-1)  # if number of fitter individuals is less than k, top_k will return at least one inf
		distance = jnp.where(jnp.isnan(distance), jnp.inf, distance)  # if no individual is fitter, set distance to inf
		distance = jnp.where(is_empty, -jnp.inf, distance)  # empty cells have distance -inf

		# Sort by distance to k-fitter-nearest neighbors
		indices = jnp.argsort(distance, descending=True)
		indices = indices[:self.size]
		is_offspring_added = jax.vmap(lambda i: jnp.any(indices == i))(jnp.arange(self.size, self.size + batch_of_fitnesses.size))

		# Sort
		genotypes = jax.tree.map(lambda x: x[indices], genotypes)
		descriptors = descriptors[indices]
		fitnesses = fitnesses[indices]
		observations = jax.tree.map(lambda x: x[indices], observations)

		return UnstructuredRepertoire(
			genotypes=genotypes,
			fitnesses=fitnesses,
			descriptors=descriptors,
			observations=observations,
			size=self.size,
		), is_offspring_added

	@partial(jax.jit, static_argnames=("num_samples",))
	def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
		"""Sample elements in the repertoire.

		Args:
			random_key: a jax PRNG random key
			num_samples: the number of elements to be sampled

		Returns:
			samples: a batch of genotypes sampled in the repertoire
			random_key: an updated jax PRNG random key
		"""

		random_key, sub_key = jax.random.split(random_key)
		grid_empty = self.fitnesses == -jnp.inf
		p = (1.0 - grid_empty) / jnp.sum(1.0 - grid_empty)

		samples = jax.tree.map(
			lambda x: jax.random.choice(sub_key, x, shape=(num_samples,), p=p),
			self.genotypes,
		)

		return samples, random_key

	@classmethod
	def init(
		cls,
		genotypes: Genotype,
		fitnesses: Fitness,
		descriptors: Descriptor,
		observations: Observation,
		size: int,
	) -> UnstructuredRepertoire:
		"""Initialize a Map-Elites repertoire with an initial population of genotypes.
		Requires the definition of centroids that can be computed with any method
		such as CVT or Euclidean mapping.

		Args:
			genotypes: initial genotypes, pytree in which leaves
				have shape (batch_size, num_features)
			fitnesses: fitness of the initial genotypes of shape (batch_size,)
			descriptors: descriptors of the initial genotypes
				of shape (batch_size, num_descriptors)
			observations: observations experienced in the evaluation task.
			size: size of the repertoire

		Returns:
			an initialized unstructured repertoire.
		"""

		# Init repertoire with dummy values
		dummy_genotypes = jax.tree.map(
			lambda x: jnp.full((size,) + x.shape[1:], fill_value=jnp.nan),
			genotypes,
		)
		dummy_fitnesses = jnp.full((size,), fill_value=-jnp.inf)
		dummy_descriptors = jnp.full((size,) + descriptors.shape[1:], fill_value=jnp.nan)
		dummy_observations = jax.tree.map(
			lambda x: jnp.full((size,) + x.shape[1:], fill_value=jnp.nan),
			observations,
		)

		repertoire = UnstructuredRepertoire(
			genotypes=dummy_genotypes,
			fitnesses=dummy_fitnesses,
			descriptors=dummy_descriptors,
			observations=dummy_observations,
			size=size,
		)

		repertoire, _ = repertoire.add(
			genotypes,
			descriptors,
			fitnesses,
			observations,
		)
		return repertoire

	def save(self, path: str = "./") -> None:
		"""Saves the grid on disk in the form of .npy files.

		Flattens the genotypes to store it with .npy format. Supposes that
		a user will have access to the reconstruction function when loading
		the genotypes.

		Args:
			path: Path where the data will be saved. Defaults to "./".
		"""

		def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
			flatten_genotype, _unravel_pytree = ravel_pytree(genotype)
			return flatten_genotype

		# flatten all the genotypes
		flat_genotypes = jax.vmap(flatten_genotype)(self.genotypes)

		# save data
		jnp.save(path + "genotypes.npy", flat_genotypes)
		jnp.save(path + "fitnesses.npy", self.fitnesses)
		jnp.save(path + "descriptors.npy", self.descriptors)
		jnp.save(path + "observations.npy", self.observations)

	@classmethod
	def load(
		cls, reconstruction_fn: Callable, path: str = "./"
	) -> UnstructuredRepertoire:
		"""Loads an unstructured repertoire.

		Args:
			reconstruction_fn: Function to reconstruct a PyTree
				from a flat array.
			path: Path where the data is saved. Defaults to "./".

		Returns:
			An unstructured repertoire.
		"""

		flat_genotypes = jnp.load(path + "genotypes.npy")
		genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)
		fitnesses = jnp.load(path + "fitnesses.npy")
		descriptors = jnp.load(path + "descriptors.npy")
		observations = jnp.load(path + "observations.npy")

		return UnstructuredRepertoire(
			genotypes=genotypes,
			fitnesses=fitnesses,
			descriptors=descriptors,
			observations=observations,
			size=fitnesses.size,
		)
