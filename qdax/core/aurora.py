"""Core class of the AURORA algorithm."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple, Any

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.environments.bd_extractors import AuroraExtraInfo
from qdax.types import (
	Genotype,
	Metrics,
	RNGKey,
)


class AURORA:
	"""Core elements of the AURORA algorithm.

	Args:
		scoring_function: a function that takes a batch of genotypes and compute
			their fitnesses and descriptors
		emitter: an emitter is used to suggest offsprings given a MAPELites
			repertoire. It has two compulsory functions. A function that takes
			emits a new population, and a function that update the internal state
			of the emitter.
		metrics_function: a function that takes a repertoire and computes
			any useful metric to track its evolution
	"""

	def __init__(
		self,
		emitter: Emitter,
		scoring_fn: Callable,
		descriptor_fn: Callable,
		fitness_fn: Callable,
		train_fn: Callable,
		metrics_fn: Callable,
	) -> None:
		self._emitter = emitter
		self._scoring_fn = scoring_fn
		self._fitness_fn = fitness_fn
		self._descriptor_fn = descriptor_fn
		self._train_fn = train_fn
		self._metrics_fn = metrics_fn

	@partial(jax.jit, static_argnames=("self",))
	def train(
		self,
		repertoire: UnstructuredRepertoire,
		train_state,
		random_key: RNGKey,
	) -> Tuple[UnstructuredRepertoire, AuroraExtraInfo]:
		key, subkey = jax.random.split(random_key)
		train_state, metrics = self._train_fn(
			subkey,
			repertoire,
			train_state,
		)

		# New fitnesses
		keys = jax.random.split(key, repertoire.size)
		new_fitnesses = jax.vmap(self._fitness_fn, in_axes=(0, None, 0))(repertoire.observations, train_state, keys)
		new_fitnesses = jnp.where(repertoire.fitnesses == -jnp.inf, -jnp.inf, new_fitnesses)

		# New descriptors
		keys = jax.random.split(key, repertoire.size)
		new_descriptors = jax.vmap(self._descriptor_fn, in_axes=(0, None, 0))(repertoire.observations, train_state, keys)
		new_descriptors = jnp.where((repertoire.fitnesses == -jnp.inf)[..., None], jnp.nan, new_descriptors)

		repertoire = repertoire.replace(
			fitnesses=new_fitnesses,
			descriptors=new_descriptors,
		)

		return repertoire, train_state, metrics

	def init(
		self,
		init_genotypes: Genotype,
		train_state,
		repertoire_size: int,
		random_key: RNGKey,
	) -> Tuple[UnstructuredRepertoire, Optional[EmitterState], AuroraExtraInfo, RNGKey]:
		"""Initialize an unstructured repertoire with an initial population of
		genotypes. Also performs the first training of the AURORA encoder.

		Args:
			init_genotypes: initial genotypes
			train_state: training state
			max_size: maximum size of the repertoire
			random_key: a random key used for stochastic operations.

		Returns:
			an initialized unstructured repertoire, with the initial state of
			the emitter, and the updated information to perform AURORA encodings
		"""
		fitnesses, descriptors, extra_scores, random_key = self._scoring_fn(
			init_genotypes,
			train_state,
			random_key,
		)
		observations = extra_scores["observations"]

		repertoire = UnstructuredRepertoire.init(
			genotypes=init_genotypes,
			fitnesses=fitnesses,
			descriptors=descriptors,
			observations=observations,
			size=repertoire_size,
		)

		# Get initial state of the emitter
		emitter_state, random_key = self._emitter.init(
			init_genotypes, random_key
		)

		# Update emitter state
		emitter_state = self._emitter.state_update(
			emitter_state=emitter_state,
			genotypes=init_genotypes,
			fitnesses=fitnesses,
			descriptors=descriptors,
			extra_scores=extra_scores,
		)

		return repertoire, emitter_state, random_key

	@partial(jax.jit, static_argnames=("self",))
	def update(
		self,
		repertoire: MapElitesRepertoire,
		emitter_state: Optional[EmitterState],
		random_key: RNGKey,
		train_state: AuroraExtraInfo,
	) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
		"""Main step of the AURORA algorithm.

		Performs one iteration of the AURORA algorithm.
		1. A batch of genotypes is sampled in the archive and the genotypes are copied.
		2. The copies are mutated and crossed-over
		3. The obtained offsprings are scored and then added to the archive.

		Args:
			repertoire: unstructured repertoire
			emitter_state: state of the emitter
			random_key: a jax PRNG random key
			train_state: extra info for computing encodings

		Results:
			the updated MAP-Elites repertoire
			the updated (if needed) emitter state
			metrics about the updated repertoire
			a new key
		"""
		# generate offsprings with the emitter
		genotypes, random_key = self._emitter.emit(
			repertoire, emitter_state, random_key
		)

		# scores the offsprings
		fitnesses, descriptors, extra_scores, random_key = self._scoring_fn(
			genotypes,
			train_state,
			random_key,
		)
		observations = extra_scores["observations"]

		# add genotypes and observations in the repertoire
		repertoire, is_offspring_added = repertoire.add(
			genotypes,
			descriptors,
			fitnesses,
			observations,
		)

		# update emitter state after scoring is made
		emitter_state = self._emitter.state_update(
			emitter_state=emitter_state,
			genotypes=genotypes,
			fitnesses=fitnesses,
			descriptors=descriptors,
			extra_scores=extra_scores,
		)

		# update the metrics
		metrics = self._metrics_fn(repertoire)
		metrics["is_offspring_added"] = is_offspring_added

		return repertoire, emitter_state, metrics, random_key
