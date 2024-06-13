import os
import time
import pickle
from functools import partial
import logging
logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp

from lenia.lenia import ConfigLenia, Lenia
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.map_elites import MAPElites
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from common import get_metric, repertoire_variance

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs/", config_name="me")
def main(config: DictConfig) -> None:
	logging.info("Starting MAP-Elites...")

	# Init a random key
	key = jax.random.PRNGKey(config.seed)

	# Lenia
	logging.info("Initializing Lenia...")
	config_lenia = ConfigLenia(
		# Init pattern
		pattern_id=config.pattern_id,

		# Simulation
		world_size=config.world_size,
		world_scale=config.world_scale,
		n_step=config.n_step,

		# Genotype
		n_params_size=config.n_params_size,
		n_cells_size=config.n_cells_size,
	)
	lenia = Lenia(config_lenia)

	# Load pattern
	init_carry, init_genotype, other_asset = lenia.load_pattern(lenia.pattern)

	# Define the scoring function
	def fitness_fn(observation):
		fitness = get_metric(observation, config.qd.fitness, config.qd.n_keep)
		assert fitness.size == 1
		fitness = jnp.squeeze(fitness)

		failed = jnp.logical_or(observation.stats.is_empty.any(), observation.stats.is_full.any())
		failed = jnp.logical_or(failed, observation.stats.is_spread.any())
		fitness = jnp.where(failed, -jnp.inf, fitness)
		return fitness

	def descriptor_fn(observation):
		descriptor = jnp.concatenate([get_metric(observation, descriptor, config.qd.n_keep) for descriptor in config.qd.descriptor])
		return descriptor

	def evaluate(genotype):
		carry = lenia.express_genotype(init_carry, genotype)
		lenia_step = partial(lenia.step, phenotype_size=config.phenotype_size, center_phenotype=config.center_phenotype, record_phenotype=config.record_phenotype)
		carry, accum = jax.lax.scan(lenia_step, init=carry, xs=jnp.arange(lenia._config.n_step))

		fitness = fitness_fn(accum)
		descriptor = descriptor_fn(accum)
		accum = jax.tree.map(lambda x: x[-1:], accum)  # to compute variance
		return fitness, descriptor, accum

	def scoring_fn(genotypes, key):
		fitnesses, descriptors, observations = jax.vmap(evaluate)(genotypes)

		fitnesses_nan = jnp.isnan(fitnesses)
		descriptors_nan = jnp.any(jnp.isnan(descriptors), axis=-1)
		fitnesses = jnp.where(fitnesses_nan | descriptors_nan, -jnp.inf, fitnesses)

		return fitnesses, descriptors, {"observations": observations}, key

	# Compute the centroids
	descriptor_min = jnp.array(config.qd.descriptor_min)
	descriptor_max = jnp.array(config.qd.descriptor_max)
	centroids, key = compute_cvt_centroids(
		num_descriptors=descriptor_min.size,
		num_init_cvt_samples=config.qd.n_init_cvt_samples,
		num_centroids=config.qd.repertoire_size,
		minval=descriptor_min,
		maxval=descriptor_max,
		random_key=key,
	)

	# Define a metrics function
	metrics_fn = partial(default_qd_metrics, qd_offset=0.)

	# Define emitter
	variation_fn = partial(isoline_variation, iso_sigma=config.qd.iso_sigma, line_sigma=config.qd.line_sigma)
	mixing_emitter = MixingEmitter(
		mutation_fn=None,
		variation_fn=variation_fn,
		variation_percentage=1.0,
		batch_size=config.qd.batch_size
	)

	# Instantiate MAP-Elites
	me = MAPElites(
		scoring_function=scoring_fn,
		emitter=mixing_emitter,
		metrics_function=metrics_fn,
	)

	# Compute initial repertoire and emitter state
	logging.info("Initializing MAP-Elites...")
	key, subkey = jax.random.split(key)
	init_genotypes = init_genotype[None, ...].repeat(config.qd.batch_size, axis=0)
	init_genotypes += jax.random.normal(subkey, shape=(config.qd.batch_size, lenia.n_gene)) * config.qd.iso_sigma
	repertoire, emitter_state, key = me.init(
		init_genotypes,
		centroids,
		key,
	)

	metrics = dict.fromkeys(["generation", "qd_score", "coverage", "max_fitness", "n_elites", "variance", "time"], jnp.array([]))
	csv_logger = CSVLogger("./log.csv", header=list(metrics.keys()))

	# Main loop
	logging.info("Starting main loop...")

	def me_scan(carry, unused):
		repertoire, key = carry

		# ME update
		(repertoire, _, metrics, key,) = me.update(
			repertoire,
			None,
			key,
		)

		return (repertoire, key), metrics

	for generation in range(0, config.qd.n_generations, config.qd.log_interval):
		start_time = time.time()
		(repertoire, key,), current_metrics = jax.lax.scan(
			me_scan,
			(repertoire, key),
			(),
			length=config.qd.log_interval,
		)
		timelapse = time.time() - start_time

		# Metrics
		current_metrics["generation"] = jnp.arange(1+generation, 1+generation+config.qd.log_interval, dtype=jnp.int32)
		current_metrics["n_elites"] = jnp.sum(current_metrics["is_offspring_added"], axis=-1)
		del current_metrics["is_offspring_added"]
		variance = repertoire_variance(repertoire)
		current_metrics["variance"] = jnp.repeat(variance, config.qd.log_interval)
		current_metrics["time"] = jnp.repeat(timelapse, config.qd.log_interval)
		metrics = jax.tree_util.tree_map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

		# Log
		log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], metrics)
		csv_logger.log(log_metrics)
		logging.info(log_metrics)

	# Metrics
	logging.info("Saving metrics...")
	with open("./metrics.pickle", "wb") as metrics_file:
		pickle.dump(metrics, metrics_file)

	# Repertoire
	logging.info("Saving repertoire...")
	os.mkdir("./repertoire/")
	repertoire.replace(observations=jnp.nan).save(path="./repertoire/")


if __name__ == "__main__":
	main()
