import pickle

import jax.numpy as jnp
import pandas as pd

from omegaconf import OmegaConf


def get_metric(observation, metric, n_keep):
	sign, *metric, operator = metric.split("_")
	metric = "_".join(metric)

	if operator == "avg":
		operator = jnp.mean
	elif operator == "var":
		operator = jnp.var
	elif operator == "max":
		operator = jnp.max
	else:
		raise NotImplementedError

	if sign == "pos":
		sign = 1.
	elif sign == "neg":
		sign = -1.
	else:
		raise NotImplementedError

	if metric == "mass":
		return sign * operator(observation.stats.mass[-n_keep:], keepdims=True)
	elif metric == "linear_velocity":  # equivalent to traveled distance from the origin
		return sign * jnp.sqrt(jnp.square(observation.stats.center_x[-1:] - observation.stats.center_x[-n_keep]) + jnp.square(observation.stats.center_y[-1:] - observation.stats.center_y[-n_keep]))
		# return sign * operator(observation.stats.linear_velocity[-n_keep:], keepdims=True)
	elif metric == "angular_velocity":
		return sign * operator(observation.stats.angular_velocity[-n_keep:], keepdims=True)
	elif metric == "angle":
		return sign * operator(observation.stats.angle[-n_keep:], keepdims=True)
	elif metric == "center_x":
		return sign * operator(observation.stats.center_x[-n_keep:], keepdims=True)
	elif metric == "center_y":
		return sign * operator(observation.stats.center_y[-n_keep:], keepdims=True)
	elif metric == "color":
		return sign * operator(observation.phenotype[-n_keep:, ...], axis=(0, 1, 2))
	else:
		raise NotImplementedError


def get_config(run_dir):
	config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
	return config


def get_metrics(run_dir):
	with open(run_dir / "metrics.pickle", "rb") as metrics_file:
		metrics = pickle.load(metrics_file)
	try:
		del metrics["loss"]
		del metrics["learning_rate"]
	except:
		pass
	return pd.DataFrame.from_dict(metrics)


def get_df(results_dir):
	metrics_list = []
	for fitness_dir in results_dir.iterdir():
		if fitness_dir.is_file():
			continue
		for run_dir in fitness_dir.iterdir():
			# Get config and metrics
			config = get_config(run_dir)
			metrics = get_metrics(run_dir)

			# Fitness
			try:
				metrics["fitness"] = config.qd.fitness
			except:
				metrics["fitness"] = "none"

			# Run
			metrics["run"] = run_dir.name

			# Number of Evaluations
			metrics["n_evaluations"] = metrics["generation"] * config.qd.batch_size

			# Coverage
			metrics["coverage"]

			metrics_list.append(metrics)
	return pd.concat(metrics_list, ignore_index=True)


def repertoire_variance(repertoire):
	is_occupied = (repertoire.fitnesses != -jnp.inf)
	var = jnp.var(repertoire.observations.phenotype[:, -1], axis=0, where=is_occupied[:, None, None, None])
	return jnp.mean(var)
