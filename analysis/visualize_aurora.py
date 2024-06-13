from pathlib import Path
import pickle
import json
from functools import partial

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from flax import serialization

from lenia.lenia import ConfigLenia, Lenia
from vae import VAE
from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire

import mediapy
from omegaconf import OmegaConf


MAX_GENOTYPES = 32


def plot_aurora_repertoire(config, repertoire, descriptors_3d):
	# Create subplots
	fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(10, 10), subplot_kw={"projection": "3d"})

	# Set title for the column
	axes[0, 0].set_title(f"AURORA repertoire {config.qd.fitness}")

	# Plot repertoire
	sc = axes[0, 0].scatter(descriptors_3d[:, 0], descriptors_3d[:, 1], descriptors_3d[:, 2], c=repertoire.fitnesses, cmap="viridis")

	# Color bar
	colorbar = fig.colorbar(sc, ax=axes[0, 0], shrink=0.5, aspect=5)

	return fig, axes


def visualize_aurora(run_dir):
	# Create directories
	visualization_dir = run_dir / "visualization"
	video_dir = visualization_dir / "video"
	video_dir.mkdir(parents=True, exist_ok=True)
	phenotype_small_dir = visualization_dir / "phenotype_small"
	phenotype_small_dir.mkdir(parents=True, exist_ok=True)
	phenotype_medium_dir = visualization_dir / "phenotype_medium"
	phenotype_medium_dir.mkdir(parents=True, exist_ok=True)
	phenotype_large_dir = visualization_dir / "phenotype_large"
	phenotype_large_dir.mkdir(parents=True, exist_ok=True)
	vae_dir = visualization_dir / "vae"
	vae_dir.mkdir(parents=True, exist_ok=True)

	# Get config
	config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

	# Init a random key
	key = jax.random.PRNGKey(config.seed)

	# Lenia
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

	# Lenia steps
	lenia_step = partial(lenia.step, phenotype_size=config.world_size, center_phenotype=False, record_phenotype=True)
	lenia_step_small = partial(lenia.step, phenotype_size=config.phenotype_size, center_phenotype=True, record_phenotype=True)
	lenia_step_medium = partial(lenia.step, phenotype_size=64, center_phenotype=True, record_phenotype=True)
	lenia_step_large = partial(lenia.step, phenotype_size=config.world_size, center_phenotype=True, record_phenotype=True)

	# Load pattern
	init_carry, init_genotype, other_asset = lenia.load_pattern(lenia.pattern)

	# Load repertoire
	_, reconstruction_fn = ravel_pytree(init_genotype)
	repertoire = UnstructuredRepertoire.load(reconstruction_fn=reconstruction_fn, path=str(run_dir) + "/repertoire/")

	# Plot repertoire
	tsne = TSNE(n_components=3, perplexity=10., max_iter=1000)
	descriptors_3d = tsne.fit_transform(repertoire.descriptors)

	fig, _ = plot_aurora_repertoire(config, repertoire, descriptors_3d)
	fig.savefig(visualization_dir / "repertoire.pdf", bbox_inches="tight")
	plt.show()
	plt.close()

	# Save repertoire html
	with open(visualization_dir / "descriptors_3d.json", "w") as f:
		json.dump(descriptors_3d.tolist(), f)

	with open("analysis/aurora_population.html", "r", encoding="utf-8") as f:
		content = f.read()

	content = content.replace("{TITLE}", f"AURORA repertoire {config.qd.fitness}")

	with open(visualization_dir / "population.html", "w", encoding="utf-8") as file:
		file.write(content)

	# Instantiate AURORA
	key, subkey_1, subkey_2 = jax.random.split(key, 3)
	phenotype_fake = jnp.zeros((config.phenotype_size, config.phenotype_size, lenia.n_channel))
	vae = VAE(img_shape=phenotype_fake.shape, latent_size=config.qd.hidden_size, features=config.qd.features)
	params = vae.init(subkey_1, phenotype_fake, subkey_2)

	with open(run_dir / "params.pickle", "rb") as params_file:
		state_dict = pickle.load(params_file)
	params = serialization.from_state_dict(params, state_dict)

	def evaluate(genotype):
		carry = lenia.express_genotype(init_carry, genotype)
		carry, accum = jax.lax.scan(lenia_step, init=carry, xs=jnp.arange(lenia._config.n_step))
		return accum

	def evaluate_small(genotype):
		carry = lenia.express_genotype(init_carry, genotype)
		carry, accum = jax.lax.scan(lenia_step_small, init=carry, xs=jnp.arange(lenia._config.n_step))
		return accum

	def evaluate_medium(genotype):
		carry = lenia.express_genotype(init_carry, genotype)
		carry, accum = jax.lax.scan(lenia_step_medium, init=carry, xs=jnp.arange(lenia._config.n_step))
		return accum

	def evaluate_large(genotype):
		carry = lenia.express_genotype(init_carry, genotype)
		carry, accum = jax.lax.scan(lenia_step_large, init=carry, xs=jnp.arange(lenia._config.n_step))
		return accum

	assert repertoire.size % MAX_GENOTYPES == 0
	for i, genotypes in enumerate(jnp.split(repertoire.genotypes, repertoire.size // MAX_GENOTYPES)):
		# Evaluate
		accum = jax.vmap(evaluate)(genotypes)
		accum_small = jax.vmap(evaluate_small)(genotypes)
		accum_medium = jax.vmap(evaluate_medium)(genotypes)
		accum_large = jax.vmap(evaluate_large)(genotypes)

		# VAE
		key, subkey_1, subkey_2 = jax.random.split(key, 3)
		latent = vae.apply(params, accum_small.phenotype, subkey_1, method=vae.encode)
		phenotype_recon = vae.apply(params, latent, subkey_2, method=vae.generate)

		# Save videos, phenotypes and VAE
		for j in range(MAX_GENOTYPES):
			genotype_index = i * MAX_GENOTYPES + j

			# Video
			mediapy.write_video(video_dir / f"{genotype_index:04d}.mp4", accum.phenotype[j], fps=50)

			# Phenotype small
			mediapy.write_image(phenotype_small_dir / f"{genotype_index:04d}.png", accum_small.phenotype[j, -1])

			# Phenotype medium
			mediapy.write_image(phenotype_medium_dir / f"{genotype_index:04d}.png", accum_medium.phenotype[j, -1])

			# Phenotype large
			mediapy.write_image(phenotype_large_dir / f"{genotype_index:04d}.png", accum_large.phenotype[j, -1])

			# VAE
			mediapy.write_image(vae_dir / f"{genotype_index:04d}.png", phenotype_recon[j, -1])


if __name__ == "__main__":
	run_dirs = [
		"/workspace/src/output/aurora/unsupervised/2024-06-12_230539_507409",
	]

	for run_dir in run_dirs:
		visualize_aurora(Path(run_dir))
