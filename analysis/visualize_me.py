from pathlib import Path
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from lenia.lenia import ConfigLenia, Lenia
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.utils.plotting import get_voronoi_finite_polygons_2d

import mediapy
from omegaconf import OmegaConf


MAX_GENOTYPES = 32

# Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc("font", size=12)


def plot_me_repertoire_on_ax(ax, repertoire, minval, maxval, vmin, vmax, display_descriptors=False, cbar=True):
	assert repertoire.centroids.shape[-1] == 2, "Descriptor space must be 2d"

	repertoire_empty = repertoire.fitnesses == -jnp.inf

	# Set axes limits
	ax.set_xlim(minval[0], maxval[0])
	ax.set_ylim(minval[1], maxval[1])

	# Create the regions and vertices from centroids
	regions, vertices = get_voronoi_finite_polygons_2d(repertoire.centroids)

	# Colors
	cmap = matplotlib.cm.viridis
	norm = Normalize(vmin=vmin, vmax=vmax)

	# Fill the plot with contours
	for region in regions:
		polygon = vertices[region]
		ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

	# Fill the plot with the colors
	for idx, fitness in enumerate(repertoire.fitnesses):
		if fitness > -jnp.inf:
			region = regions[idx]
			polygon = vertices[region]
			ax.fill(*zip(*polygon), alpha=0.8, color=cmap(norm(fitness)))

	# if descriptors are specified, add points location
	if display_descriptors:
		descriptors = repertoire.descriptors[~repertoire_empty]
		ax.scatter(
			descriptors[:, 0],
			descriptors[:, 1],
			c=repertoire.fitnesses[~repertoire_empty],
			cmap=cmap,
			s=10,
			zorder=0,
		)

	# Aesthetic
	if cbar:
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
		cbar.ax.tick_params()

	return ax


def plot_me_repertoire(config, repertoire):
	# Get vmin and vmax
	vmin = get_min_fitnesses(repertoire)
	vmax = get_max_fitnesses(repertoire)

	# Get minval and maxval
	minval = jnp.array(config.qd.descriptor_min)
	maxval = jnp.array(config.qd.descriptor_max)

	# Create subplots
	fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(10, 10))

	# Set title for the column
	axes[0, 0].set_title(f"MAP-Elites repertoire {config.qd.fitness}")

	# Set the x and y labels
	axes[0, 0].set_xlabel(config.qd.descriptor[0])
	axes[0, 0].set_ylabel(config.qd.descriptor[1])

	# Plot repertoire
	plot_me_repertoire_on_ax(axes[0, 0], repertoire, minval, maxval, vmin, vmax)

	return fig, axes


def get_min_fitnesses(repertoire):
	return repertoire.fitnesses.min(initial=jnp.inf, where=(repertoire.fitnesses != -jnp.inf))


def get_max_fitnesses(repertoire):
	return repertoire.fitnesses.max(initial=-jnp.inf, where=(repertoire.fitnesses != -jnp.inf))


def visualize_me(run_dir):
	# Create directories
	visualization_dir = run_dir / "visualization"
	video_dir = visualization_dir / "video"
	video_dir.mkdir(parents=True, exist_ok=True)
	phenotype_small_dir = visualization_dir / "phenotype_small"
	phenotype_small_dir.mkdir(parents=True, exist_ok=True)
	phenotype_large_dir = visualization_dir / "phenotype_large"
	phenotype_large_dir.mkdir(parents=True, exist_ok=True)

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
	lenia_step_large = partial(lenia.step, phenotype_size=config.world_size, center_phenotype=True, record_phenotype=True)

	# Load pattern
	init_carry, init_genotype, other_asset = lenia.load_pattern(lenia.pattern)

	# Load repertoire
	_, reconstruction_fn = ravel_pytree(init_genotype)
	repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=str(run_dir) + "/repertoire/")

	# Plot repertoire
	if repertoire.centroids.shape[-1] == 2:
		fig, _ = plot_me_repertoire(config, repertoire)
		fig.savefig(visualization_dir / "repertoire.pdf", bbox_inches="tight")
		plt.close()

	def evaluate(genotype):
		carry = lenia.express_genotype(init_carry, genotype)
		carry, accum = jax.lax.scan(lenia_step, init=carry, xs=jnp.arange(lenia._config.n_step))
		return accum

	def evaluate_small(genotype):
		carry = lenia.express_genotype(init_carry, genotype)
		carry, accum = jax.lax.scan(lenia_step_small, init=carry, xs=jnp.arange(lenia._config.n_step))
		return accum

	def evaluate_large(genotype):
		carry = lenia.express_genotype(init_carry, genotype)
		carry, accum = jax.lax.scan(lenia_step_large, init=carry, xs=jnp.arange(lenia._config.n_step))
		return accum

	assert config.qd.repertoire_size % MAX_GENOTYPES == 0
	for i, genotypes in enumerate(jnp.split(repertoire.genotypes, config.qd.repertoire_size // MAX_GENOTYPES)):
		# Evaluate
		accum = jax.vmap(evaluate)(genotypes)
		accum_small = jax.vmap(evaluate_small)(genotypes)
		accum_large = jax.vmap(evaluate_large)(genotypes)

		# Save
		for j in range(MAX_GENOTYPES):
			genotype_index = i * MAX_GENOTYPES + j

			if repertoire.fitnesses[genotype_index] == -jnp.inf:
				continue

			# Video
			mediapy.write_video(video_dir / f"{genotype_index:04d}.mp4", accum.phenotype[j], fps=50)

			# Phenotype small
			mediapy.write_image(phenotype_small_dir / f"{genotype_index:04d}.png", accum_small.phenotype[j, -1])

			# Phenotype large
			mediapy.write_image(phenotype_large_dir / f"{genotype_index:04d}.png", accum_large.phenotype[j, -1])


if __name__ == "__main__":
	run_dirs = [
		"/workspace/src/output/me/neg_mass_var/2024-06-06_045445_432667",
	]
	run_dirs = [Path(run_dir) for run_dir in run_dirs]

	for run_dir in run_dirs:
		visualize_me(run_dir)
