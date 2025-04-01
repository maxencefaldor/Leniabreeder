# Leniabreeder

Repository for "Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity" (ALIFE 2024).

## Installation

```bash
git clone https://github.com/maxencefaldor/Leniabreeder.git && cd Leniabreeder
```

### Using virtual environment

At the root of the repository, execute:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Using container

At the root of the repository, execute:
```bash
apptainer build \
	--fakeroot \
	--force \
	--warn-unused-build-args \
	apptainer/container.sif apptainer/container.def
```

## Run Experiments

### Using virtual environment

At the root of the repository, execute:
```bash
source venv/bin/activate
```

### Using container

At the root of the repository, execute:
```bash
apptainer shell \
	--bind $(pwd):/workspace/src/ \
	--cleanenv \
	--containall \
	--home /tmp/ \
	--no-home \
	--nv \
	--pwd /workspace/src/ \
	--workdir apptainer/ \
	apptainer/container.sif
```

### Commands

To run an experiment with the default configuration, execute the following command:
```bash
python main.py seed=$RANDOM qd=<algo>
```
with `<algo>` replaced with either `me` or `aurora`.

All hyperparameters are available in the `configs/` directory and can be overridden via the command line. For example, to run the MAP-Elites experiments as described in the paper, use:
```bash
python main.py seed=$RANDOM qd=me qd.n_generations=4_000 qd.repertoire_size=32_000 qd.fitness=pos_linear_velocity_avg qd.descriptor=[color] qd.descriptor_min=[0.,0.,0.] qd.descriptor_max=[1.,1.,1.]
```

## Analyze Experiments

When you run an experiment, a directory is created in `output/`. To analyze the results, you can either run a script from the `analysis/` directory or use the notebooks from the `notebooks/`directory. Don't forget to change `run_dir` to the path of your experiment.

For MAP-Elites, you can use `analysis/visualize_me.py` or [`notebooks/visualize_me.ipynb`](https://github.com/maxencefaldor/Leniabreeder/blob/main/notebooks/visualize_me.ipynb). For AURORA, you can use `analysis/visualize_aurora.py` or [`notebooks/visualize_aurora.ipynb`](https://github.com/maxencefaldor/Leniabreeder/blob/main/notebooks/visualize_aurora.ipynb).

## BibTeX

```
@proceedings{10.1162/isal_a_00827,
	author = {Faldor, Maxence and Cully, Antoine},
	title = {Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity},
	volume = {ALIFE 2024: Proceedings of the 2024 Artificial Life Conference},
	series = {Artificial Life Conference Proceedings},
	pages = {85},
	year = {2024},
	month = {07},
	doi = {10.1162/isal_a_00827},
	url = {https://doi.org/10.1162/isal\_a\_00827},
	eprint = {https://direct.mit.edu/isal/proceedings-pdf/isal2024/36/85/2461065/isal\_a\_00827.pdf},
}
```
