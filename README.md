# Leniabreeder

Repository for "Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity" (ALIFE 2024).

## Installation

```bash
git clone https://github.com/maxencefaldor/Leniabreeder.git && cd Leniabreeder
```

### Using virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Using container

```bash
apptainer build \
	--build-arg github_user=$GITHUB_USER \
	--build-arg github_token=$GITHUB_TOKEN \
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

Running an experiment (see previous section) creates a directory in the `output/` directory. To analyze the results, you can use run the script in the `analysis` directory or run the notebooks.

## BibTeX

```
@article{faldor2024leniabreeder,
	author    = {Faldor, Maxence and Cully, Antoine},
	title     = {Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity},
	journal   = {ALIFE},
	year      = {2024},
}
```