# CSC3034 Computational Intelligence Labs

Teaching material and runnable examples for the CSC3034 Computational
Intelligence practical sessions. The site is built with Material for MkDocs.

## Local setup

Create the Conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate csc3034-labs
```

Alternatively, create a Python 3.12 virtual environment and install the core
lab dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/files/requirements.txt
python -m pip install mkdocs mkdocs-material
```

Deep-learning and computer-vision examples require the optional dependencies:

```bash
python -m pip install -r src/files/requirements-deep-learning.txt
```

NVIDIA Isaac Sim has its own installation and Python environment. Follow the
instructions on the site instead of installing Isaac Sim into the environment
above.

## Preview and build the site

```bash
mkdocs serve
mkdocs build --strict --clean
```

The generated static site is written to `docs/`. The configuration deliberately
uses HTML filenames so the generated site can also be opened from the local
filesystem.

## Run examples

Run a core example from the repository root:

```bash
python src/files/lab3_ga.py
```

Set `MPLBACKEND=Agg` when running plotting examples in a headless environment.

## Quality checks

```bash
python -m compileall -q src/files hooks
python -m unittest discover -s tests -v
ruff check src/files hooks tests
mkdocs build --strict --clean
```

Source material belongs in `src/`. Do not edit generated files under `docs/`
directly; rebuild the site after changing the source.
