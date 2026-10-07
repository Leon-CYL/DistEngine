# DistEngine

A learning project building a minimal TinyLlama inference engine, then separating
prefill and decode across GPUs to measure the effect of prefill work stealing.
The implementation follows a small, nano-vLLM-style design. See
[PROGRESS.md](PROGRESS.md) for the staged roadmap and completed work.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
cd /path/to/DistEngine
uv sync --locked
uv run distengine
uv run python -m distengine
```

Python 3.11 is selected by `.python-version`. `uv sync` creates `.venv`, installs
the locked dependencies, and installs this package in editable mode. Source edits
are immediately available to `uv run`; activating the environment is optional.
PyTorch provides tensor/model execution and Transformers provides the Hugging Face
TinyLlama reference for the next stages. No model weights are downloaded at stage 0.
Later GPU stages target Linux with NVIDIA GPUs; the setup command can run on macOS.
Ray and Modal will be added when their stages are implemented.

## How to make a package with uv

For a **new** project, use:

```sh
uv init --package --name distengine --python 3.11 --build-backend hatch
uv add torch transformers
uv sync
uv run distengine
uv build
```

This repository is already initialized, so start with `uv sync --locked`.
`--package` creates an installable package and a build backend in `pyproject.toml`.
The distribution name is `distengine`; Python imports it as `import distengine`.
Code lives in `src/distengine/`, and `[project.scripts]` maps the `distengine`
command to `distengine.main`. `uv build` writes a wheel and source archive to `dist/`.
Building is separate from publishing; it does not upload anything.

Use `uv add <dependency>` to add runtime dependencies, `uv add --dev <tool>` for
development tools, and `uv run <command>` to execute in the project environment.
Commit both `pyproject.toml` and `uv.lock`; `.venv/` and build outputs are ignored.

## Current structure

```text
DistEngine/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
├── PROGRESS.md
└── src/
    └── distengine/
        ├── __init__.py
        └── __main__.py
```

Engine components will be added as their stages are reached. Stage 0 only provides
the package scaffold, dependencies, and a small setup command.
