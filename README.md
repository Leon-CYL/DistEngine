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

## Run the TinyLlama reference

From the project root, run this command in your terminal:

```sh
uv run python -m distengine.hf_reference \
  --prompt "What is the capital of France?" \
  --max-new-tokens 32 \
  --output references/factual.json \
  --revision fe8a4ea1ffedaf415f4da2f062534de366a451e6
```

The module chooses CUDA, Apple MPS, or CPU automatically and uses float32 with
eager attention. It formats the prompt with TinyLlama's chat template and runs
greedy Hugging Face generation. The first run downloads the model files.
It prints the resolved revision, prompt, token IDs, and decoded response, then
saves a JSON reference with generation settings and environment metadata.
`--max-new-tokens` is an upper limit; generation can end earlier at the EOS token.
The command overwrites the output file if it already exists.

For a repeated run, use the same command with
`--output references/factual_repeat.json`. Keep the prompt, revision, output
length, device, dtype, and library versions the same when comparing token IDs.

You can also run `uv run python -m distengine.hf_reference` with defaults. Its
default revision is `main`, which is resolved to a commit and saved under
`resolved_revision` in `references/tinyllama_hf.json`. Use that commit with
`--revision` for subsequent comparisons. View all options with:

```sh
uv run python -m distengine.hf_reference --help
```

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
├── references/
│   ├── factual.json
│   ├── factual_repeat.json
│   └── tinyllama_hf.json
└── src/
    └── distengine/
        ├── __init__.py
        ├── __main__.py
        └── hf_reference.py
```

Stage 0 provides the package scaffold and setup command. Stage 1 adds a Hugging
Face TinyLlama reference with saved token outputs. Engine components will be added
as their stages are reached.
