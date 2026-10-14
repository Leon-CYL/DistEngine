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
Ray is included for local worker orchestration. The Modal CLI is installed
separately for the GPU verification commands below.

## Run the TinyLlama reference

From the project root, run this command in your terminal:

```sh
uv run python -m hf_reference \
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

You can also run `uv run python -m hf_reference` with defaults. Its
default revision is `main`, which is resolved to a commit and saved under
`resolved_revision` in `references/tinyllama_hf.json`. Use that commit with
`--revision` for subsequent comparisons. View all options with:

```sh
uv run python -m hf_reference --help
```

## Run the custom TinyLlama model

From the project root:

```sh
uv run python -m manual_generate \
  --prompt "What is the capital of France?" \
  --max-new-tokens 32
```

This loads checkpoint `fe8a4ea1ffedaf415f4da2f062534de366a451e6` in float32,
copies its weights into the custom PyTorch model, and chooses CUDA, MPS, or CPU
automatically. It uses the same pinned tokenizer and chat template as the
reference, then generates greedily without HF `generate()`. It prints generated
token IDs and decoded text; it does not save a reference JSON file.

Generation handles one unpadded request, stops at EOS or the output limit, and
recomputes the full sequence at every step. Paged KV caching is implemented in stage 6 and verified with `uv run python -m verify`.
The initial loader temporarily holds both HF and custom weights on CPU, requiring
roughly 9 GB for float32 weights plus overhead. Model files download on the first
run if they are not already cached.

Custom-model execution and HF comparison have passed, including paged KV caching. See [TinyLlama architecture notes](docs/tinyllama.md) for
the model flow and tensor shapes.

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
Engine code lives in `src/distengine/`, with model architecture in `src/models/`,
layers in `src/layers/`, and standalone tools directly under `src/`. `[project.scripts]` maps the `distengine`
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
├── docs/
│   └── tinyllama.md
├── references/
│   ├── factual.json
│   ├── factual_repeat.json
│   └── tinyllama_hf.json
└── src/
    ├── distengine/
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── engine.py
    │   ├── scheduler.py
    │   ├── model_runner.py
    │   ├── block_manager.py
    │   ├── paged_kv_cache.py
    │   ├── request.py
    │   ├── workers.py
    │   └── ray_workers.py
    ├── models/
    │   ├── __init__.py
    │   └── llama.py
    ├── layers/
    │   ├── __init__.py
    │   ├── embed_head.py
    │   ├── linear.py
    │   ├── layernorm.py
    │   ├── activation.py
    │   ├── rotary_embedding.py
    │   ├── attention.py
    │   └── sampler.py
    ├── generation.py
    ├── load_model.py
    ├── hf_reference.py
    ├── manual_generate.py
    ├── verify.py
    ├── verify_blocks.py
    ├── verify_cache.py
    ├── verify_runner.py
    ├── verify_scheduler.py
    ├── verify_engine.py
    ├── verify_workers.py
    └── verify_ray_workers.py
```

Stage 0 provides the package scaffold and setup command. Stage 1 adds a Hugging
Face TinyLlama reference with saved token outputs. Stage 2 documents the architecture;
stages 3–6 implement requests, the custom model, manual generation, and paged
KV caching, with correctness verification complete. Stages 7–9 add block management,
the model runner, and FIFO scheduling. Stage 10 engine integration is verified on
CPU/MPS and one NVIDIA Tesla T4 through Modal. Stage 10 is complete.
Stage 11 adds ordinary prefill/decode workers and thin Ray actors, verified with
a deterministic small CPU model in separate local processes.

## Block manager and shared KV storage (Stage 7)

Run the CPU-only allocation and shared-cache checks:

```bash
uv run python -m verify_blocks
```

`BlockManager` owns block assignments. `allocate()` registers a request,
`ensure_capacity()` grows its total token capacity without shrinking, and
`free()` releases its blocks. Invalid arguments raise `ValueError`, unknown
requests raise `KeyError`, and insufficient capacity raises `RuntimeError`
without changing assignments. Zero-token requests have empty block tables.

Create one `KVStorage` for a physical pool and pass it as `storage=` to each
`PagedKVCache` with matching dimensions, device, and dtype. Each view keeps its
own lengths and manager-provided block table. Install the result of
`ensure_capacity()` with `cache.update_block_table(table)` before appending.
Updates preserve existing mappings and lengths. Returned manager tables are copies.

Retire a request's cache view before calling `free()`; old views must never be
used after release. New views start with zero lengths, so reused blocks expose
only newly written tokens. `reset()` clears valid lengths while retaining block
ownership. The caller must use manager-assigned, disjoint mappings for active
requests; cache views do not enforce ownership themselves.

## Model runner verification (Stage 8)

Run the small CPU model checks, including capacity exhaustion and cleanup after
partial KV-write failures:

```bash
uv run python -m verify_runner
```

Also compare runner generation against uncached generation with the pinned
TinyLlama checkpoint for the three reference prompts:

```bash
uv run python -m verify_runner --checkpoint
```

The checkpoint mode loads model weights and uses the available model device.
Each request is explicitly started, prefills its prompt, decodes one pending
token at a time, and releases its cache and block allocation when finished.

## FIFO scheduler (Stage 9)

Run sequential scheduling and failure-cleanup checks with a small CPU model:

```bash
uv run python -m verify_scheduler
```

`Scheduler` exclusively controls a fresh runner and block pool. Submit waiting
requests with `add_request()` and call `step()` while `has_pending_requests` is
true. Each step advances one request with one model call; completed requests
release their blocks and enter `finished`. IDs stay reserved for the scheduler's
lifetime. Requests whose maximum cached history exceeds the pool are rejected.
Execution failures release resources and propagate; later waiting requests stay
queued. Failed requests retain their status and are not recorded as finished.

## Worker roles and local Ray orchestration (Stage 11)

```bash
uv run python -m verify_workers
uv run python -m verify_ray_workers
```

`PrefillWorker` and `DecodeWorker` own independent model runners, request state,
block pools, and KV storage. Ray actors construct a deterministic small CPU model
inside their processes and delegate operations to these ordinary workers.
`ready()` reports role, PID, device, active IDs, and free blocks. `close()` releases
unfinished requests; the driver then terminates actors and shuts down Ray.
Automatic actor restarts and method retries are disabled to avoid replaying
state-changing inference calls without a recovery protocol.

The ordinary checks cover output agreement, immediate EOS, one-token limits,
invalid/duplicate requests, snapshot isolation, capacity exhaustion and retry,
partial-write failure cleanup, surviving requests, and close behavior. The Ray
checks cover process isolation, dispatch, output agreement, remote exceptions,
actor survival, and unfinished-request cleanup.

The local Ray verification reuses the driver's Python environment and disables
automatic `uv run` environment replication before importing Ray. This avoids
recreating dependencies in each temporary Ray session. It also verifies that
both actors use the driver's Python executable. This setup assumes all actors
run on the same machine; remote nodes will need their own dependency setup.
Standalone modules included through wheel `force-include` are copied into the
environment. After editing a verification module, refresh that copy with
`uv sync --locked --reinstall-package distengine`, or run its source file directly
(for example, `uv run python src/verify_ray_workers.py`).

Decode tests use `prepare_local_fixture()` to populate KV inside the decode
worker. No KV is transferred between actors yet. GPU placement belongs to stage
12 and KV export/import to stage 13. IDs are unique among active requests and
can be reused after release. The Ray model factory currently supports only the
small CPU verification model, not TinyLlama checkpoint execution.

## Engine integration (Stage 10)

```bash
uv run python -m verify_engine
uv run python -m verify_engine --checkpoint
```

The engine is verified on CPU/MPS and one NVIDIA Tesla T4 through Modal, using
float32 and the pinned TinyLlama checkpoint. All three CUDA reference-prompt
outputs exactly matched uncached generation; resource release and engine reuse
also passed.

Run stage 10 on one NVIDIA T4 through Modal from the project root:

```bash
uv tool install 'modal>=1.1'
modal setup
modal run src/Modal/run_engine.py
```

The runner uses the existing `uv.lock` for remote dependencies, uploads `src/`,
and caches the pinned TinyLlama weights in a Modal Volume. It requires exactly
one visible CUDA GPU and retains float32. It runs the CPU integration suite,
checks CUDA engine outputs against uncached generation for all three reference
prompts, and verifies block release and engine reuse. GPU output is printed and
saved locally to `results/stage10-modal.json` only after all checks pass.
The first run builds the image and downloads weights. Modal GPU runs incur
usage charges under your account.

To add a custom prompt and choose an output file:

```bash
modal run src/Modal/run_engine.py --prompt "Explain paged KV caching." \
  --max-new-tokens 32 --output results/stage10-custom.json
```

Reference prompts use a 32-token limit unless they are the selected prompt.
The output file is overwritten on a successful run. Stage 10's GPU checkpoint
passed on October 12, 2026; this single-GPU helper does not implement the later
two-GPU deployment stage.

```python
from distengine.engine import LLMEngine

# model is an already loaded custom LlamaForCausalLM on your chosen device.
engine = LLMEngine(model, num_blocks=32, block_size=16)
request = engine.add_request("example", [10, 20, 30], 8, eos_token_id=2)
completed = engine.run_until_complete()
print(completed[0].generated_token_ids)
```
