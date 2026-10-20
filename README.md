# DistEngine

The active benchmark uses **two prefill plus two decode A100 80 GB replicas
at 5 RPS**, matching the selected Stage 21 baseline: 200 Poisson requests,
three passes, float32, decode batch four and four warm-up requests per pass.
The launcher requests `A100-80GB:4` and checks at least 75 GiB per GPU.
Stage 22 adds optional prefill work stealing with `--stealing`; disabled is the
default. Both modes retain independent prefill/decode execution. Reports and
default output directories distinguish the two modes. Per-pass deadlines are
disabled; the overall Modal timeout remains 600 seconds.
Eight-GPU and A100 40 GB trials below are historical/deferred configurations.

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

Stage 11 decode tests use `prepare_local_fixture()` to populate KV inside the
decode worker and test roles independently. Stages 12 and 13 add GPU placement
and actual KV handoff. IDs are unique among active requests and can be reused
after release. CPU tests use the small model; GPU tests also support the pinned
TinyLlama checkpoint.

## Two-GPU worker placement (Stage 12)

Verified on October 13, 2026 with pinned TinyLlama on two Tesla T4 GPUs: separate
actor/GPU placement, all three reference-prompt baseline comparisons, and cleanup
passed. Stage 13 subsequently verified cross-actor handoff too.

```bash
modal run src/Modal/run_two_gpu.py
```

This allocates two T4 GPUs in one container, starts a local Ray runtime, and
reserves one GPU per actor. Each actor sees its assigned GPU as `cuda:0`; the
verification checks disjoint Ray GPU IDs, separate processes, and model/KV
placement. It loads the pinned TinyLlama checkpoint in float32 and checks all
three reference prompts against uncached CUDA generation, then checks full
release and closing unfinished requests. Model files use the existing Modal
Hugging Face cache Volume. Two-GPU usage incurs Modal charges.

Successful runs save `results/stage12-placement.json` with placement, environment,
and token outputs. The file is overwritten only after successful verification.
Use `--no-checkpoint` for the small deterministic model without downloads, or
`--output results/stage12-custom.json` to choose the local output path.

On an existing Linux machine with exactly two NVIDIA GPUs, run the source
directly with `uv run python src/verify_gpu_workers.py --checkpoint`. No GPU
fallback is allowed. The stage 11 CPU actor defaults remain unchanged.

Decode still prepares KV locally; these checks verify separate-GPU placement and
independent execution. The roadmap's end-to-end two-GPU output checkpoint is
verified by stage 13's KV transfer path.

## KV handoff (Stage 13)

Verified on October 13, 2026 with pinned TinyLlama in float32 on two Tesla T4s:
all three transferred requests matched uncached CUDA generation exactly, with
block release/reuse and unfinished-request cleanup passing.

```bash
uv run python src/verify_transfer.py
modal run src/Modal/run_transfer.py
```

Prefill exports versioned CPU tensors in logical token order, plus request ID,
prompt/output IDs, output limit, EOS, cache length, and model/config identity.
Physical block IDs are not transferred. Decode validates the payload, allocates
its own blocks, copies each layer into its local device storage, and continues
from the pending first output token without recomputing the prompt. The source
releases its blocks only after the destination acknowledges a successful import.
Requests finishing during prefill complete there without a KV transfer.

This initial implementation uses GPU → CPU → GPU through ordinary Ray object
transport, not direct device transfer. Imports reject incompatible models,
versions, metadata, shapes, dtypes, nonfinite KV, and duplicate active IDs. Copy
failures roll back destination allocation while source KV remains available.
Inference weights/configuration must remain immutable: pinned checkpoints use
revision identity; other models cache a fingerprint of their weights and config.
Retries are explicit, and no durable or exactly-once handoff protocol is claimed.

The Modal runner checks all three pinned TinyLlama reference prompts end to end
across separate GPUs and saves `results/stage13-transfer.json` after success.
`--no-checkpoint` selects the small deterministic model. The stage 12 runner
continues to test independent worker execution; stage 13 enables actual handoff
in the shared GPU verification module.

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

## Reproducible workloads (Stage 18)

Run the CPU verification without model downloads or GPUs:

```bash
PYTHONPATH=src uv run python src/verify_workload.py
```

This checks constant and seeded Poisson schedules, JSON validation, replay
timing with a fake clock, and save/load/replay through two small-model prefill
workers and one decode worker. Both arrival modes verify six exact baseline
outputs, completion delivery, drained queues, and full KV release.

Generate and save a workload from Python:

```python
from distengine.workload import (
    Workload, generate_workload, save_workload, load_workload, replay_workload,
)

requests = generate_workload(
    arrival_mode="poisson",  # Or "constant".
    rate_rps=4.0,
    num_requests=6,
    seed=42,
    prompt_token_ids=[10, 20, 30],  # Example IDs for the small CPU model.
    max_new_tokens=4,
    eos_token_id=-1,  # Disable EOS matching for this fixture.
)
workload = Workload(
    schema_version=1, arrival_mode="poisson", rate_rps=4.0,
    seed=42, requests=tuple(requests),
)
save_workload(workload, "workload.json")
restored = load_workload("workload.json")
assert restored == workload

# With a fresh MultiPrefillCoordinator or MultiPrefillRayCoordinator:
# results = replay_workload(restored, coordinator)
```

Use actual tokenizer-produced IDs and EOS for checkpoint runs. Constant traffic
starts at zero; Poisson traffic starts after the first sampled exponential gap.
Offsets are seconds relative to replay start. JSON stores the exact tokens and
schedule alongside the seed and settings; loading does not regenerate arrivals.
Saving replaces an existing destination after validating the workload.

Replay requires an exclusively owned coordinator with no pending work; use a
fresh coordinator for each run because request IDs remain reserved. It submits
due requests in schedule order and drains inference before returning completion
dictionaries. Its monotonic clock and sleep function can be injected for tests.
Coordinator calls block, so actual submissions can be late; overdue arrivals
are submitted before the next inference step. Exceptions propagate, and the
caller remains responsible for worker cleanup.

Stage 18 establishes CPU correctness. GPU/Ray replay and performance measurement
are unverified; mixed prompt lengths follow in stage 19, with timing metrics
and baseline benchmarking in stage 21.

## Mixed prompt workload (Stage 19)

The frozen benchmark input is `workloads/stage19/mixed-v1.json`. It contains
100 requests with exact counts of 60 short, 30 medium, and 10 long prompts,
Poisson arrivals at an intended average 2 RPS, and a 32-token output limit.
EOS may finish requests earlier. Nine prompts use three topics with repeated
context paragraphs; measured lengths after the pinned TinyLlama chat template
are 53–64, 137–160, and 520–529 tokens respectively.

`prompt-bank.json` saves source text and token IDs. `manifest.json` records
model/tokenizer revision, both seeds, selection settings, request-to-prompt
mapping, SHA-256 hashes, and round-robin token totals. For two prefill sources,
those totals are 6,404 and 6,751: a modest 5.4% difference in input tokens.
This input distribution alone does not establish runtime imbalance or a benefit
from stealing; stage 21 measures the baseline and stage 22 will compare modes.

Run the verification from the repository root:

```bash
PYTHONPATH=src uv run python src/verify_mixed_workload.py
# Also verify cached pinned-tokenizer outputs and byte-identical rebuilding:
PYTHONPATH=src uv run python src/verify_mixed_workload.py --tokenizer
```

The default checks need no model download or GPU. CPU inference checks use a
small model with token IDs mapped into its vocabulary while preserving all nine
prompt lengths; they do not run the 100-request TinyLlama benchmark. The optional
check requires the pinned tokenizer to be cached locally.

To reproduce the frozen files in a new directory:

```bash
PYTHONPATH=src uv run python src/freeze_workload.py \
  --output /tmp/distengine-stage19-rebuild --local-files-only
```

The destination must be empty. Omit `--local-files-only` to permit a tokenizer
download. The helper uses the pinned revision, fixed settings, and independent
arrival/selection seeds. Class quotas use largest-remainder rounding, then a
seeded shuffle; prompt choice within each class is uniform. The library API is
`distengine.mixed_workload.generate_mixed_workload()` for other configurations.

Replay the saved file with `load_workload()` and `replay_workload()` as in stage
18. Configure every worker with at least 1,024 KV token slots (for example,
`num_blocks=64, block_size=16`) to match the manifest's capacity validation.
Keep the frozen request order and file checksums identical across comparisons.
Stage 21 collects the GPU baseline at the frozen offered rate;
the frozen 2 RPS setting is an experimental input, not a capacity estimate.

## Multiple decode workers (Stage 20)

Multiple decode workers now precede baseline benchmarking; execution continues
19 → 20 → 21 → 22. Existing `stage22-multi-decode.json` reports retain their
original filenames. The baseline and later stealing comparison
must use the same prefill/decode worker counts and capacity settings.

```python
from distengine.multi_decode_coordinator import MultiDecodeCoordinator
from distengine.multi_decode_ray_coordinator import MultiDecodeRayCoordinator

# Each list contains fresh, independent workers (or Ray actor handles).
# coordinator = MultiDecodeCoordinator(prefill_workers, decode_workers)
# coordinator = MultiDecodeRayCoordinator(prefill_actors, decode_actors)
# results = replay_workload(load_workload("workloads/stage19/mixed-v1.json"), coordinator)
```

Submission remains round-robin across prefill workers. Decode placement happens
when held KV can be admitted: the driver rotates target/source priority and
checks each target independently. A full target does not block another target.
After import, the request stays on its assigned decoder until it finishes;
`coordinator.decode_assignments` records this placement. Prefill-only completions
have no decode assignment. Each `decode_ready[index]` is a target's round-robin
queue. Each tick advances one request per nonempty target and admits at most
one new request per target. Ray decode calls are dispatched across actors before
waiting, and concurrent prefill dispatch is preserved.

Every submitted request must fit the assigned source and every target. Models
must remain compatible and immutable. Source KV stays owned until import is
acknowledged. Pending source releases are tracked before release can fail, so a
later caller tick can reconcile without importing twice. Exceptions propagate;
completions from successful peers can be recovered with `drain_completed()`.
Unknown actor outcomes require caller intervention. The caller owns worker
cleanup; closing workers releases unfinished KV. Existing single-decode
coordinators remain available.

Run the ordinary-worker suite and the real four-process Ray CPU suite:

```bash
PYTHONPATH=src uv run python src/verify_multi_decode.py
PYTHONPATH=src uv run python src/verify_multi_decode_ray.py
```

The ordinary suite includes capacity guards, failure recovery, dispatch-order
checks, per-target fairness, workload replay and exact small-model output checks.
Ray checks inspect ownership every tick, both decoders progressing, holding and
upstream pressure, complete output agreement, and unfinished-request cleanup.

Run the four-GPU correctness checkpoint on Modal (GPU charges apply):

```bash
modal run src/Modal/run_multi_decode.py
# Optional small-model GPU checkpoint:
modal run src/Modal/run_multi_decode.py --no-checkpoint \
  --output results/stage22-small-multi-decode.json
```

This uses two prefill and two decode actors on four distinct T4s. It runs CPU
checks first, then compares eight transferred requests with uncached CUDA
baselines, verifies fixed decode ownership and GPU placement, and closes
unfinished work. Successful runs save `results/stage22-multi-decode.json`.
This checkpoint uses short reference prompts; the frozen stage 19 workload
still requires 1,024 KV token slots per worker when used for benchmarking.
No performance claim is made by these correctness checks.

Verified on October 18, 2026: ordinary CPU, real four-process Ray CPU, and pinned
TinyLlama float32 on four Tesla T4s all passed. Eight GPU outputs matched
uncached generation in 94 ticks, with fixed ownership and full cleanup.
The saved report is `results/stage22-multi-decode.json`.

## Driver request metrics (Stage 21)

`distengine.metric` provides `MetricsRecorder` and `LatencyLimits`. Ordinary and
Ray multi-prefill/multi-decode coordinators accept an optional `metrics=` recorder.
Replay uses that same recorder and clock, records submission delay, and freezes
measurement duration when the workload drains. Models and warm-up should be
ready before starting measurement. For example, with a fresh coordinator:

```python
from distengine.metric import LatencyLimits, MetricsRecorder
from distengine.workload import load_workload, replay_workload

# coordinator already owns fresh workers; model setup is outside measurement.
workload = load_workload("workloads/stage19/mixed-v1.json")
metrics = MetricsRecorder()
coordinator.metrics = metrics
results = replay_workload(workload, coordinator, metrics=metrics)
# Example limits only: choose benchmark thresholds before running comparisons.
report = metrics.report(LatencyLimits(ttft_s=2.0, tpot_s=0.1, e2e_s=10.0))
```

Reports include raw timestamps, TTFT, mean per-request TPOT, E2E, submission delay,
wait until first prefill dispatch, p50/p95/p99, completed requests/second,
completed output tokens/second, goodput, failures, and unfinished counts.
One-token outputs have undefined TPOT and use only TTFT/E2E for goodput. Prefill
counts the first token; KV import never counts it twice. Ray results are observed
in readiness order, and local results are observed before the next worker runs.
These measurements include driver/transport delays and are not GPU kernel times.

A coordinator exception can leave a request live, such as after successful import
but failed source release. Replay records such exceptions separately; only rejected
submissions and definitively retired decode requests are classified as failed.
Errors still propagate, and the caller remains responsible for worker cleanup.

```bash
PYTHONPATH=src uv run python src/verify_metrics.py
PYTHONPATH=src uv run python src/verify_multi_decode_ray.py --metrics
```

These passed CPU and four-process Ray verification. The system sampler and
benchmark runner below add queue/KV state, worker activity, GPU utilization,
and repeated baseline reports.

## System sampling and baseline benchmark (Stage 21)

The benchmark records request timing alongside upstream/decode queue lengths,
held requests, KV blocks used, per-worker activity/idle time, and background
`nvidia-smi` utilization/memory samples. Workers synchronize CUDA at the boundaries
of measured operations. Nested operations count once, so a decode's internal
release is not double-counted. Worker idle means wall time outside measured
operations; GPU utilization is measured separately.

Check the sampler and run a small CPU benchmark:

```bash
PYTHONPATH=src uv run python src/verify_system_metrics.py
PYTHONPATH=src uv run python src/benchmark_stage21.py --cpu --repeats 3 \
  --output results/stage21-my-cpu-smoke
```

Start with a cost-bounded calibration on two prefill and two decode A100 80 GB
GPUs (Modal charges apply). The current trial has 200 requests at 4 RPS,
16 output tokens maximum, three repeats, and four warm-up requests:

```bash
modal run src/Modal/run_benchmark.py \
  --repeats 3 --max-run-seconds 90 \
  --output results/stage21-a100-200-rps4
```

The last scheduled arrival is at 50.012 seconds; this is not a runtime guarantee.
Measured replay stops between calls at 90 seconds by default, saving partial
results if necessary. It skips remaining repeats after a partial run, validates
completed outputs, and closes unfinished workers. Inspect `complete`, `status`,
`not_submitted`, request unfinished counts, and `cleanup` in the report. Partial
reports are censored calibration data, not a completed performance baseline.

The Modal function timeout is 600 seconds, including remote reference generation,
model loading, warm-up and replay inside the function. A soft deadline cannot
interrupt a blocking Ray/model call. A hard timeout or setup failure may prevent
report export. Startup, billing granularity and other resources mean this is not
an exact dollar cap. The launcher prints a GPU-only cost estimate before invoking
GPU work. No new GPU job was launched while preparing this redesign.

Defaults are 128 blocks of 16 tokens per worker (2,048 aggregate KV slots),
prefill holding capacity four, 0.5-second sampling, 8 CPU cores and 64 GiB host RAM.
Float32, the pinned model, normal EOS and the 60/30/10 prompt-class distribution
are retained. The 16-token cap is a deliberate shorter calibration workload, so
its throughput is not directly comparable to the old 32-token runs.

If the ignored local workload files are missing, create them once:

```bash
PYTHONPATH=src uv run python src/freeze_workload.py \
  --output workloads/a100-200-rps4 --num-requests 200 --rate-rps 4 \
  --max-new-tokens 16 --kv-capacity-tokens 2048
```

The launcher exposes `--max-run-seconds` (up to 180), `--repeats`,
`--num-blocks`, `--max-held-requests`, and `--warmup-requests`. The requested
sum of measured time limits cannot exceed 270 seconds (three passes of 90 seconds). For example, for a smaller
calibration, two repeats with `--max-run-seconds 60 --repeats 2`
fit this ceiling. Use a new output directory each time. This ceiling does not
include reference generation, warm-up, setup or other billing overhead.

The previous 1,000-request/three-repeat configuration remains in
`workloads/a100-rps4/` for later experiments; it is no longer the default.
Increase only after observing completed requests/second and budget headroom.
The launcher requests exactly `A100-80GB:4`, checks GPU family/memory, and records
hardware identity. GPU benchmarks use independent prefill/decode scheduling
and support decode batching with `--decode-batch-size`.

Default goodput thresholds are TTFT 2 seconds, TPOT 0.1 seconds and E2E 10 seconds.
Override with `--ttft-limit`, `--tpot-limit`, and `--e2e-limit` before starting an
experiment, and preserve those settings across comparisons. These initial limits
are experiment definitions, not hardware performance guarantees.

Each new output directory contains `baseline.json` and `run-00.json`,
`run-01.json`, etc. Reports include raw measurements, outputs, model/configuration,
original frozen inputs, source/dependency hashes, placement, sampling settings,
per-run latency distributions, and mean/standard deviation across repeats.
Existing output directories are rejected to preserve previous results.

Queue/KV snapshots are taken between ticks, so transient peaks can be missed.
Worker activity windows use each worker's reset/stop RPC rather than the exact
request window. Sampling, RPC, and explicit CUDA synchronization overhead remains
inside the measured run. GPU utilization uses NVIDIA's sampling window; GPU memory
includes weights and allocator caches. Request latency starts at actual submission;
inspect submission delay as well when the blocking driver falls behind arrivals.
KV occupancy counts allocated/reserved blocks, including full decode reservations,
rather than only valid token positions already written.

T4 validation completed on October 18, 2026. Three four-T4 TinyLlama repeats each
completed all 100 frozen requests with exact uncached CUDA outputs and full KV
cleanup. Mean throughput was 0.9834 requests/second and 31.4678 output tokens/second.
Goodput was zero under the predefined latency limits; output correctness and
measurement verification passed, while those latency targets were not met.
The saved baseline is `results/stage21-baseline/baseline.json`, with separate
per-run JSON files in the same directory. Use a new output directory for reruns.
T4 runs serve validation; the new A100 performance profile has passed input and
CPU configuration checks but awaits its GPU run. Changing hardware, load and
capacity together establishes a new baseline rather than isolating GPU speedup.

### Lower-load comparison: 200 requests at 2 RPS

`workloads/a100-200-rps2/` preserves every request, prompt selection, output limit,
and request order from `a100-200-rps4`. Poisson arrival offsets are exactly doubled;
the final arrival is at 100.024 seconds. Use one pass with a 150-second measured
limit, leaving the hardware/worker configuration unchanged:

```bash
modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/a100-200-rps2 \
  --repeats 1 --max-run-seconds 150 \
  --output results/stage21-a100-200-rps2
```

The saved T4 baseline, A100 4-RPS baseline, and CPU smoke report all use Poisson
arrivals, not constant intervals. A configured RPS is their expected average rate;
individual gaps vary. Verification scripts also test constant arrival generation.

### Current GPU choice: four A100 40 GB replicas

The benchmark launcher now requests `A100-40GB:4`, with two prefill and two
decode replicas. Its runtime check requires A100 GPUs with at least 35 GiB of
reported memory (rather than the old 75-GiB minimum for 80-GB hardware).
The explicit Modal request selects the 40-GB variant; actual identity is saved.
All earlier 80-GB measurements remain unchanged.

Rerun the identical 200-request, Poisson 4-RPS input with three passes and a
90-second measured limit per pass:

```bash
modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/a100-200-rps4 \
  --repeats 3 --max-run-seconds 90 \
  --output results/stage21-a100-40gb-200-rps4
```

The 16-token output cap, 128 KV blocks per worker, holding capacity four,
four-request warm-up and 600-second function timeout are unchanged. A partial
pass still stops subsequent repeats. This change has passed mocked launcher and
input checks; no GPU run was launched while preparing it.

### Full-drain 40 GB rerun

The default per-pass deadline is now disabled at the user's request. Use zero
explicitly to drain every request in each pass:

```bash
modal run src/Modal/run_benchmark.py --repeats 3 --max-run-seconds 0 \
  --output results/stage21-a100-40gb-200-rps4-drained
```

The 600-second function timeout remains and includes setup plus all passes.
There is no measured-budget guard when the per-pass deadline is disabled.
The previous partial report is preserved. No GPU execution was launched while
preparing this change.


### Decode batching benchmark

Use `--decode-batch-size 4` to process up to four active requests per decode
worker in one model forward. Batches update every scheduling tick; they do not
wait to fill. Prefill remains unbatched. The default batch size is one.

```bash
PYTHONPATH=src uv run python src/verify_decode_batch.py --ray
modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/a100-200-rps4 \
  --decode-batch-size 4 --repeats 3 --max-run-seconds 0 \
  --output results/stage21-a100-80gb-200-rps4-batch4
```

This uses two prefill and two decode A100 80 GB replicas and preserves the
200-request, 4-RPS Poisson workload. Reports include batch-size histograms in
worker telemetry. Compare throughput and TTFT/TPOT/E2E with the earlier
batch-size-one report. GPU performance is pending measurement; CPU correctness
and real CPU Ray actor checks pass. The overall function timeout remains 600s.


### Independent prefill and decode

The Modal benchmark uses independent scheduling exclusively. Each Ray actor has
at most one outstanding operation; a decoder can continue generating while
other GPUs prefill. KV admission and transfer acknowledgements remain bounded
and explicit. The CPU snapshot transfer path is unchanged.

```bash
PYTHONPATH=src uv run python src/verify_independent.py
modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/a100-200-rps4 \
  --decode-batch-size 4 \
  --repeats 3 --max-run-seconds 0 \
  --output results/stage21-a100-80gb-200-rps4-independent-batch4
```

The execution-policy option has been removed. GPU benchmarks always use
independent scheduling on four A100 80 GB GPUs. Reports identify the scheduler;
periodic worker snapshots include their age and whether an
operation is in flight. They represent the most recently completed operation,
not a synchronous view of every actor. Final worker telemetry verifies cleanup.
CPU Ray correctness checks pass; GPU performance remains to be measured.


### Eight-GPU scaling experiment: four prefill and four decode

The selected 2+2, 5-RPS baseline remains unchanged. The launcher now uses 4+4 A100 40 GB GPUs
at 10 RPS, preserving offered load per replica. Hardware differs from the
selected 80-GB baseline, so this is not an isolated replica-scaling comparison. It keeps 200 requests, so the
arrival window is about 20 seconds rather than the baseline's 40 seconds.
Warm-up increases from four to eight requests. Per-worker KV/holding limits,
decode batch size four, float32, 16 output tokens and three passes are unchanged.

```bash
PYTHONPATH=src uv run python src/verify_independent.py --workers-per-role 4
modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/a100-200-rps10 \
  --decode-batch-size 4 --warmup-requests 8 \
  --repeats 3 --max-run-seconds 0 \
  --output results/stage21-a100-40gb-4p4d-200-rps10-batch4
```

The workload is prepared locally. To regenerate into a new directory, use
`src/freeze_workload.py --num-requests 200 --rate-rps 10 --max-new-tokens 16
--kv-capacity-tokens 2048 --num-prefill-workers 4 --output <new-directory>`
with `PYTHONPATH=src uv run python` before the script path.
The launcher requests 16 CPUs and 128 GiB host memory. Its displayed GPU-only
estimate doubles with eight GPUs; actual charges also include other resources.
CPU checks passed; eight-GPU results are pending.


### Stage 22: prefill work stealing

An idle prefill worker with an empty queue can take an unstarted request from
the source with the most queued prompt tokens, selecting from the donor's tail.
It checks local KV/holding capacity and excludes every request already dispatched.
No held KV or active decode request moves. Reports include actual prefill
assignments, steal events and counts. CPU Ray checks pass; GPU comparison is pending.

```bash
PYTHONPATH=src uv run python src/verify_work_stealing.py
modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/a100-200-rps5 \
  --decode-batch-size 4 --warmup-requests 4 \
  --repeats 3 --max-run-seconds 0 --no-stealing \
  --output results/stage22-a100-80gb-200-rps5-no-stealing
modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/a100-200-rps5 \
  --decode-batch-size 4 --warmup-requests 4 \
  --repeats 3 --max-run-seconds 0 --stealing \
  --output results/stage22-a100-80gb-200-rps5-stealing
```

Compare both reports using identical frozen inputs and settings. A benefit is
not guaranteed: few steals may occur at this load. The earlier selected baseline
remains preserved; the fresh disabled run controls for the Stage 22 code changes.


### Stage 22: compare both modes in one GPU allocation

```bash
modal run src/Modal/run_benchmark.py \
  --compare-stealing --repeats 3 \
  --workload-dir workloads/a100-200-rps5 \
  --decode-batch-size 4 --warmup-requests 4 --max-run-seconds 0 \
  --output results/stage22-a100-80gb-200-rps5-paired
```

This reuses the same four loaded actors for six measured passes, three per mode,
in off/on, on/off, off/on order. Each pass warms up separately and resets metrics.
Results are saved in one `baseline.json`, with separate summaries under
`comparison.no-stealing` and `comparison.stealing`, and mode labels in each run.
The overall Modal timeout remains 600 seconds. CPU verification:
`PYTHONPATH=src uv run python src/verify_paired.py`.
