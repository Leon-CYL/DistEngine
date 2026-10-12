# DistEngine progress

## Stage 0 — Project setup

Status: complete.

- Initialized a Python 3.11 uv project with a Hatchling build backend.
- Created the installable `src/distengine` package and command-line entry point.
- Added PyTorch and Transformers for the upcoming TinyLlama reference/model stages.
- Added a reproducible dependency lockfile and `.gitignore`.
- Documented setup, editable installation, dependency management, and packaging.
- Kept model loading and inference implementation for later stages.

## Stage 1 — TinyLlama reference

Status: complete.

- Implemented `src/hf_reference.py` with prompt, output-length,
  output-path, and revision arguments.
- Loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, applies its chat template, and
  generates greedily using Hugging Face `generate()` with KV caching enabled.
- Resolves branch/tag revisions through Hugging Face Hub before loading; model
  and tokenizer use the same immutable commit.
- Saves input/output token IDs, formatted prompt, generation settings, revision,
  and environment versions as JSON.

Verified saved results on October 6, 2026:

- Commit: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`.
- Environment: macOS, MPS, `torch.float32`, Python 3.11.9, PyTorch 2.14.1,
  Transformers 5.19.0. The implementation uses eager attention.
- Prompt: “What is the capital of France?” — 22 input tokens, 8 generated tokens
  with a maximum of 32; response: “The capital of France is Paris.”
- Saved files: `references/factual.json`, `references/factual_repeat.json`, and
  the default-output copy `references/tinyllama_hf.json`.
- Compared the factual and repeated runs: model commit, environment, formatted
  prompt, generation settings, input token IDs, and generated token IDs all match.
  This verification inspected existing saved results without rerunning inference.

The user's October 8, 2026 verification run also generated HF references for
“Explain why the sky is blue in two sentences.” (32 generated tokens) and
“Complete this sentence: The best way to learn programming is” (21 generated
tokens), using the same pinned revision. These were compared in memory with the
custom model; separate reference JSON files were not saved for these examples.
The README documents the model-running command.

## Stage 2 — Understand TinyLlama

Status: complete.

- Added [TinyLlama architecture notes](docs/tinyllama.md) covering embeddings,
  decoder blocks, RMSNorm, GQA, RoPE, causal attention, SwiGLU, and logits.
- Documented model dimensions, Q/K/V shapes, and planned per-layer KV-cache shapes.

## Stage 3 — Request / sequence

Status: complete.

- Implemented `Request` and `RequestStatus` in `src/distengine/request.py`.
- Tracks request ID, copied prompt tokens, generated tokens, output limit, EOS,
  lifecycle status, and finish reason (`reason`).
- Provides combined token IDs, token counts, and completion state as properties.
- Enforces `WAITING → RUNNING → FINISHED`; finishes on EOS or output limit.
- Integrated the request into `generate_greedy()` while preserving its existing
  tensor input/output interface.

Verification:

- The user ran `distengine.request` successfully and reran `verify`
  after integration; all checks passed, including zero logit differences and
  exact HF token agreement across all three prompts.
- Additional local checks passed for empty prompts, nonpositive output limits,
  prompt copying, independent requests, invalid lifecycle operations, length
  stopping, and EOS priority when EOS reaches the output limit.

## Stage 4 — Model implementation

Status: complete.

- Implemented embeddings, LM head, linear projections, RMSNorm, SwiGLU, RoPE,
  and full-sequence causal grouped-query attention in PyTorch.
- Assembled 22 decoder blocks and the final normalization in `models/llama.py`.
- Added pinned checkpoint loading with strict parameter-name and shape matching.
- Fixed config typing and read the RoPE base from the installed Transformers
  version's `rope_parameters` dictionary.
- Scoped the downloaded-model ignore rule to `/models/`, keeping model source
  files under `src/models/` available for tracking.

The initial user-reported run on October 8, 2026 failed during custom model
construction because `config.rope_theta` was unavailable. After the config fix,
the user's October 8 screenshot confirmed successful model construction, strict
checkpoint loading, and inference for the factual prompt.

Verified from the user's October 8, 2026 terminal output of
`uv run python -m verify`, on MPS with float32 and the pinned checkpoint:

- Logit shapes and finite-value checks passed.
- Causal masking passed: appending a token preserved earlier-position logits
  within the script's numerical tolerance.
- Custom and HF full-sequence logits matched for factual, explanation, and
  sentence-completion prompts; maximum and mean absolute errors were both zero
  for every prompt.

## Stage 5 — Manual generation

Status: complete.

- Added greedy sampling and a manual generation loop without HF `generate()`.
- Added `manual_generate.py` with prompt and maximum-output-length arguments.
- Reuses the pinned tokenizer and chat template; currently supports one unpadded
  request and recomputes the full sequence without a KV cache.
- Documented the run command in the README.

Verified from the user's October 8, 2026 screenshot:

- Prompt: “What is the capital of France?” with `--max-new-tokens 32`.
- Generated token IDs: `[1576, 7483, 310, 3444, 338, 3681, 29889, 2]`.
- All eight token IDs exactly match the saved HF factual reference.
- Response: “The capital of France is Paris.”
- Generation stopped at EOS (`2`) after eight tokens, before the output limit.

The user's October 8, 2026 `verify` run passed all generation checks:

- Exact HF token agreement for factual (8 tokens), explanation (32 tokens), and
  sentence-completion (21 tokens) prompts.
- Repeatability, prompt preservation, output-limit stopping, and factual EOS
  stopping passed.

Additional local validation confirmed `generate_greedy()` raises `ValueError`
for an empty prompt, a batch size of two, a one-dimensional input, and zero or
negative output limits. All stage 4/5 verification checks are complete for the
current single-request, unpadded, full-sequence implementation. KV-cache
correctness remains part of stage 6.

## Stage 6 — Paged KV cache

Status: complete.

- Implemented `PagedKVCache` with fixed-size physical KV blocks and a copied
  logical-to-physical block table for one request.
- Validates block IDs, input shapes, device, dtype, and mapped capacity; supports
  per-layer append, logical-order reads, consistent lengths, and reset/reuse.
- Gathers valid paged K/V into contiguous temporary tensors for existing eager
  attention. Model and cached generation interfaces use the paged cache backend.

Verification:

- Local storage checks passed exact equality with expected token histories, shuffled block
  mappings, boundary crossings, partial blocks, overflow, and reset/reuse.
- Local tiny-model checks passed prefill, decode, chunks, generation agreement,
  length tracking, and cache guards for the paged backend.
- The user's October 10, 2026 `verify` run passed on MPS with float32
  and checkpoint `fe8a4ea1ffedaf415f4da2f062534de366a451e6`.
- Paged generation exactly matched uncached output for all
  three prompts; the outputs also matched HF (8, 32, and 21 generated tokens).
- Paged real-checkpoint cached logits matched the uncached baseline
  within `atol=1e-4`, `rtol=1e-4`. Existing stage 4/5 checks also passed.

Stage 6 scope: one request per cache object with an explicit block table.
Stage 7 adds shared storage, block allocation, freeing, and reuse management;
direct attention over pages is not implemented.

## Stage 7 — Block manager

Status: complete.

- Added `BlockManager` with deterministic allocation, total-capacity growth,
  free/reuse, free-block tracking, copied mappings, and strict argument validation.
- Zero-token requests register empty mappings; unknown requests and repeated
  frees raise `KeyError`. Capacity failures preserve ownership and free blocks.
- Added shared `KVStorage` and request-local `PagedKVCache` views, including
  empty views and prefix-preserving block-table expansion.
- Cache views must be retired before freeing their blocks; new views start with
  zero valid lengths. Ownership enforcement remains the caller's responsibility.
- Added `verify_blocks` and README usage/lifecycle instructions.

Verification on October 10, 2026:

- CPU allocation checks passed boundaries, validation, copies, disjoint ownership,
  growth, atomic exhaustion, and freeing/reuse.
- Shared CPU storage checks passed interleaved request isolation, cache growth,
  invalid mapping rejection, empty-view growth, reset, and stale-data exclusion.
- Existing `verify_cache` tiny-model prefill/decode/generation checks
  and `distengine.paged_kv_cache` storage checks passed.
- The user's full verification run passed on MPS with float32 and checkpoint
  `fe8a4ea1ffedaf415f4da2f062534de366a451e6`. Paged generation and cached
  logits matched the uncached baseline for all three prompts. Custom/HF logits
  had zero maximum and mean error; generated tokens matched exactly (8, 32,
  and 21 tokens). Causal masking, repeatability, output-limit stopping, and
  factual EOS stopping also passed.

## Stage 8 — Model runner

Status: complete.

- Added `ModelRunner` with model device/dtype and shared-storage validation,
  request validation, token/absolute-position preparation, prefill, decode,
  and explicit cache release.
- Executes one request per model call using manager-backed shared KV storage.
  Decode grows block mappings as needed and processes the latest generated token;
  the newly sampled token remains uncached until the next step.
- Added `verify_runner` assertions for the runner lifecycle,
  failure handling, and generation agreement, plus optional `--checkpoint`
  verification against the pinned TinyLlama uncached baseline.

Verified from the user's October 11, 2026 screenshot and terminal output:

- `verify_runner` passed input tensors, positions, validation,
  prefill/decode, block growth, interleaved generation agreement, release,
  block reuse, output-limit/EOS stopping, unfinished cleanup, and repeated release.
- `verify_blocks` and `verify_cache` passed.
- Existing `verify` regressions passed on MPS with float32 and
  checkpoint `fe8a4ea1ffedaf415f4da2f062534de366a451e6`: paged generation,
  cached logits, causal masking, repeatability, output-limit stopping, and
  factual EOS stopping. Custom/HF logits had zero maximum and mean error;
  generated tokens matched exactly for all three prompts (8, 32, and 21 tokens).
  These real-checkpoint checks exercise the existing generation paths, not
  `ModelRunner` itself.

Additional local verification on October 11, 2026:

- Admission/decode exhaustion preserved tokens, cache lengths, mappings, and
  cached values; decode succeeded after another request released capacity.
- Injected prefill/decode failures after first-layer KV writes released the
  affected allocation (including decode growth) and preserved another request
  whose decode subsequently succeeded. All CPU runner checks passed.

- `verify_runner --checkpoint` passed locally on CPU with float32
  and pinned checkpoint `fe8a4ea1ffedaf415f4da2f062534de366a451e6`. Runner
  generation exactly matched the uncached baseline for factual (8 tokens),
  explanation (32 tokens), and sentence-completion (21 tokens) prompts. Cache
  length tracking and full block release passed for each real-checkpoint request.

## Stage 9 — Basic scheduler

Status: complete for sequential, single-active-request FIFO execution.

- Added `Scheduler` with waiting/running/finished queues, lifetime request-ID
  reservation, fresh-runner checks, and explicit admission rejection.
- Admission checks the maximum cached history using ceiling division, including
  the uncached final output token adjustment. Each step performs one prefill or
  decode call and returns the request that advanced; empty steps return `None`.
- Completed requests release cache ownership before moving into finished history.
  Failed requests leave the running queue, release any remaining cache, and
  propagate the error while preserving later waiting requests. Failed requests
  retain their current status and reserved ID and do not enter finished history.
- Added `verify_scheduler`; batching and multiple active requests
  remain outside this stage's scope.

Verified locally on October 11, 2026 with a small CPU model:

- FIFO execution, exactly one model call per step, exact uncached generation
  agreement, output-limit/EOS stopping, block reuse, and empty steps passed.
- Duplicate IDs, oversized requests, invalid lifecycle/tokens, and initialization
  with active caches or zero-block registered requests were rejected.
- Prefill/decode failures before execution and after partial KV writes released
  resources and allowed the subsequent waiting request to finish correctly.
- All CPU `verify_runner` regression checks passed.

## Stage 10 — Engine integration

Status: implemented and verified on CPU/MPS; NVIDIA GPU verification pending.

- Added `LLMEngine` to construct and connect `BlockManager`, shared `KVStorage`,
  `ModelRunner`, and `Scheduler` on the model's existing device and dtype.
- Supports request submission, one-operation stepping, pending-work tracking,
  and running until completion. Each completion call returns only newly finished
  requests; failures propagate while later queued requests remain available.
- Added `verify_engine` for integration, output agreement, stopping, repeated use,
  admission validation, cleanup, and partial-write failure recovery.

Verified from the user's October 11, 2026 terminal output:

- All CPU engine integration checks passed.
- `verify_engine --checkpoint` passed on MPS with float32 and checkpoint
  `fe8a4ea1ffedaf415f4da2f062534de366a451e6`. Engine outputs exactly matched
  the uncached baseline for factual (8 tokens), explanation (32 tokens), and
  sentence-completion (21 tokens) prompts.
- The roadmap's one-NVIDIA-GPU execution target has not yet been verified.

Source layout updated on October 11, 2026:

- `src/distengine/` contains engine components and package entry points.
- `src/models/` contains the model architecture; `src/layers/` contains layers.
- Generation helpers, checkpoint loading, reference/CLI tools, and verification
  modules live directly under `src/`. Imports and wheel packaging were updated.
- Run verification as `uv run python -m verify_engine` (likewise for other
  top-level verification modules).
- After reorganization, all five CPU verification suites passed: engine,
  scheduler, runner, blocks, and cache. The package entry point ran; editable
  installation, wheel/source builds, and wheel contents checks passed.

## Roadmap

| Stage | Focus | Completion checkpoint | Status |
| --- | --- | --- | --- |
| 0 | Project setup | uv project, dependencies, package, README, PROGRESS, and .gitignore. | Complete |
| 1 | TinyLlama reference | Load TinyLlama-1.1B with Hugging Face and generate reference text. | Complete |
| 2 | Understand TinyLlama | Document embeddings, layers, GQA, RoPE, attention, FFN, logits, and KV shapes. | Complete |
| 3 | Request / sequence | Prompt tokens, generated tokens, status, and max output length. | Complete |
| 4 | Model implementation | Minimal TinyLlama model components in PyTorch. | Complete |
| 5 | Manual generation | Greedy generation without HF generate(). | Complete |
| 6 | Paged KV cache | Fixed-size KV blocks and per-request block tables. | Complete |
| 7 | Block manager | Allocation, freeing, reuse, and available-block tracking. | Complete |
| 8 | Model runner | Prepare inputs, positions, and KV information; execute prefill/decode. | Complete |
| 9 | Basic scheduler | FIFO waiting/running/finished queues; no new continuous-batching research. | Complete |
| 10 | nano-vLLM-style engine | Connect LLMEngine, Scheduler, ModelRunner, and BlockManager on one GPU. | CPU/MPS verified; NVIDIA GPU pending |
| 11 | Prefill/decode roles + Ray orchestration | Implement PrefillWorker and DecodeWorker as thin Ray actors; start with a local Ray runtime, actor lifecycle, and request dispatch; keep model execution in ordinary Python classes. | Pending |
| 12 | Two-GPU DistEngine | Place prefill and decode actors on separate GPUs with explicit Ray resource reservations; verify placement and execution, using the stage 13 transfer path for end-to-end output correctness. | Pending |
| 13 | KV transfer | Implement explicit KV and request-metadata transfer between actors, initially GPU → CPU → GPU if necessary; verify end-to-end output correctness across the two GPUs. | Pending |
| 14 | Prefill KV holding pool | Retain ready KV until decode admission; bounded capacity. | Pending |
| 15 | Decode-pull admission | Decode pulls a request and KV only when capacity is available. | Pending |
| 16 | Backpressure | Stop prefill admission when the holding pool is full; queue upstream. | Pending |
| 17 | Multiple prefill workers | Independent prefill workers and request queues. | Pending |
| 18 | Workload generator | Constant and Poisson arrivals; save reproducible request inputs, arrival schedules, and seeds for replay. | Pending |
| 19 | Mixed request workload | Short, medium, and long prompts that imbalance prefill work; freeze a benchmark workload. | Pending |
| 20 | Metrics + baseline benchmark | Instrument TTFT, TPOT, E2E, prefill wait, throughput, goodput, GPU utilization, idle time, queue length, and KV occupancy; save a no-stealing baseline before implementing stealing. | Pending |
| 21 | Prefill work stealing + initial comparison | Steal unstarted whole requests; add an enable/disable switch; replay the frozen workload in both modes with identical configuration. | Pending |
| 22 | Multiple decode workers | Simple placement; keep each request on its assigned decode worker. | Pending |
| 23 | Modal deployment | Start with one prefill GPU and one decode GPU; scale as necessary. | Pending |
| 24 | Final benchmark | Repeat the controlled stealing-disabled/enabled comparison on the final system; analyze performance and correctness against the saved baseline. | Pending |
| 25 | Cleanup + report | Simplify code; document architecture, results, limitations, and RDMA/FlashInfer/TP future work. | Pending |

## Benchmark checkpoints

- Stage 20 establishes the baseline before stealing is implemented, using the
  multiple-prefill-worker system from stage 17 and the workloads from stages
  18–19. Save raw measurements, workload inputs, and system configuration.
- Stage 21 performs the first comparison after implementing stealing. Run both
  disabled and enabled modes on the same code revision and configuration;
  retain the earlier baseline to detect unrelated regressions.
- Stage 24 repeats both modes on the final deployed system. If hardware, worker
  counts, or deployment changes after stage 21, capture a fresh disabled baseline
  on that system before comparing it with enabled stealing.

For each comparison, keep model revision, dtype, worker counts, hardware, queue
and KV limits, request inputs, arrival schedule, and seeds identical. Use the
same warm-up procedure, repeat runs, define goodput thresholds in advance, and
verify matching request outputs. Report latency distributions and variation
across runs alongside throughput and goodput.

## Verification

Passed locally on macOS with Python 3.11.9:

- `uv sync --locked --offline --no-cache`: lockfile and environment agree.
- `uv run --no-sync --no-cache distengine`: package command runs.
- `uv run --no-sync --no-cache python -m distengine`: module command runs.
- Imported `distengine`, `torch`, and `transformers` successfully.
- `uv build --no-cache`: produced wheel and source archive in `dist/`.
- `git check-ignore .venv dist`: environment and build artifacts are ignored.

No model download, inference, or GPU correctness checks were performed at stage 0.
