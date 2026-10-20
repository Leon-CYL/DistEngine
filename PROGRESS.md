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

Status: complete; verified on CPU/MPS and one NVIDIA GPU.

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

Added `src/Modal/run_engine.py` for the single-GPU checkpoint. It requests
one Modal T4, uses locked dependencies and a persistent Hugging Face cache,
and checks float32 CUDA engine generation against the uncached baseline for
the three reference prompts, including resource release and repeated engine
use. Successful runs save GPU/environment metadata and outputs locally to
`results/stage10-modal.json`. See the README for setup and run commands.

Verified from the user's successful Modal run and saved
`results/stage10-modal.json` on October 12, 2026:

- One Tesla T4 on `cuda:0`, float32, Python 3.11.12, PyTorch 2.14.1+cu130,
  Transformers 5.19.0, and the pinned checkpoint above.
- CPU integration checks passed in the remote environment.
- CUDA engine outputs exactly matched uncached generation for factual
  (8 tokens, EOS), explanation (32 tokens, length limit), and sentence-completion
  (21 tokens, EOS) prompts.
- Shared KV storage placement, full block release, empty-engine behavior, and
  repeated engine use passed. The Modal app completed successfully.
- Run: https://modal.com/apps/leonlin0106/main/ap-xEu5D5nlAB4SqwbT2zZaBB

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

## Stage 11 — Worker roles and Ray orchestration

Status: complete; initially verified on local CPU, with worker roles and Ray
orchestration subsequently verified on separate GPUs in stages 12–15.

- Added ordinary `PrefillWorker` and `DecodeWorker` with independent runner/KV
  ownership, copied request snapshots, explicit release, and idempotent close.
- Added thin typed Ray actors that construct a deterministic small model inside
  their own processes. Actors reserve one CPU each; automatic restarts and task
  retries are disabled. Ray is included in the dependency lockfile.
- Added both verification modules to wheel packaging.

Verified locally on October 13, 2026:

- Ordinary worker checks passed prefill/decode baseline agreement, immediate EOS,
  one-token limits, invalid and duplicate active IDs, snapshot isolation, close
  behavior, capacity exhaustion with state-preserving retry, and partial KV-write
  failure cleanup while another request survives and finishes correctly.
- Real Ray actor checks passed separate process IDs, dispatch, baseline agreement,
  remote exception propagation, actor survival, block release, and closing with
  unfinished work. The driver terminated actors and shut down its local runtime.
- CPU engine, runner, scheduler, block-manager, and cache regressions all passed.
- Wheel build passed; inspected the archive to confirm both worker modules and
  both verification modules are included.

Original stage 11 scope: decode used a locally prefetched fixture, and actor
construction used the small CPU model. Stage 12 subsequently verified separate
GPU placement and pinned-checkpoint execution; stages 13–15 verified cross-actor
KV handoff and decode-pull admission on two T4 GPUs, including pinned TinyLlama
baseline agreement and cleanup. See those stages for GPU verification evidence.
Request IDs can be reused after release; only active duplicates are rejected.

## Stage 12 — Separate GPU placement

Status: complete; separate-GPU placement, independent execution, and stage 13
end-to-end handoff verified.

- Extended typed actors with explicit CPU/CUDA device selection and optional
  pinned TinyLlama loading. CPU defaults preserve stage 11 verification.
- CUDA construction requires one Ray GPU reservation and exactly one visible
  GPU. Readiness checks model and KV placement and reports Ray GPU IDs/node ID.
- Added `verify_gpu_workers.py` to compare prefill and locally initialized decode
  with uncached CUDA generation, verify disjoint GPU assignments, release/reuse,
  and closing unfinished requests. Added it to wheel packaging.
- Added `src/Modal/run_two_gpu.py`, requesting two T4 GPUs, four CPUs, and 32 GiB
  of host memory. The default run uses pinned TinyLlama and caches model files;
  `--no-checkpoint` selects the small model. Successful reports are saved locally.
- Local ordinary-worker and real CPU Ray checks passed after the changes;
  invalid-device and missing-GPU-reservation guards and source syntax passed.
- Modal verification passed on October 13, 2026 with pinned TinyLlama in float32
  on two Tesla T4 GPUs. Actors had separate PIDs and disjoint Ray GPU IDs, each
  seeing one `cuda:0`; model parameters and both KV tensors were correctly placed.
- Prefill first tokens and independent decode outputs matched the uncached CUDA
  baseline for all three reference prompts. Full release, reuse across prompts,
  and closing both workers with unfinished requests passed.
- Saved `results/stage12-placement.json`; successful Modal run:
  https://modal.com/apps/leonlin0106/main/ap-dJEYHcQHxL1WRwyCFvftmp
- Wheel build and archive inspection passed for the new verification module.

Stage 13 subsequently verified cross-actor KV transfer and end-to-end output
agreement, completing the remaining two-GPU checkpoint below.

## Stage 13 — KV transfer

Status: complete; CPU, real Ray CPU, and two-GPU TinyLlama handoff verified.

- Added a versioned CPU snapshot carrying valid logical KV, request metadata,
  pending first output token, cache length, and model/config identity.
- Destination import validates compatibility before allocation, constructs its
  own block mapping, copies all layers, and publishes the request only after
  successful completion. Partial-copy failures release destination blocks.
- Source KV remains owned until the driver receives a successful import ACK.
  Requests already finished in prefill skip transfer. GPU imports synchronize
  host-to-device copies before acknowledging success.
- CPU checks passed different block sizes/mappings, exact KV and generation
  agreement, no prompt recomputation, snapshot isolation, malformed/incompatible
  payloads, duplicate active IDs, temporary capacity failure/retry, and rollback
  after a partial-layer copy. Real Ray CPU end-to-end transfer also passed.
- Existing ordinary worker, engine, runner, scheduler, block and cache checks
  passed. Added `verify_transfer.py` to wheel packaging.
- Added `src/Modal/run_transfer.py` for two-T4 pinned TinyLlama handoff, using the
  existing cache Volume and shared GPU verification code. Results are saved only
  after success to `results/stage13-transfer.json`.
- Modal verification passed on October 13, 2026: prefill ran on T4 GPU 0 and
  decode on T4 GPU 1, using float32 and pinned revision
  `fe8a4ea1ffedaf415f4da2f062534de366a451e6`. Imported-KV decode exactly matched
  the uncached CUDA baseline for factual (8 tokens, EOS), explanation (32 tokens,
  length), and sentence-completion (21 tokens, EOS) prompts. Both workers freed
  all blocks between requests and on close with unfinished work.
- Saved the successful report in `results/stage13-transfer.json`. Run:
  https://modal.com/apps/leonlin0106/main/ap-GTagK4GdPMWGC640iOXGh1

Scope: GPU → CPU → GPU snapshots; immutable inference models, explicit retries,
no durable/exactly-once protocol, direct GPU transport, or later holding-pool
admission/backpressure policy.

## Stage 14 — Prefill KV holding pool

Status: complete; CPU lifecycle checks and two-GPU holding/admission handoff
verified.

- Added `PrefillHoldingPool` as a bounded FIFO collection of request IDs with
  capacity validation, duplicate rejection, non-destructive peek, removal,
  length, and copied ID snapshots. Request objects and KV remain owned by the
  prefill worker and runner; the pool does not store extra CPU KV snapshots.
- Integrated the pool into `PrefillWorker`: full pools reject admission before
  model execution, and only successful unfinished prefills enter the pool.
  Immediate EOS and one-token-limit completions do not occupy holding slots.
- Export retains held source KV until explicit release after successful import.
  Release and idempotent close remove held entries and free KV blocks. Failed
  prefill does not leave held entries; failed import preserves source ownership.
- Updated worker and transfer verification constructors and supplied a separate
  holding pool when constructing the Ray prefill worker.
- Added `verify_holding_pool.py` and included it in wheel packaging.

Verified from the user's October 17, 2026 macOS terminal output:

- `PYTHONPATH=src uv run python -m verify_holding_pool` passed pool validation,
  FIFO ordering, bounds, rejection atomicity, copied snapshots, ID reuse,
  immediate completion, partial-prefill failure cleanup, bounded admission,
  handoff retry, generation continuation, and resource cleanup.
- `PYTHONPATH=src uv run python -m verify_workers` passed all ordinary worker
  checks, including baseline agreement, stopping, validation, capacity retry,
  partial-write failure recovery, block release, and close.
- `PYTHONPATH=src uv run python -m verify_transfer` passed local CPU handoff,
  remapping, no prompt recomputation, isolation, validation, retry, rollback,
  and real Ray CPU payload handoff, continuation, and cleanup.

Scope: holding capacity is a request-count limit; physical KV capacity remains
enforced by the block manager. Operations assume serial worker execution.
The stage 15 Modal run verified retained source ownership during denied GPU
admission, successful handoff, and cleanup on two T4 GPUs. Stage 15 adds
decode-pull admission; upstream queuing and coordinated backpressure remain stage 16.

## Stage 15 — Decode-pull admission

Status: complete; CPU, real Ray CPU, and two-GPU pinned TinyLlama admission verified.

- Added non-destructive `peek_ready()` metadata inspection on the prefill worker
  and actor, avoiding KV export before a decode capacity decision.
- Added `DecodeWorker.can_admit()` and its Ray wrapper. Queries validate metadata,
  reject requests exceeding total pool capacity, and return false for temporary
  free-block shortages without reserving or changing state.
- Added opt-in `reserve_full=True` import through the worker and actor layers.
  Import reserves blocks for prompt length plus maximum output length minus one,
  while valid KV length remains the prompt length. Default imports retain their
  previous prompt-only allocation behavior. Partial-copy failure frees the full
  reservation before propagating the error.
- Added serial `pull_one()` and `pull_one_ray()` coordinators: peek the FIFO head,
  check destination capacity, export only when admissible, import with full
  reservation, and release source ownership only after successful import.
- Extended `verify_admission.py` with coordinator checks and updated the real Ray
  transfer check to exercise denied admission, capacity release, retry, and
  reserved handoff. Added explicit Ray object-reference types for editor inference.

Verified from the user's October 17, 2026 macOS terminal output:

- `PYTHONPATH=src uv run python -m verify_admission` passed metadata/FIFO checks,
  validation, capacity rounding, read-only queries, full reservation, stale-query
  rejection, copy-failure rollback, exact KV preservation, fixed block mappings,
  baseline generation agreement, denial without export, retry, and release-failure
  propagation.
- `PYTHONPATH=src uv run python -m verify_transfer` passed local CPU handoff
  regressions and real Ray CPU pull denial, capacity retry, reserved handoff,
  generation continuation, and cleanup.
- Ray reported the filesystem containing its session directory was over 95% full;
  the verification completed successfully despite the warning.

Verified on Modal on October 17, 2026 from the user's successful terminal output
and `results/stage15-admission.json`:

- Both the deterministic small-model run (`--no-checkpoint`) and the default
  pinned TinyLlama run completed successfully on two Tesla T4 GPUs.
- Prefill and decode ran in separate processes on disjoint Ray GPU IDs 0 and 1;
  each actor saw one `cuda:0`, with model and KV storage correctly placed.
- TinyLlama used float32, revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`,
  Python 3.11.12, PyTorch 2.14.1+cu130, and Ray 2.59.0.
- Actual decode-pull orchestration passed capacity denial with retained source
  ownership, retry after capacity release, full destination reservation, and
  unchanged reserved block counts during unfinished decode.
- All three reference prompts exactly matched uncached CUDA generation: factual
  (8 tokens, EOS), explanation (32 tokens, length), and sentence completion
  (21 tokens, EOS). Full block release between requests and close with unfinished
  work passed. CPU admission regression checks also passed in the remote image.
- The saved report contains the pinned TinyLlama run; it replaced the small-model
  report at the same default path.
- Small-model run: https://modal.com/apps/leonlin0106/main/ap-W3LhtoEGLiwKMC8xnYVtUG
- TinyLlama run: https://modal.com/apps/leonlin0106/main/ap-ZTC6Ccpfk98BmrAcrEXpcu

Scope: one serial coordinator per source, FIFO admission, explicit retries, and
immutable models. Source-release failure after successful import is surfaced;
automatic recovery and durable/exactly-once handoff are not implemented. All
production decode admissions must use full reservation to retain the guarantee
that admitted requests can finish without competing for future KV blocks.

## Stage 16 — Backpressure

Status: complete; ordinary CPU, real Ray CPU, and two-GPU pinned TinyLlama
backpressure verified.

- Added `Coordinator` and driver-side `RayCoordinator` with copied immutable
  submissions, upstream FIFO queues, lifetime request-ID reservation, and
  validation against both workers' vocabularies and total KV capacities.
  Coordinators require fresh open workers and exclusive serial ownership.
- Added read-only `can_prefill()` checks and the Ray actor wrapper. Requests wait
  upstream when holding slots or prompt KV blocks are unavailable. Prefill
  removes queued metadata only after successful execution.
- Each tick advances one decode request in round-robin order, attempts one FIFO
  decode pull, then attempts one prefill. Decode progress frees capacity so held
  transfers and upstream admission resume automatically.
- Completed results are retained across later tick failures and drained once.
  Failed decode requests retired by workers are removed from scheduling; errors
  propagate rather than being treated as ordinary capacity pressure.
- Made the Ray prefill holding limit configurable and exposed vocabulary and
  block-size metadata for remote validation.
- Added `verify_backpressure.py`, `verify_ray_backpressure.py`,
  `verify_gpu_backpressure.py`, and `src/Modal/run_backpressure.py`; included the
  verification modules in wheel packaging.

Verified from the user's October 17, 2026 local terminal output:

- `PYTHONPATH=src uv run python -m verify_backpressure` passed upstream waiting,
  continued decode, automatic resumption, baseline agreement, cleanup, completion
  retention after later failure, single completion delivery, immediate finish,
  fresh-worker guards, and round-robin queue rotation.
- `PYTHONPATH=src uv run python -m verify_ray_backpressure` passed real Ray CPU
  upstream waiting, decode progress, automatic resumption, output agreement,
  and full resource release.

Verified on Modal on October 17, 2026 (Pacific time; remote logs use October 18
UTC), from the user's terminal output and `results/stage16-backpressure.json`:

- Small-model and pinned TinyLlama runs both passed on two Tesla T4 GPUs, with
  separate prefill/decode processes and disjoint Ray GPU IDs 0 and 1.
- The actual `RayCoordinator` used a holding limit of one and three queued
  requests per prompt. Tests demonstrated one decoding request, one held request,
  and one upstream waiting request, followed by automatic admission resumption.
- TinyLlama used float32, revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`,
  Python 3.11.12, PyTorch 2.14.1+cu130, and Ray 2.59.0.
- All three requests in each prompt case exactly matched uncached CUDA generation:
  factual (8 tokens each, 23 ticks), explanation (32 tokens each, 95 ticks), and
  sentence completion (21 tokens each, 62 ticks). Queues drained, completions were
  delivered once, and all KV blocks were released between cases.
- Closing workers with unfinished decoding and held work released their KV.
  CPU backpressure checks also passed in the remote image.
- The initial small-model test setup allowed two one-block requests to fit and
  failed its saturation precondition. Increasing its output limit to eight
  ensured all cases force pressure; both subsequent GPU runs passed.
- The saved default report contains the pinned TinyLlama run, replacing the
  earlier small-model report.
- Small-model run: https://modal.com/apps/leonlin0106/main/ap-BySxAGAwM8kCQCee5VXD9L
- TinyLlama run: https://modal.com/apps/leonlin0106/main/ap-KMHAKpDd6jlv42j2dviuuW

Scope: one prefill worker and one decode worker, synchronous serial scheduling,
request-level FIFO admission, and round-robin unbatched decode. The upstream
queue is in memory and unbounded; durable recovery, concurrent coordinators,
multiple prefill workers, and performance claims remain outside this stage.

## Stage 17 — Multiple prefill workers

Status: complete; ordinary CPU scheduling checks, real Ray CPU orchestration,
and three-GPU pinned TinyLlama verification passed.

- Added `MultiPrefillCoordinator` with independent upstream FIFO queues and
  holding pools, fixed round-robin submission placement, global lifetime ID
  reservation, rotating source pull priority, and round-robin decode scheduling.
  A blocked source does not prevent admission from another eligible source.
- Added driver-side `MultiPrefillRayCoordinator`. Eligible prefill calls are
  dispatched across actors before waiting for their results, allowing concurrent
  prefill execution. All dispatched results are reconciled before propagating
  the first prefill failure; successful peers are not left untracked.
- Constructors reject duplicate workers/actors and require fresh open workers.
  Ray readiness metadata now includes actor identity. Completion buffering,
  explicit error propagation, full decode reservations, and source ownership
  until successful import are preserved.
- Added `verify_multi_prefill_ray.py`, `verify_gpu_multi_prefill.py`, and
  `src/Modal/run_multi_prefill.py`; verification modules are included in wheel
  packaging. The Modal entry point runs real Ray CPU checks before GPU checks.

Verification on October 17, 2026 (Pacific time):

- Local ordinary-worker checks passed round-robin placement, global duplicate
  rejection, source failure isolation, baseline output agreement, completion
  delivery, and cleanup. Deferred actor-adapter checks verified dispatch before
  waiting and reconciliation of a successful peer after another source failed.
- The user's `PYTHONPATH=src uv run python -m verify_multi_prefill_ray` run passed
  real Ray CPU placement, duplicate guards, independent holding, admission,
  mixed-prompt output agreement, and cleanup.
- Both Modal commands passed: `uv run modal run src/Modal/run_multi_prefill.py
  --no-checkpoint` and `uv run modal run src/Modal/run_multi_prefill.py`.
- Three Tesla T4 GPUs hosted separate actor processes: two prefill actors and one
  decode actor, with disjoint Ray GPU IDs 0, 1, and 2. Each actor saw one CUDA
  device, with model and KV placement verified.
- Pinned TinyLlama used float32, revision
  `fe8a4ea1ffedaf415f4da2f062534de366a451e6`, Python 3.11.12,
  PyTorch 2.14.1+cu130, and Ray 2.59.0.
- Six queued requests mixed the three reference prompts, with three submissions
  assigned to each source. Tests verified separate held requests, one source
  admitting and refilling while the other remained held, and upstream waiting
  while decode capacity was occupied.
- All six outputs exactly matched uncached CUDA generation: two factual outputs
  of 8 tokens, two explanation outputs of 32 tokens, and two sentence-completion
  outputs of 21 tokens. All completed once within 118 ticks; all queues drained
  and KV blocks were released. Closing all three workers with unfinished held
  and decoding requests also released their KV.
- Saved `results/stage17-multi-prefill.json` contains the pinned TinyLlama run,
  replacing the small-model report at the same default path.
- Small-model run: https://modal.com/apps/leonlin0106/main/ap-OIOV21uAMdMOTQA6YUW1eT
- TinyLlama run: https://modal.com/apps/leonlin0106/main/ap-mQ1mSxh6rQPLxrj1mBs5fy

Scope: fixed source assignment with no stealing, one decode worker, serial driver
ticks with concurrent prefill dispatch, unbatched decode, in-memory queues, and
explicit retries. These are correctness checks, not performance measurements.
Stage 18 adds reproducible arrival workloads; mixed workloads and benchmark
instrumentation are covered in stages 19 and 21, respectively.

## Stage 18 — Workload generator

Status: complete; local CPU verification passed on October 18, 2026, including
the user's `PYTHONPATH=src uv run python src/verify_workload.py` run.

- Added `ScheduledRequest`, `Workload`, and `generate_workload()` in
  `src/distengine/workload.py`. Constant arrivals start at zero with offsets
  `index / rate_rps`; Poisson arrivals accumulate exponential gaps, including
  the first gap, using a local seeded random generator.
- Requests retain immutable prompt tuples, unique IDs within each generated
  workload, output limits, EOS IDs, and relative arrival offsets.
- Added version-1 JSON save/load for generation settings, seeds, exact token
  inputs, and arrival schedules. Loading restores stored arrivals without
  regenerating them. Validation rejects unsupported schemas, duplicate IDs,
  invalid request fields, and nonfinite, negative, or decreasing offsets.
  Equal arrival offsets and empty workloads are supported. Saving validates
  before replacing an existing file.
- Added `replay_workload()` against the ordinary/Ray coordinator interface,
  with a monotonic clock and injectable clock/sleep functions. Replay submits
  all due requests in order, handles idle gaps, and drains pending work.
  Blocking coordinator calls may delay actual arrivals; overdue requests are
  submitted before the next step. Errors propagate without automatic retries
  or closing caller-owned workers.
- Added and packaged `src/verify_workload.py`. Checks passed for constant
  offsets, seeded Poisson gaps, random-state isolation, JSON round trips and
  invalid files, validation before overwrite, empty replay, idle gaps, tied
  and overdue arrivals, and final draining.
- Both arrival modes passed save/load/replay through two ordinary CPU prefill
  workers and one decode worker. Each mode completed six requests exactly once
  with outputs matching uncached greedy generation; queues drained and all
  worker KV blocks were released. Fake-clock tests avoid real arrival delays.

Scope: the stage 18 generator repeats one prompt and output limit. Mixed prompt
lengths and a frozen benchmark workload are added in stage 19. Stage 18 did not
run GPU/Ray replay verification or measure performance. Actual submission-time
instrumentation and benchmark metrics are covered in stage 21.

## Stage 19 — Mixed request workload

Status: complete; frozen TinyLlama-tokenized inputs, CPU correctness checks,
and cached pinned-tokenizer regeneration passed on October 18, 2026.

- Added `src/distengine/mixed_workload.py` with `Prompt` records and
  `generate_mixed_workload()`. Exact class counts use largest-remainder quotas,
  followed by a seeded shuffle and uniform selection within each class.
  Arrival and prompt-selection randomness use independent seeds; changing
  prompt selection preserves arrival offsets. Validation checks prompt IDs,
  weights, vocabulary, EOS, model context, and per-worker KV capacity.
- Added `src/freeze_workload.py` to construct nine prompts across three topics
  and length classes using TinyLlama's pinned chat template and tokenizer.
  Source paragraphs are repeated to reach the target class without truncating
  token sequences. Measured lengths include the chat template:
  short 53–64 tokens, medium 137–160, and long 520–529.
- Froze `workloads/stage19/mixed-v1.json`: 100 Poisson arrivals at an intended
  average 2 requests/second, arrival seed 42, prompt seed 19, exact class counts
  60 short / 30 medium / 10 long, and a fixed 32-token output limit with normal
  EOS stopping. The companion prompt bank saves source text and exact tokens.
- `workloads/stage19/manifest.json` records tokenizer/model revision, chat
  template, settings, per-request prompt identity, class counts, SHA-256 hashes,
  and round-robin prompt-token totals. For two sources the totals are 6,404 and
  6,751 tokens (maximum/minimum ratio about 1.054); this is modest input imbalance,
  not a measured runtime imbalance or evidence of a stealing speedup.
- Pinned model/tokenizer revision is
  `fe8a4ea1ffedaf415f4da2f062534de366a451e6`. The workload is validated against
  a 32,000-token vocabulary, 2,048-token model context, and 1,024 KV token slots
  per worker (for example, 64 blocks of 16). Later benchmarks must configure
  sufficient capacity on every worker; output length can end early at EOS.
- Added and packaged `verify_mixed_workload.py` and the freeze helper. Verification
  passed frozen-file checksums, quotas, seed independence, invalid settings,
  context/KV rejection, and small/empty workload counts. All nine prompt lengths
  passed save/load/replay on two prefill CPU workers and one decode worker,
  with exact greedy baseline outputs and full KV cleanup. This fixture maps
  token IDs into a 64-token small-model vocabulary and uses four output tokens;
  it does not claim TinyLlama output verification for the frozen 100 requests.
- Cached pinned-tokenizer retokenization reproduced every bank entry, and
  rebuilding the artifacts produced byte-identical files. The stage 18 CPU
  suite remains the regression check for ordinary workload generation/replay.

Run `PYTHONPATH=src uv run python src/verify_mixed_workload.py`; add `--tokenizer`
for cached tokenizer and byte-rebuild checks. The freeze CLI refuses to overwrite
a nonempty directory. Frozen inputs should remain unchanged across comparisons.
Stage 21 adds benchmark instrumentation, warm-up, and GPU baseline measurements
at the frozen offered rate. No stage 19 GPU performance run was performed.

## Stage 20 — Multiple decode workers

Status: complete; local ordinary/Ray CPU checks and the four-GPU pinned
TinyLlama checkpoint passed on October 18, 2026 (Pacific time).
The roadmap is numbered in execution order: 19 → 20 → 21 → 22 → 23 → 24 → 25.
Existing reports retain their original filenames and stage IDs;
`results/stage22-multi-decode.json` records the checkpoint now documented as stage 20.

- Added `MultiDecodeCoordinator` and `MultiDecodeRayCoordinator` for multiple
  prefill sources and decode targets. Prefill submission stays round-robin.
  Rotating target/source pull priority chooses an eligible decoder at KV
  admission; `decode_assignments` retains that request's fixed decode owner.
  Requests finishing in prefill have no decode assignment.
- Each target has an independent round-robin active queue and advances at most
  one request per tick. Each target can admit at most one held request per tick.
  A full target does not prevent another target from pulling; holding pools
  and upstream queues retain the existing backpressure behavior.
- Ray dispatches eligible decode calls across targets before waiting for their
  results, and reconciles every dispatched result before propagating the first
  error. Successful completions remain buffered if a peer or later phase fails.
  Existing concurrent prefill dispatch is reused.
- Successful imports are recorded before source release. A release failure
  retains explicit pending-release state, preventing a duplicate import; a
  subsequent caller tick reconciles source release. Import failures retain
  source ownership. Unreachable actors and unknown remote outcomes still need
  caller intervention; no durable recovery or exactly-once transport is claimed.
- Constructors require distinct, fresh, open workers/actors with correct roles.
  Ordinary workers must have independent KV storage. Submission validation
  checks the assigned source and every decode target before mutating queues;
  requests must fit every target individually. Compatible immutable models
  remain a requirement of KV transfer.
- The original single-decode coordinators remain supported. Only their target
  validation hook is extended so the new coordinators can validate all targets.
- Added `verify_multi_decode.py`, `verify_multi_decode_ray.py`,
  `verify_gpu_multi_decode.py`, and `src/Modal/run_multi_decode.py`; standalone
  verification modules are included in wheel packaging.
- Local CPU checks passed guards, placement, full-target bypass, import retry,
  tracking after source-release failure, decode-peer failure reconciliation,
  per-target rotation, workload replay, output agreement and unfinished cleanup.
  Deferred Ray checks verify dispatch before waiting and successful-peer
  reconciliation. Real Ray checks passed four-process isolation, ownership on
  every tick, both decode actors progressing, pressure, exact outputs and cleanup.
- Stage 18 workload, stage 19 mixed workload, and CPU backpressure regressions
  passed. The four-T4 checkpoint runs ordinary/Ray CPU checks before testing
  pinned TinyLlama GPU placement, transfers, outputs, ownership and cleanup.
- `modal run src/Modal/run_multi_decode.py` passed on four Tesla T4 GPUs in
  separate actor processes: two prefill actors and two decode actors with
  disjoint Ray GPU assignments. Each actor saw one CUDA GPU, with model and KV
  on `cuda:0`. Python 3.11.12, PyTorch 2.14.1+cu130, and Ray 2.59.0 were used.
- Pinned TinyLlama float32 revision
  `fe8a4ea1ffedaf415f4da2f062534de366a451e6` produced all eight outputs exactly
  matching uncached CUDA generation in 94 ticks: three factual outputs of
  8 tokens, three explanation outputs of 32 tokens, and two completion outputs
  of 21 tokens. Both decoders progressed, ownership was checked on every tick,
  and queues drained. Closing all four actors with unfinished held and decode
  requests released all KV blocks.
- Report: `results/stage22-multi-decode.json`.
  Modal run: https://modal.com/apps/leonlin0106/main/ap-NbBOXUQlY4Snh7KF5bH8w9
  Remote logs use October 19 UTC; the run completed October 18 Pacific time.

Scope: one request per decode worker per tick, no batching, migration or stealing,
driver-controlled admission, and in-memory state. Performance instrumentation
and a baseline on this expanded worker topology remain stage 21.

## Stage 21 — Metrics + baseline benchmark

Status: complete. Metrics and validation are finished, and the selected baseline
is 200 Poisson requests at 5 RPS, three passes, independent prefill/decode
scheduling and decode batch size four on two prefill plus two decode A100 80 GB
GPUs. All 600 outputs matched exactly, with zero failures or unfinished requests
and full KV cleanup. The tables below record performance and SLO attainment.

- Extended `src/distengine/metric.py` with shared clock/origin support, raw
  request events, first-prefill-dispatch timing, token observations, failures,
  submission delay, TTFT, TPOT, E2E, prefill wait, percentile summaries,
  completed-request/output-token throughput, and latency-threshold goodput.
  Single-token outputs have no TPOT; their goodput uses TTFT and E2E only.
- Replay shares the coordinator's recorder and monotonic clock, records planned
  versus actual submission, rejects mismatched/nonfresh recorders, and freezes
  measurement duration after draining. Later report/export delays do not change
  throughput. Direct coordinator submissions are also supported, using the
  submission instant as their scheduled time.
- Instrumented ordinary/Ray multi-prefill and multi-decode coordinators. Prefill
  observes the first generated token even while KV is held. Every successful
  decode observes one additional token; import observes none. First dispatch
  time survives explicit prefill retries. Definitively retired decode requests
  are marked failed; recoverable transfer errors do not retire a request.
  Replay records coordinator exceptions separately as orchestration errors.
- Ray prefill and decode dispatch all eligible actor calls before waiting and
  consume results in readiness order with `ray.wait()`. Ordinary decode records
  a result before executing the next worker. Successful peer results are still
  reconciled if another worker fails.
- `verify_metrics.py` passed exact timing/rate arithmetic, lifecycle validation,
  one-token TPOT, frozen report duration, clock/recorder guards, metrics-on/off
  output agreement, handoff counting, and failure classification. Deferred Ray
  checks verify readiness order and timestamps after a peer failure. Real
  `verify_multi_decode_ray.py --metrics` passed four-process output, per-token
  event, completion, ownership and cleanup checks.
- Workload, mixed-workload, multi-decode, and CPU backpressure regression checks
  passed.
- Added `telemetry.py`: optional synchronized worker operation timing with
  nested calls counted once, per-worker busy/idle windows, tick-boundary queue
  and KV-block occupancy samples, and independent background `nvidia-smi`
  GPU utilization/memory polling. GPU polling failures are explicit; CPU runs
  record GPU sampling as unavailable. Worker idle is not GPU idle.
- Added `benchmark_stage21.py`, `benchmark_io.py`, and
  `src/Modal/run_benchmark.py`. The benchmark uses two prefill and two decode
  workers, 64 blocks of 16 tokens per worker, holding capacity two, and no
  stealing. Every repeat warms up four longest-prompt requests before starting
  fresh measurements. Model loading, reference generation, warm-up and output
  validation are outside the request measurement window.
- The GPU runner replays the frozen stage 19 workload unchanged for three
  repeats by default. It saves the input files and checksums, source/dependency
  content hashes, model revision/dtype, actor placement, thresholds, raw timings,
  state/GPU samples, activity totals, assignments, outputs, and per-run and
  across-run summaries. Every output must match uncached CUDA generation and
  every worker must release KV before a run is accepted. Output directories
  must be new, preserving earlier reports.
- Default goodput thresholds are declared before execution: TTFT ≤ 2 seconds,
  TPOT ≤ 0.1 seconds, E2E ≤ 10 seconds. These are initial experimental thresholds,
  not measured hardware capacity. CLI overrides are saved in the report and
  must remain fixed across comparisons.
- `verify_system_metrics.py` passed deterministic nested/failure activity timing,
  GPU parsing/polling/failure handling, sampling interval gates, occupied and
  drained KV snapshots, repeated warm-up/replay, output agreement and JSON export.
  Three CPU smoke repeats also passed and saved
  `results/stage21-cpu-smoke/baseline.json`. This synthetic six-request workload
  is a smoke check, not the frozen TinyLlama GPU baseline.

Timing scope is driver-observed. Prefill wait ends at first dispatch and includes
no worker-side execution-start timestamp; Ray scheduling/transport contribute
to observed token latency. State snapshots are taken between driver ticks and
can miss transient peaks. Worker windows are local to each worker's reset/stop
RPC, so they differ slightly from the driver's request window. Sampling/RPC and
explicit CUDA synchronization overhead is included in the measured run. Actual
submission delay is reported separately from latency measured since submission.
KV occupancy counts allocated/reserved blocks, including full decode reservations,
rather than only token positions already written.

T4 validation measurements on October 18, 2026 (retained for correctness history):

- `modal run src/Modal/run_benchmark.py` completed with stealing disabled on two
  prefill and two decode Tesla T4 actors. The local Modal CLI uses Python 3.9;
  the launcher uses standard-library-only file handling locally and executes
  the model/benchmark under Python 3.11.12 remotely.
- All 100 frozen requests completed in each of three repeats with exact uncached
  CUDA output agreement, no failed/unfinished requests, and full KV release.
  Measured durations were 103.554, 100.257, and 101.318 seconds.
- Request throughput mean was 0.9834 requests/second (sample standard deviation
  0.0162); completed output throughput was 31.4678 tokens/second (standard
  deviation 0.5179). Goodput was zero in all repeats under the predefined
  TTFT/TPOT/E2E limits. Passing verification means correct measurements and
  outputs; it does not mean the latency targets were met.
- Each run collected 303–312 GPU samples covering all four devices with no
  polling errors, and 309–317 state snapshots including final drained state.
  Worker busy/idle windows were valid. The saved source/dependency fingerprint
  was checked against the local implementation after the run.
- Saved `results/stage21-baseline/baseline.json` and per-run `run-00.json`,
  `run-01.json`, and `run-02.json` include all raw data and replay inputs.
  Modal run: https://modal.com/apps/leonlin0106/main/ap-ooBDCSJJMGF94aQG3Gl9ic
  Remote logs use October 19 UTC; execution completed October 18 Pacific time.

Previous large A100 profile prepared on October 19, 2026 (superseded):

- Performance measurements now target four explicitly requested A100 80 GB GPUs
  (`A100-80GB:4`), two prefill and two decode model replicas. T4 helpers remain
  validation tools. The container requests 16 CPU cores, 96 GiB host memory and
  a two-hour timeout. GPU family and at least 75 GiB of physical GPU memory are
  checked before measurement; actual GPU names/memory are recorded.
- Frozen `workloads/a100-rps4/` contains 1,000 requests: 600 short, 300 medium,
  100 long, Poisson arrivals at 4 RPS, and the original independent seeds.
  The last scheduled arrival is at 263.764 seconds. Output limit stays at 32,
  with normal EOS, float32 and the pinned model/tokenizer unchanged.
- Each worker has 512 blocks of 16 tokens (8,192 aggregate KV token slots),
  prefill holding capacity is eight, warm-up is 16 longest-prompt requests per
  repeat, and three measured repeats remain the default. More KV permits more
  in-flight requests; it does not add batching or increase the model context.
- KV block count, holding capacity, warm-up count and input directory are CLI
  settings and saved in the report. Freeze CLI exposes request count, rate,
  output limit and capacity. The original 100-request fixture remains unchanged.
- CPU configurable-capacity tests, legacy byte-identical tokenizer rebuild,
  new input checksums, exact quotas, deterministic generation and rate/selection
  independence passed. Execution was left to the user, who later reported
  excessive runtime/cost; no completed local report is available.
- The large input files are preserved, but this profile is no longer the launch
  default. Goodput limits stay fixed at 2 / 0.1 / 10 seconds for TTFT / TPOT / E2E.
- Execution remains phase-based: sources run concurrently during prefill, and
  targets run concurrently during decode, with no overlap between phases or
  per-GPU batching. Reports identify this policy. An asynchronous pipeline is a
  separate future change requiring a fresh baseline. This larger profile also
  changes load and capacity, so it is not a pure hardware-only comparison.


Cost-bounded A100 redesign on October 19, 2026:

- The user reported a large-profile pass taking too long and costing over $5.
  No local completed A100 report was available to infer its throughput; the
  previous request count and repeat count were too aggressive for calibration.
- Keep four A100 80 GB replicas (two per role), but default to 60 requests at
  2 RPS, 16 output tokens, one measured repeat, four warm-up requests, 128 KV
  blocks per worker, holding capacity four and 0.5-second system sampling.
  The frozen short profile has 36/18/6 prompt-class counts and its last arrival
  is at 24.503 seconds. Original large inputs/results remain intact.
- The launcher now requests 8 CPU cores/64 GiB RAM and a 600-second function
  timeout, replacing the two-hour timeout. A 90-second soft measured deadline
  returns partial outputs/metrics, suppresses subsequent repeats, and closes
  unfinished workers. Deadline checks occur between calls and can overrun an
  in-flight operation; hard timeouts/setup errors can prevent report export.
- Reports distinguish calibration from full baselines, record completion,
  not-submitted/unfinished counts and stop reason, and verify post-close KV
  cleanup. Completed-only latency distributions in partial reports are censored.
- CLI measured limits must be positive and at most 180 seconds per repeat, and
  repeats × limit cannot exceed 240 seconds. GPU-only pricing is printed before
  remote invocation: four A100 80 GB GPUs at the published $0.000694/GPU/s rate
  are about $0.17/minute, or $1.67 for 600 seconds. CPU, memory, startup and other
  billing are additional; these settings do not guarantee an exact dollar cap.
- CPU tests passed soft-deadline behavior, bounded idle sleeps, saved partial
  reports, repeat suppression and post-close cleanup. The short frozen inputs
  passed checksums, class counts and output-limit checks. No additional GPU
  job was launched for this redesign.
- Run `modal run src/Modal/run_benchmark.py --output results/stage21-a100-calibration`.
  Inspect this single short run before selecting a rate sweep or repeats. Full
  asynchronous phase overlap and per-GPU batching remain separate improvements.

Current bounded trial requested on October 19, 2026:

- Prepared `workloads/a100-200-rps4/`: 200 requests at 4 RPS, with exact
  120/60/20 prompt-class counts, a 16-token output limit, and last scheduled
  arrival at 50.012 seconds. Hashes, quotas and output limits were checked.
- Launcher defaults now select this profile and three passes with a 90-second
  measured limit per pass. The measured-budget guard is raised from 240 to
  270 seconds. Four A100 80 GB replicas, four warm-up requests, 128 KV blocks,
  holding capacity four, and the 600-second function timeout are unchanged.
- A mocked launcher check accepted 3 × 90 seconds and rejected 4 × 90 seconds
  before remote invocation. No GPU job was launched while preparing this trial.
- Run `modal run src/Modal/run_benchmark.py --repeats 3 --max-run-seconds 90
  --output results/stage21-a100-200-rps4` (one command). Partial runs still save
  results and suppress subsequent passes; completing all three passes is not
  guaranteed by the soft time limits. Setup and warm-up are outside measured
  limits and remain inside the function timeout.

### Selected baseline: independent scheduling, batch four, 200 requests at 5 RPS

Source: `results/stage21-a100-80gb-200-rps5-independent-batch4/baseline.json`.
[Modal run](https://modal.com/apps/leonlin0106/main/ap-Oh8umIqRVI8aEitIWHbGZW).

Configuration: four A100-SXM4 80 GB GPUs (two prefill, two decode), independent
prefill/decode execution, decode batches up to four, 200 frozen Poisson requests
at 5 RPS per pass, float32, 16 output tokens maximum, three passes, four warm-up
requests per pass, 128 blocks × 16 tokens per worker and holding capacity four.
No per-pass deadline; the overall Modal function timeout remains 600 seconds.

| Metric | Pass 1 | Pass 2 | Pass 3 |
| --- | ---: | ---: | ---: |
| Measured duration (s) | 41.255 | 41.234 | 41.251 |
| Completed requests | 200 | 200 | 200 |
| Failed requests | 0 | 0 | 0 |
| Unfinished requests | 0 | 0 | 0 |
| Request throughput (req/s) | 4.848 | 4.850 | 4.848 |
| Output throughput (tokens/s) | 77.57 | 77.61 | 77.57 |
| Goodput (req/s) | 2.569 | 2.619 | 2.279 |
| Median Submission delay (s) | 0.001 | 0.001 | 0.001 |
| P95 Submission delay (s) | 0.002 | 0.002 | 0.002 |
| Median Prefill queue wait (s) | 0.000162 | 0.000171 | 0.000159 |
| P95 Prefill queue wait (s) | 0.367 | 0.373 | 0.367 |
| Median TTFT (s) | 0.146 | 0.147 | 0.145 |
| P95 TTFT (s) | 0.505 | 0.511 | 0.502 |
| Median TPOT (s) | 0.097 | 0.095 | 0.104 |
| P95 TPOT (s) | 0.228 | 0.222 | 0.233 |
| Median End-to-end latency (s) | 1.644 | 1.616 | 1.728 |
| P95 End-to-end latency (s) | 3.687 | 3.722 | 3.743 |

| Aggregate across passes | Mean | Sample standard deviation |
| --- | ---: | ---: |
| Request throughput (req/s) | 4.8489 | 0.0013 |
| Output throughput (tokens/s) | 77.5823 | 0.0210 |
| Goodput (req/s) | 2.4891 | 0.1839 |

GPU usage across all three measured passes (sample-weighted utilization means):

| GPU | Role | Mean GPU utilization | Peak GPU utilization | GPU memory used (GiB) | Mean worker busy fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Prefill | 12.5% | 43% | 5.40 | 33.2% |
| 1 | Prefill | 14.3% | 44% | 4.89 | 36.3% |
| 2 | Decode | 25.4% | 38% | 4.73–4.76 | 85.0% |
| 3 | Decode | 25.7% | 40% | 4.73–4.76 | 86.8% |

GPU memory samples include model weights and allocator memory. GPU utilization
is sampled by `nvidia-smi`.
Worker busy fraction measures wall time inside worker operations, including
CPU work, transfers and synchronization; it is not GPU compute utilization.

| Latency requirement | Pass 1 | Pass 2 | Pass 3 | Total attainment |
| --- | ---: | ---: | ---: | ---: |
| TTFT ≤ 2 s | 200/200 | 200/200 | 200/200 | 600/600 (100.0%) |
| TPOT ≤ 100 ms | 106/200 | 108/200 | 94/200 | 308/600 (51.3%) |
| E2E ≤ 10 s | 200/200 | 200/200 | 200/200 | 600/600 (100.0%) |
| All three limits | 106/200 | 108/200 | 94/200 | 308/600 (51.3%) |

All 600 outputs exactly matched uncached CUDA generation. No requests failed
or remained unfinished, and all workers closed with zero allocated KV blocks.
Latencies are driver-observed from actual submission; TPOT includes scheduling
and transfer waits. Measured durations exclude setup, reference generation,
warm-up and export.

Selected baseline: 5-RPS Poisson arrivals with independent scheduling and
decode batch size four. Use the frozen `workloads/a100-200-rps5` inputs for
future controlled comparisons. Preserve the model, precision, GPU allocation,
capacities, output limits and latency requirements when evaluating changes.
Only the documented experimental variable should change.

The last request arrives at 40.010 s; mean full-drain duration is 41.247 s.
Mean throughput is 4.849 requests/s and 77.582 output tokens/s. This finite
workload does not establish maximum sustainable capacity. All requests meet
TTFT and E2E limits, but only 308/600 (51.3%) meet the 100-ms TPOT limit.
This is the chosen engineering baseline, not a 90%-SLO operating point.

The 6-RPS and 8-RPS trials increase throughput but reduce joint SLO attainment
and goodput. Keep their reports as load-sweep evidence. Low GPU utilization
with roughly 85–87% decode-worker busy time warrants profiling decode and
KV-import overhead.


## Stage 22 — Prefill work stealing + initial comparison

Status: complete. Work stealing, the enable/disable switch, CPU Ray verification
and the initial GPU comparison are finished. At the user's request, the earlier
separate-allocation comparison remains the main Stage 22 result. Its allocation-
variation caveat remains: it does not isolate a causal speedup from stealing.
The completed paired run is retained as supplementary evidence, not a pending
checkpoint. Stage 21's selected baseline is preserved.

- Restored the active launcher to `A100-80GB:4`: two prefill plus two decode
  replicas, 8 CPUs and 64 GiB host memory. Uses `workloads/a100-200-rps5`, 200
  Poisson requests, three passes, decode batch four, four warm-up requests,
  128 KV blocks × 16 tokens per worker and holding capacity four. Per-pass
  deadlines are disabled; overall Modal timeout remains 600 seconds.
- Added `stealing_enabled` to the independent Ray coordinator and benchmark API.
  Modal exposes `--stealing` / `--no-stealing`; disabled remains the default.
  Output directories default to separate mode-specific names.
- A source can steal only when it has no outstanding operation and its own
  waiting queue is empty. Donors are ranked by queued, unstarted prompt tokens;
  the receiver takes a compatible request from the selected donor's tail.
  The donor's in-flight prefill head is explicitly excluded. Receiver holding
  space, free KV blocks, total request capacity and vocabulary are checked.
- Stealing moves only the immutable, unstarted submission. It creates no KV
  transfer, resubmission or duplicate metric events. Prefill assignment is
  recorded at execution; held KV and active decode requests never migrate,
  and decode ownership remains fixed. Existing pull admission, acknowledgement
  and failure reconciliation are unchanged.
- Each measured run records `prefill_assignments`, `steal_count`, and
  `steal_events` (request id, donor, receiver, prompt tokens, observation time).
  Warm-up uses a separate coordinator, so its steals do not enter measured logs.
- `verify_work_stealing.py` passed policy guards plus real CPU Ray checks with
  stealing disabled/enabled: the enabled peer completes queued donor work while
  the donor's active prefill is deliberately gated; exact outputs, seeded replay,
  one observation per output token, fixed decode ownership and full KV cleanup
  agree across modes. Batch, metrics and system-report regressions passed.
- Verification: `PYTHONPATH=src uv run python src/verify_work_stealing.py`.
- Run fresh disabled and enabled measurements on the same frozen inputs and
  settings. Separate allocations can introduce hardware/runtime variation;
  compare per-pass distributions as well as aggregate means. A small steal
  count at 5 RPS is possible; no performance gain is assumed. No GPU job was
  launched during implementation.

```bash
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


### Stage 22 measured comparison: separate GPU allocations

Sources:
- `results/stage22-a100-80gb-200-rps5-no-stealing/baseline.json`
- `results/stage22-a100-80gb-200-rps5-stealing/baseline.json`

Both reports have identical source hashes, frozen workload and configuration:
two prefill plus two decode A100-SXM4 80 GB GPUs, 200 Poisson requests at 5 RPS,
three passes, float32, decode batch four, four warm-up requests, 128 KV blocks
and holding capacity four. They were collected in separate allocations.

| Metric | No stealing | Stealing |
| --- | ---: | ---: |
| Mean full-drain duration (s) | 41.712 | 40.865 |
| Request throughput (req/s) | 4.795 | 4.894 |
| Output throughput (tokens/s) | 76.716 | 78.306 |
| Goodput (req/s) | 0.439 | 4.331 |
| Median TTFT (s) | 0.165 | 0.068 |
| P95 TTFT (s) | 0.609 | 0.345 |
| Median TPOT (s) | 0.253 | 0.057 |
| P95 TPOT (s) | 0.353 | 0.134 |
| Median End-to-end latency (s) | 4.012 | 0.961 |
| P95 End-to-end latency (s) | 5.620 | 2.195 |
| Median Prefill queue wait (s) | 0.000 | 0.000 |
| P95 Prefill queue wait (s) | 0.439 | 0.088 |
| Exact output matches | 600/600 | 600/600 |
| Failed / unfinished requests | 0 / 0 | 0 / 0 |
| All workers closed with KV released | Yes | Yes |

Latencies are means of per-pass percentiles; rates are means across passes.

| Pass | No-stealing duration (s) | Stealing duration (s) | Steals | No-stealing joint SLO passes | Stealing joint SLO passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.661 | 40.870 | 88 | 17/200 | 176/200 |
| 2 | 41.658 | 40.861 | 91 | 15/200 | 178/200 |
| 3 | 41.818 | 40.865 | 85 | 23/200 | 177/200 |
| Total | — | — | 264 | 55/600 (9.2%) | 531/600 (88.5%) |

Joint SLO limits: TTFT ≤ 2 s, TPOT ≤ 100 ms and E2E ≤ 10 s. All 600 requests
in each mode meet TTFT and E2E; TPOT determines joint attainment.

| GPU / role | No-stealing mean GPU utilization | Stealing mean GPU utilization |
| --- | ---: | ---: |
| GPU 0 / Prefill | 13.3% | 16.0% |
| GPU 1 / Prefill | 15.3% | 8.8% |
| GPU 2 / Decode | 22.5% | 28.8% |
| GPU 3 / Decode | 22.9% | 27.6% |

GPU utilization is sampled by nvidia-smi, not worker wall-time busy fraction.

| Mean worker operation time | No stealing | Stealing |
| --- | ---: | ---: |
| Prefill | 155.22 ms | 103.76 ms |
| KV import | 123.29 ms | 79.62 ms |
| Decode batch | 58.75 ms | 33.20 ms |

Interpretation: stealing moved 264 unstarted submissions across three passes.
Throughput increased about 2.1%, while median TTFT/TPOT/E2E and joint SLO
attainment improved substantially. Outputs match across modes and every pass
drained cleanly. However, per-operation execution times also differ strongly
between allocations. These measurements demonstrate working stealing and an
observed performance difference; they do not isolate the causal speedup from
stealing. The new same-allocation comparison below controls physical allocation
and actor lifetime. The enabled result reaches 88.5% joint attainment, below 90%.

### Same-allocation comparison prepared

- Added `--compare-stealing`. One remote invocation loads the model/actors once
  and runs three passes per mode on the same four GPUs. Order for three pairs:
  off, on, on, off, off, on. Each pass gets fresh warm-up, scheduling queues and
  metrics; all workers must drain and release KV before the next pass.
- `--repeats 3` means three passes per mode (six measured passes total). The
  overall 600-second timeout remains unchanged; no GPU trial was launched here.
- Combined `baseline.json` contains `comparison_order`, per-run mode labels,
  and separate `comparison.no-stealing` / `comparison.stealing` aggregates.
  Root aggregate fields are empty in paired mode to avoid mixing both modes.
- CPU Ray verification passed six alternating passes with the same actor PIDs,
  equal outputs, separate per-mode summaries and complete KV cleanup. Existing
  system-report regression checks passed.

```bash
PYTHONPATH=src uv run python src/verify_paired.py
modal run src/Modal/run_benchmark.py \
  --compare-stealing --repeats 3 \
  --workload-dir workloads/a100-200-rps5 \
  --decode-batch-size 4 --warmup-requests 4 --max-run-seconds 0 \
  --output results/stage22-a100-80gb-200-rps5-paired
```

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
| 10 | nano-vLLM-style engine | Connect LLMEngine, Scheduler, ModelRunner, and BlockManager on one GPU. | Complete |
| 11 | Prefill/decode roles + Ray orchestration | Implement PrefillWorker and DecodeWorker as thin Ray actors; start with a local Ray runtime, actor lifecycle, and request dispatch; keep model execution in ordinary Python classes. | Complete |
| 12 | Two-GPU DistEngine | Place prefill and decode actors on separate GPUs with explicit Ray resource reservations; verify placement and execution, using the stage 13 transfer path for end-to-end output correctness. | Complete |
| 13 | KV transfer | Implement explicit KV and request-metadata transfer between actors, initially GPU → CPU → GPU if necessary; verify end-to-end output correctness across the two GPUs. | Complete |
| 14 | Prefill KV holding pool | Retain ready KV until decode admission; bounded capacity. | Complete |
| 15 | Decode-pull admission | Decode pulls a request and KV only when capacity is available. | Complete |
| 16 | Backpressure | Stop prefill admission when the holding pool is full; queue upstream. | Complete |
| 17 | Multiple prefill workers | Independent prefill workers and request queues. | Complete |
| 18 | Workload generator | Constant and Poisson arrivals; save reproducible request inputs, arrival schedules, and seeds for replay. | Complete |
| 19 | Mixed request workload | Short, medium, and long prompts that imbalance prefill work; freeze a benchmark workload. | Complete |
| 20 | Multiple decode workers | Simple placement; keep each request on its assigned decode worker. | Complete |
| 21 | Metrics + baseline benchmark | Instrument TTFT, TPOT, E2E, prefill wait, throughput, goodput, GPU utilization, idle time, queue length, and KV occupancy; save a no-stealing baseline before implementing stealing. | Complete |
| 22 | Prefill work stealing + initial comparison | Steal unstarted whole requests; add an enable/disable switch; replay the frozen workload in both modes with identical configuration. | Complete |
| 23 | Modal deployment | Start with one prefill GPU and one decode GPU; scale as necessary. | Pending |
| 24 | Final benchmark | Repeat the controlled stealing-disabled/enabled comparison on the final system; analyze performance and correctness against the saved baseline. | Pending |
| 25 | Cleanup + report | Simplify code; document architecture, results, limitations, and RDMA/FlashInfer/TP future work. | Pending |

## GPU correctness checkpoints

- Stages 14–15 two-GPU checkpoint completed on October 17, 2026 using
  `uv run modal run src/Modal/run_admission.py`. Actual pull orchestration,
  pinned TinyLlama output agreement, capacity denial/retry, full reservation,
  source retention, and cleanup passed; report: `results/stage15-admission.json`.
- Stage 16 two-GPU checkpoint completed on October 17, 2026 using
  `uv run modal run src/Modal/run_backpressure.py`: holding-pool saturation,
  upstream waiting, continued decode, automatic resumption, baseline output
  agreement, and cleanup passed. Report: `results/stage16-backpressure.json`.
- Stage 17 three-GPU checkpoint completed on October 17, 2026 using
  `uv run modal run src/Modal/run_multi_prefill.py`: multiple prefill actors,
  independent queues and holding pools, mixed-prompt output agreement, and
  cleanup passed. Report: `results/stage17-multi-prefill.json`.
- Stage 20 four-GPU checkpoint completed on October 18, 2026, before stage 21,
  using `modal run src/Modal/run_multi_decode.py`: two prefill/two decode actors,
  fixed ownership, all eight CUDA baseline outputs, pressure and cleanup passed.
  Report: `results/stage22-multi-decode.json`.
- Stage 21 four-GPU baseline completed on October 18, 2026 using
  `modal run src/Modal/run_benchmark.py`: three repeats of 100 frozen requests,
  exact CUDA output agreement, request/system/GPU telemetry and full cleanup.
  Report: `results/stage21-baseline/baseline.json`.
- Repeat GPU correctness checks after scheduling changes in stage 22 and
  decode-worker expansion in stage 20, and before the final stage 24 benchmark.

## Benchmark checkpoints

- Stage 20 is now executed before stage 21 so the first benchmark already uses
  multiple decode workers. Stage 21 establishes the baseline before stealing is
  implemented, using stages 17 and 20 worker orchestration and the frozen inputs
  from stages 18–19. Save raw measurements, workload inputs, and configuration.
- Stage 22 performs the first comparison after implementing stealing. Run both
  disabled and enabled modes on the same code revision and configuration;
  retain the earlier baseline to detect unrelated regressions.
- Stage 24 repeats both modes on the final deployed system. If hardware, worker
  counts, or deployment changes after stage 22, capture a fresh disabled baseline
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

## Lower-load A100 comparison prepared

- Prepared `workloads/a100-200-rps2/` on October 19, 2026 for one measured pass
  with a 150-second soft limit. All 200 request inputs/order and 16-token output
  limits match the completed 4-RPS experiment. Poisson offsets are exactly twice
  the originals; the final scheduled arrival is at 100.024 seconds.
- Checksums and per-request equivalence passed. Hardware, KV/holding capacity,
  warm-up and latency thresholds remain unchanged through explicit CLI settings.
  No GPU job was launched during preparation.
- Inspected locally saved baseline reports: the T4 100-request 2-RPS baseline,
  A100 200-request 4-RPS baseline, and six-request CPU smoke all use Poisson
  arrivals. Constant schedules are covered by verification, not these reports.

## A100 40 GB hardware comparison prepared

- Changed only the benchmark GPU variant to explicit `A100-40GB:4` (two prefill,
  two decode), matching the user's cost-comparison request. Updated the runtime
  minimum-memory check from 75 to 35 GiB and the displayed base GPU cost estimate
  to $0.000583/GPU/s. Actual GPU identities/memory remain recorded.
- Reuses `workloads/a100-200-rps4/` unchanged: 200 Poisson requests at 4 RPS,
  16 output tokens, three passes with 90-second soft limits, 128 KV blocks,
  holding capacity four, four warm-up requests and a 600-second function timeout.
- New default output is `results/stage21-a100-40gb-200-rps4`, preserving the
  completed 80-GB report. Mocked launcher checks verified the GPU request,
  memory guard, workload, capacities and limits without launching GPU work.
- Run `modal run src/Modal/run_benchmark.py --repeats 3 --max-run-seconds 90
  --output results/stage21-a100-40gb-200-rps4` as one command. Results pending.

## A100 40 GB full-drain rerun prepared

- At the user's request, removed the default 90-second per-pass stop.
  `--max-run-seconds 0` (now the default) passes no replay deadline and bypasses
  the measured-budget guard. Positive limits retain the 270-second total guard.
- Three passes, 200 Poisson requests at 4 RPS, 16 output tokens, four A100 40 GB
  replicas and engine capacities are unchanged. The 600-second Modal function
  timeout still covers setup and all passes; a hard timeout can prevent export.
- New default output: `results/stage21-a100-40gb-200-rps4-drained`, preserving
  the partial 95/200-request report. Mocked launcher checks passed unbounded and
  explicit bounded modes; no GPU job was launched.

## Benchmark hardware decision: A100 80 GB

- User selected A100 80 GB for current and future performance benchmarks.
  Use two prefill and two decode replicas by default. T4 remains validation
  hardware; A100 40 GB results remain historical comparisons.
- Restored explicit `A100-80GB:4`, the 75-GiB minimum GPU-memory check, and
  matching GPU-only pricing display in `src/Modal/run_benchmark.py`.
- Default output: `results/stage21-a100-80gb-200-rps4-drained`, preserving all
  previous reports. Workload and runtime settings are unchanged: 200 Poisson
  requests at 4 RPS, 16 output tokens, three passes, no per-pass deadline,
  128 KV blocks, holding capacity four, four warm-up requests and a 600-second
  Modal function timeout covering setup and all passes.
- Launcher compilation/help checks passed. No GPU job was launched.


## Decode batching before work stealing

- Added continuous decode batching to local and Ray multi-decode coordinators.
  `decode_batch_size` defaults to one for the serial reference; use four for
  the next benchmark. Each decode worker runs up to four active requests in
  one model forward per tick, then rotates those requests through its queue.
  Requests enter and leave batches dynamically without waiting to fill them.
- Batched decode retains separate paged KV ownership and RoPE positions for
  each request. KV reads are padded to the longest history in that batch and
  padding is masked in attention. This implementation still gathers KV in
  Python and uses dense attention; it does not introduce a fused paged kernel.
  Prefill remains unbatched and request placement remains fixed.
- Finished requests release their own KV. Capacity rejection preserves all
  selected requests; a partial model-forward failure retires selected caches
  and preserves unrelated requests. Metrics observe each output token once.
- Reports now include configured `decode_batch_size` and each worker's
  `decode_batch_sizes` histogram (batch size -> successful decode calls).
  Warm-up counts are reset before each measured pass.
- CPU runner, worker, scheduling, metrics, system-report and dedicated batch
  checks passed. The dedicated check also passed with four real CPU Ray actors.
  GPU speed and exact-output agreement remain to be measured on A100 80 GB.
- Run `PYTHONPATH=src uv run python src/verify_decode_batch.py --ray` to verify.
- Next benchmark (one command):
  `modal run src/Modal/run_benchmark.py --workload-dir workloads/a100-200-rps4 --decode-batch-size 4 --repeats 3 --max-run-seconds 0 --output results/stage21-a100-80gb-200-rps4-batch4`
- Retains four A100 80 GB GPUs (two prefill, two decode), 200 Poisson requests
  at 4 RPS, 16 output tokens, 128 KV blocks/worker and all existing goodput
  limits. Per-pass deadlines remain disabled; the overall Modal timeout is
  still 600 seconds. Prior baseline artifacts are preserved. No GPU job was launched.


## Independent prefill and decode execution

- Added `IndependentRayCoordinator` for the multi-GPU benchmark. It tracks one
  outstanding operation per worker and processes ready Ray results without a
  global decode/transfer/prefill barrier. Separate actors can run prefill,
  decode batches and KV handoff concurrently. Ordinary local workers remain
  the serial CPU reference.
- Submission and admission use cached actor metadata/capacity. Each completed
  operation returns an updated worker snapshot; only idle actors are scheduled
  from those snapshots. Actors must be exclusively owned by the coordinator.
  Driver steps wait at most 2 ms for a ready result, keeping arrival replay and
  deadline checks responsive (driver overhead is additional).
- KV export refs pass directly to destination actors, retaining the existing
  CPU snapshot transport without pulling payloads into the controller. Source
  KV is released only after import acknowledgement. Capacity reserves the full
  decode budget, ownership remains fixed, and imports alternate with decode
  batches while a target has active requests to prevent admission starvation.
- Successful imports remain tracked through source-release retries. Failed
  decode caches are reconciled from returned actor state; unrelated workers
  remain schedulable. Unknown actor/task outcomes stop further dispatch and
  require caller cleanup rather than blindly retrying an import.
- Periodic system telemetry uses last completed-operation snapshots, including
  snapshot age and operation-in-flight flags; it does not enqueue blocking
  telemetry RPCs behind busy actors. Final telemetry remains authoritative.
  GPU utilization still comes from the independent nvidia-smi sampler.
- Modal uses independent scheduling exclusively; the execution-policy option
  and previous benchmark selection branch have been removed. Reports identify
  the scheduler. Warm-up drains the new event-driven
  coordinator with a 60-second limit before measured work begins.
- Real CPU Ray verification passed: decoders and peer prefill complete while
  another prefill is deliberately gated; exact outputs, one metric observation
  per token, immediate completion, bounded holding/KV pressure, actor reuse,
  source-release retries, partial decode failure reconciliation and deadline
  cleanup all passed. Existing batch, metrics, scheduling and system checks
  also passed. The user subsequently completed the GPU benchmark; see the
  independent scheduling result table above.
- Verification: `PYTHONPATH=src uv run python src/verify_independent.py`.
- Benchmark (one command):
  `modal run src/Modal/run_benchmark.py --workload-dir workloads/a100-200-rps4 --decode-batch-size 4 --repeats 3 --max-run-seconds 0 --output results/stage21-a100-80gb-200-rps4-independent-batch4`
- Keeps two prefill and two decode A100 80 GB replicas, the frozen 200-request
  4-RPS Poisson workload, existing capacities and a 600-second overall Modal
  timeout. Per-pass limits remain disabled. Compare with the earlier phased
  batch-four result; all previous reports are preserved.


## Independent scheduling is the only GPU benchmark path

- Removed the execution-policy argument from the Modal CLI and benchmark API.
  GPU runs always construct `IndependentRayCoordinator`. Existing historical
  artifacts are preserved; ordinary CPU workers remain a serial correctness
  reference. No GPU benchmark was launched for this cleanup.
- Updated current commands and documentation. Reports still describe the
  actual execution mechanism for provenance; it is no longer selectable.


## Eight-GPU scaling trial prepared: 4 prefill + 4 decode at 10 RPS

- User requested four A100 prefill replicas and four A100 decode replicas.
  Retained A100 80 GB. Doubled Poisson arrival rate from the selected baseline's
  5 RPS to 10 RPS, preserving nominal load per replica. The selected 2+2/5-RPS
  baseline table and artifacts remain unchanged; this is a scaling experiment.
- Generalized benchmark worker counts, source/target slicing, CPU/Ray resources,
  visible-GPU/process placement checks and report configuration. The Modal
  launcher requests `A100-80GB:8`, 16 CPUs and 128 GiB host memory, and supplies
  four prefill plus four decode replicas to the independent coordinator.
- Prepared `workloads/a100-200-rps10`; hashes and request contents verified
  against 5 RPS, with every arrival offset halved. Frozen manifest records
  four-source round-robin prompt totals. Added `--num-prefill-workers` to the
  freezer for reproducible metadata; existing defaults remain two.
- Keeps 200 requests, three passes, float32, 16 output tokens, decode batch four,
  128 KV blocks × 16 tokens per worker and holding capacity four. Warm-up grows
  to eight requests for the larger layout. Last arrival is 20.005 seconds;
  this finite trial has a shorter arrival window than the 5-RPS baseline.
- No per-pass deadline; overall Modal timeout remains 600 seconds. Displayed
  GPU-only estimate doubles to about $0.33/minute at the launcher's stored rate;
  setup and CPU/memory billing are additional. No GPU job was launched here.
- CPU eight-worker benchmark and real eight-actor Ray checks passed exact
  outputs, all four decoder assignments, per-token metrics and KV cleanup.
  Verification command: `PYTHONPATH=src uv run python src/verify_independent.py
  --workers-per-role 4` (one command).
- Run `modal run src/Modal/run_benchmark.py --workload-dir workloads/a100-200-rps10
  --decode-batch-size 4 --warmup-requests 8 --repeats 3 --max-run-seconds 0
  --output results/stage21-a100-80gb-4p4d-200-rps10-batch4` as one command.
  GPU results pending.


## Eight-GPU availability fallback: A100 40 GB

- At the user's request, switched the active 4-prefill/4-decode scaling trial
  from `A100-80GB:8` to `A100-40GB:8` after difficulty obtaining eight 80-GB GPUs.
  Reduced the minimum GPU-memory check from 75 to 35 GiB; actual GPU identity
  and memory remain recorded. The selected 2+2 A100 80 GB / 5-RPS baseline is
  preserved. This hardware change makes the new trial a hardware-and-replica
  comparison rather than an isolated scaling measurement.
- Kept 10 RPS, the same frozen 200 requests, decode batch four, three passes,
  eight warm-up requests, per-worker capacities, 16 CPUs, 128 GiB host memory
  and the 600-second overall timeout. Per-pass deadlines remain disabled.
- Updated default output to `results/stage21-a100-40gb-4p4d-200-rps10-batch4`.
  Updated the displayed estimate using Modal's published $0.000583/GPU/s rate:
  about $0.28/minute for eight GPUs, or $2.80 GPU-only for 600 seconds. Other
  billing is additional; this is not a cap or an availability guarantee.
- Launcher compilation and CLI-help checks passed. No GPU job was launched.
- Command: `modal run src/Modal/run_benchmark.py --workload-dir workloads/a100-200-rps10 --decode-batch-size 4 --warmup-requests 8 --repeats 3 --max-run-seconds 0 --output results/stage21-a100-40gb-4p4d-200-rps10-batch4`.


## Eight-GPU trial deferred; restored the selected baseline layout

- User deferred the eight-A100 experiment. The active launcher is again two
  prefill plus two decode A100 80 GB replicas at 5 RPS, with four warm-up
  requests. Eight-GPU workload files and historical reports are preserved.
- Current work is Stage 22; use its disabled/enabled commands above. A100 40 GB
  and eight-GPU instructions earlier in this document describe deferred trials.
