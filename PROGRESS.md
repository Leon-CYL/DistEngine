# DistEngine progress

Current implementation, reviewed October 20, 2026. All listed stages are complete.
Stage 23 is omitted by request. Generated workloads and results are local files
ignored by Git. This document describes the active code and benchmark workflow.

## Stage 0 — Project setup

Status: complete.

Python 3.11+ project with Hatchling, `uv.lock`, and packages `distengine`,
`models`, `layers`, and `utils`. `uv run distengine` and `python -m distengine`
print package information; they do not start an inference server. Dependencies
include PyTorch, Transformers, Ray and Hugging Face Hub; Modal is installed
separately. Workloads, results, model downloads and build artifacts are ignored.

## Stage 1 — TinyLlama reference

Status: complete.

Reference generation lives in `src/utils/hf_reference.py`. It resolves the
requested model revision and saves Hugging Face greedy-generation metadata.
The engine loader pins TinyLlama-1.1B-Chat-v1.0 to commit
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`. Pass `--output` to choose where to
save a reference report.

## Stage 2 — Understand TinyLlama

Status: complete.

`src/models/llama.py` and `src/layers/` implement token embeddings,
pre-normalized decoder blocks, RMSNorm, grouped-query causal attention, default
RoPE, SwiGLU and the LM head. Only default RoPE is supported. Architecture notes
remain in `docs/tinyllama.md`; README includes the final system diagram.

## Stage 3 — Request / sequence

Status: complete.

`src/distengine/request.py` retains `Request` and `RequestStatus` with
WAITING → RUNNING → FINISHED transitions, copied prompt tokens, generated tokens,
EOS/length termination and finish reason. `Submission` in `coordinator.py` is an
immutable queued description; workers construct the mutable running request.

## Stage 4 — Model implementation

Status: complete.

The custom Llama model remains in `src/models/llama.py`, with components
in `src/layers/`. `src/utils/load_model.py` loads pinned Hugging Face weights
strictly into the custom model in float32. Attention uses PyTorch matrix
operations and float32 softmax; fused FlashAttention/FlashInfer kernels,
tensor parallelism and CUDA graphs are not implemented.

## Stage 5 — Manual generation

Status: complete.

`src/utils/generation.py` provides uncached `generate_greedy()` and
`generate_greedy_cached()` using the custom model and greedy sampler.
`src/utils/manual_generate.py` is the text-generation CLI. The benchmark checks
completed outputs against a float32 cached single-sequence reference.

## Stage 6 — Paged KV cache

Status: complete.

`src/distengine/paged_kv_cache.py` contains shared `KVStorage`,
request-local `PagedKVCache` views and `BatchedPagedKVCache`. Physical storage is
[layer, block, KV head, block offset, head dimension]. Decode batches append one
token per row, gather/pad KV and mask padding for dense attention. This is paged
storage, not a fused PagedAttention kernel.

## Stage 7 — Block manager

Status: complete.

`src/distengine/block_manager.py` owns block allocation, capacity checks,
non-shrinking expansion and release. Cache views reference manager-owned block
tables; views are retired before blocks are freed. The final generated token
remains uncached, so maximum required KV is prompt length + output limit − 1.

## Stage 8 — Model runner

Status: complete.

`src/distengine/model_runner.py` validates requests/model/KV placement,
runs prompt prefill and single/batched decode, and releases request caches.
Prefill produces the first output token. Batched decode preflights capacity
before modifying KV; execution errors retire affected caches. Workers wrap the
runner's request lifecycle and release finished requests.

## Stage 9 — Basic scheduler

Status: complete.

`Coordinator` in `src/distengine/coordinator.py` owns FIFO prefill queues
and rotating decode queues. Each prefill worker has a waiting deque. Decode
workers rotate active request IDs after each batch, with admission interleaved
with decode progress.

## Stage 10 — Engine integration

Status: complete.

`Coordinator`, `PrefillWorker`, `DecodeWorker` and `ModelRunner` integrate
request scheduling, model execution and KV management. One multi-worker
scheduler accepts worker lists, including lists containing one replica.

## Stage 11 — Worker roles and Ray orchestration

Status: complete.

`src/distengine/worker.py` contains ordinary prefill/decode replicas,
shared runner setup, Ray actor implementations and scheduled-operation envelopes.
Models are constructed inside actors. Ordinary workers use the same coordinator
synchronously for CPU checks; Ray workers execute independently.

## Stage 12 — Separate GPU placement

Status: complete.

The GPU benchmark reserves one GPU per actor via
`Actor.options(num_gpus=1)`. Actors validate a single visible device and colocated
model/KV storage; the harness checks distinct GPU assignments/processes, GPU
family and memory. Modal allocates four A100-80GB GPUs, two per role.

## Stage 13 — KV transfer

Status: complete.

`src/distengine/kv_transfer.py` exports versioned, cloned CPU snapshots
with model identity, request metadata and KV tensors. Import validates payload,
shape, dtype and capacity before publishing destination state. Ray passes the
export reference to the destination actor. Source KV remains owned until import
is acknowledged, then release is scheduled. No RDMA/direct GPU transfer exists.

## Stage 14 — Prefill KV holding pool

Status: complete.

`PrefillHoldingPool` in `src/distengine/holding_pool.py` bounds ready
requests per source. `PrefillWorker` adds unfinished prefill results and removes
them on source release. The active Modal benchmark defaults to four held
requests per prefill worker; `max_new_tokens=1` or immediate EOS can finish at
prefill without a transfer.

## Stage 15 — Decode-pull admission

Status: complete.

`Coordinator._dispatch_available()` checks idle decode workers and their
free KV blocks before importing held requests. Destination admission reserves
prompt + maximum output − 1 KV tokens through `import_scheduled()`. Decode owner
is fixed after import. Pending acknowledgements release source KV before new
prefills.

## Stage 16 — Backpressure

Status: complete.

Backpressure is integrated into `Coordinator`: prefill starts only when
its source has a free holding slot and enough prompt KV blocks. Pending work
stays in upstream queues until decode/import/release frees capacity. At most one
operation is outstanding per actor. Independent actor progress avoids a global
prefill/decode barrier.

## Stage 17 — Multiple prefill workers

Status: complete.

One `Coordinator` accepts multiple prefill workers with independent
queues, holding pools and KV storage. Submissions enter round-robin source
queues; optional stealing can change execution ownership before prefill starts.
The benchmark uses two prefill workers.

## Stage 18 — Workload generator

Status: complete.

`src/distengine/workload.py` implements seeded constant/Poisson arrivals,
validated schema-v1 JSON save/load and timed replay. Randomness is local; replay
uses saved offsets, submits overdue arrivals and drains pending work. Optional
soft deadlines leave cleanup to the caller. Replay shares the coordinator's
metrics recorder and clock.

## Stage 19 — Mixed request workload

Status: complete.

`src/distengine/mixed_workload.py` selects exact class quotas using
largest remainder, independent arrival/prompt seeds and deterministic sampling.
`src/utils/freeze_workload.py` freezes nine chat-formatted TinyLlama prompts,
checksums and a manifest with assignments and round-robin totals. The selected
workload uses 200 requests, 5 RPS, 16 output tokens and a 60/30/10 mix in
`workloads/baseline/`. These are explicit README arguments; freezer CLI defaults
remain 100 requests, 2 RPS, 32 output tokens and `workloads/stage19`.

## Stage 20 — Multiple decode workers

Status: complete.

Multiple decode replicas are handled by the same `Coordinator` and
`DecodeWorker` implementations. Imports select an eligible idle target; requests
stay on that target for their lifetime. Each target runs continuous decode
batches, default maximum four in Modal. The benchmark uses two decode workers.

## Stage 21 — Metrics + baseline benchmark

Status: complete.

`metric.py` records driver-observed TTFT, TPOT, end-to-end latency,
prefill queue wait, submission delay, throughput, failures and joint-SLO goodput.
`telemetry.py` records worker activity, queues, KV occupancy and sampled
nvidia-smi utilization. Worker busy fraction is not GPU utilization.
`src/utils/benchmark.py` performs warm-up, float32 cached-reference checks,
measurement and cleanup; `src/Modal/run_benchmark.py` launches two prefill plus
two decode A100-80GB actors. Defaults: three passes, batch four, 128 blocks × 16
tokens, holding four, warm-up four, 0.5-second sampling, no measured deadline and
600-second overall Modal timeout. Output directories are chosen at launch.

## Stage 22 — Prefill work stealing + initial comparison

Status: complete.

`Coordinator._steal_prefill()` lets an idle source with an empty queue
steal one compatible, unstarted whole request from the tail of the donor with
the most queued prompt tokens. No in-flight work, held KV or active decode
migrates. `--stealing` enables it; `--compare-stealing` runs counterbalanced
off/on, on/off, off/on passes on the same actors. Reports include steal events
and final prefill/decode assignments.

## Stage 24 — Final benchmark

Status: complete.

The final benchmark uses the same frozen workload and configuration for
baseline and stealing. Completed outputs must match the float32 cached greedy
reference; successful runs drain all requests and release KV. Use the paired
command below to compare both policies on the same loaded actors. New GPU
correctness and performance verification is pending execution of that command.

## Stage 25 — cleanup and final documentation

Status: complete.

Code is organized into `src/distengine/`, `src/models/`, `src/layers/`,
`src/utils/` and one Modal benchmark launcher. README includes the architecture
diagram, setup, benchmark commands, results, limitations, future work and
references. Runtime validation, benchmark output checks and KV cleanup checks
remain in the active implementation.

## Run the GPU benchmark

Run from the repository root after installing dependencies and authenticating
Modal as described in README.md. The frozen workload should exist at
`workloads/baseline/` (200 Poisson requests, 5 RPS, up to 16 output tokens).

Recommended comparison: both policies on the same four loaded GPU actors,
three passes per policy (six measured passes total):

```bash
modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/baseline \
  --compare-stealing --decode-batch-size 4 \
  --repeats 3 --max-run-seconds 0 \
  --output results/final-paired
```

For independent baseline and stealing runs:

```bash
modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/baseline \
  --no-stealing --decode-batch-size 4 \
  --repeats 3 --max-run-seconds 0 \
  --output results/final-baseline

modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/baseline \
  --stealing --decode-batch-size 4 \
  --repeats 3 --max-run-seconds 0 \
  --output results/final-work-stealing
```

Choose unused output directories. Reports are saved locally as `baseline.json`
and `run-00.json`, `run-01.json`, etc. In the paired report compare
`comparison.no-stealing` and `comparison.stealing`. Each run includes outputs,
request metrics, worker activity, GPU samples and cleanup state in the full
report. An output mismatch or cleanup failure raises an error.

Configuration: two prefill + two decode A100-80GB GPUs, float32, 128 KV blocks
of 16 tokens per worker, holding capacity four, four warm-up requests per pass,
and 0.5-second sampling. Zero disables the measured deadline; the overall Modal
function timeout remains 600 seconds and includes setup, warm-up and all passes.
GPU runs incur charges. Separate allocations can vary in execution speed;
prefer paired measurements when evaluating the effect of stealing.

## Validation and limitations

Local CPU baseline/stealing smoke checks and frozen-input checksum checks pass.
CPU checks use a deterministic small Llama and do not establish TinyLlama GPU
numerics or throughput. No new GPU run was launched for this documentation edit.
TTFT/TPOT/E2E are driver-observed; worker busy time is distinct from sampled GPU
utilization. Current reference generation uses the custom float32 cached model.
KV handoff passes through CPU memory; attention gathers/pads KV for dense
computation. RDMA, FlashInfer and tensor parallelism are future work. Performance
at the selected 5-RPS load does not establish maximum serving capacity.
