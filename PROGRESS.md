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
strictly into the custom model, using float32 on CPU and CUDA. Attention uses PyTorch matrix
operations and float32 softmax; fused FlashAttention/FlashInfer kernels,
tensor parallelism and CUDA graphs are not implemented.

## Stage 5 — Manual generation

Status: complete.

`src/utils/generation.py` provides uncached `generate_greedy()` and
`generate_greedy_cached()` using the custom model and greedy sampler.
`src/utils/manual_generate.py` is the text-generation CLI. The benchmark checks
completed outputs against a cached single-sequence reference in float32.

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

`src/distengine/kv_transfer.py` exports version-2 owned KV snapshots with model
identity, request metadata and an explicit transport. CUDA workers use Ray
Direct Transport/NCCL exclusively. CPU-only smoke checks exchange CPU tensors. Decode validates tensor shape, dtype, finiteness and device,
loads all layers by physical block, and acknowledges after device writes finish.
The source retains KV until acknowledgment. The driver passes references only;
`ray.wait(fetch_local=False)` prevents export materialization on the driver.
Actual NCCL GPU validation remains pending; local CPU/Gloo direct transport is
checked. NCCL link selection depends on the hardware; RDMA is not established.


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
command below to compare both policies on the same loaded actors. NCCL GPU
correctness and performance must be checked in a fresh run.

## Stage 25 — cleanup and final documentation

Status: complete.

Code is organized into `src/distengine/`, `src/models/`, `src/layers/`,
`src/utils/` and the online Modal benchmark launcher. README includes the architecture
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
  --output results/float32-paired
```

For independent baseline and stealing runs:

```bash
modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/baseline \
  --no-stealing --decode-batch-size 4 \
  --repeats 3 --max-run-seconds 0 \
  --output results/float32-baseline

modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/baseline \
  --decode-batch-size 64 \
  --repeats 3 --max-run-seconds 0 \
  --output results/float32-work-stealing
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
utilization. Current reference generation uses the custom model in float32.
KV handoff uses NCCL CUDA tensors exclusively on GPU;
attention gathers/pads KV for dense
computation. Cross-node RDMA, FlashInfer and tensor parallelism are future work. Performance
at the selected 5-RPS load does not establish maximum serving capacity.


## NCCL KV handoff

- The online GPU benchmark defaults to ``; CPU benchmark
  runs use `cpu`. Actors enable Ray's tensor-transport support. The benchmark
  creates and destroys the actor collective group around all repeated passes.
- Prefill returns owned CUDA tensors through RDT; decode receives those tensors
  on its assigned GPU. Metadata/control stays on Ray. A CPU payload is rejected
  in NCCL mode. The source remains held until successful import acknowledgment.
- Snapshot loading copies physical blocks across all layers rather than issuing
  one CUDA copy per token per layer. Allocation/publish rollback remains intact.
- Every source/target pair passes a dtype/device/content probe before warm-up.
  Reports retain GPU topology, probe outcomes, per-worker sent/received KV bytes,
  and end-to-end `kv_handoffs` events. Full-drain byte totals must agree.
- The Modal image adds `cupy-cuda13x[ctk]==14.2.0` to its GPU environment. CuPy is
  not added to the local CPU dependency set. Ray RDT is experimental.
- Local checks cover noncontiguous block tables, float32 exact CPU outputs,
  snapshot isolation, malformed transport rejection, source ownership and failed
  import rollback. Real CPU Ray/Gloo checks cover all four source/target pairs,
  nested tensor envelopes, acknowledgment, actor reuse and cleanup. A100/NCCL
  performance and numerical validation have not been run for this change.

```bash
modal run src/Modal/run_benchmark.py \
  --workload-dir workloads/baseline \
  --stealing \
  --decode-batch-size 64 --repeats 1 --max-run-seconds 0 \
  --output results/nccl-work-stealing
```

GPU-to-CPU staging has been removed. Historical saved reports remain unchanged. Link-level
routing (NVLink/PCIe/network) is hardware-dependent; no RDMA claim is made.

The precision selector and BF16 configuration were removed. Model loading,
workers and reference generation use float32. GPU topology collection is best
effort; a diagnostic failure is recorded and mandatory NCCL probes still run.

## Saturation measurement

`src/Modal/run_capacity.py` reuses the online harness and GPU actors for a decode
batch sweep with simultaneous arrivals and fixed output lengths (EOS disabled).
It retains float32 cached-reference validation, NCCL preflight, telemetry and KV
cleanup. KV blocks are configurable independently of the online baseline.
Reports retain the derived workload and per-batch summaries. A partial pass stops
the sweep; observed token throughput is distinct from full-drain throughput.
Local CPU sweep checks passed; GPU capacity numbers require a new Modal run.

## Decode profiling

Added opt-in `src/utils/profiller.py` and `src/Modal/run_profiler.py`. One A100
profiles the real decode-worker batch path in isolation. Separate CUDA-event and
PyTorch trace passes exclude prefill/setup and compare generated tokens. Trace
ranges distinguish KV gather/padding, reads, attention, forward and sampling.
Reports and compressed traces return to local storage. CPU trace/range, output
consistency and cleanup checks passed; CUDA profiling requires the Modal run.
This does not measure NCCL/Ray scheduling overhead or change serving execution.

## Vectorized decode KV updates

`BatchedPagedKVCache` now precomputes write/read block indices per decode step
and reuses them across layers. Shared-storage batches perform one indexed write
per K/V tensor and gather directly into batched attention inputs, zeroing padded
positions. The per-request append/read/concatenate/pad loop is bypassed in the
engine path. Independent storage views are no longer accepted by batched decode. Shared
batches reject overlapping block ownership and repeated layer appends.
Local checks cover exact cache equivalence, fragmented block tables, mixed and
empty prefixes, boundary crossings, NaN padding and float32/BF16 storage. The
float32 engine baseline/stealing reference checks pass; GPU correctness and
performance need the new profiler and capacity runs before claiming a speedup.

## Final defaults and cleanup

GPU workers now always use device-resident NCCL handoff. Removed the transport
selector and GPU-to-CPU staging and CPU-to-GPU import copies. CPU-only smoke
workers still exchange native CPU tensors. Batched decode requires shared KV
storage and uses the vectorized path; the legacy independent-storage loop was
removed. The capacity command defaults to the final batch-64, 256-request,
64-output-token, three-pass configuration with stealing enabled. Online benchmarks
and the coordinator default to batch 64 and stealing. The no-stealing switch
remains only to reproduce the documented baseline. Saved reports are preserved.

## Benchmark and profiling guide

Moved from README.md to keep the project introduction concise.

## Saturation benchmark

The final configuration is now the default: 256 simultaneous requests, 64 output
tokens each with EOS disabled, batch limit 64, three passes, 2,048 KV blocks per
worker, float32, NCCL, vectorized KV updates and work stealing. Use a fresh output
directory for each run:

```bash
modal run src/Modal/run_capacity.py --output results/optimized-final
```

Results are local in `results/optimized-final/baseline.json` and per-pass files.
Inspect `capacity_summary` for throughput and completion, and `runs` for latency,
GPU activity and exact-output checks. Each pass has a 90-second measured limit;
a partial pass stops the run. Setup/reference/warm-up are excluded from measured
throughput but included in the 600-second function timeout. GPU jobs incur charges.
Workload size and batch-sweep options remain available for further experiments.

## Batched KV updates

Decode batches sharing a physical KV pool use indexed writes for all requests
per layer, followed by direct batched gathering and zero padding. Block mappings
are built once per decode step and reused across layers. This replaces the
per-request copy/concatenate/pad path while keeping attention math, block
ownership, capacity checks and transfer behavior unchanged. Batched decode requires a shared KV pool; the older independent-storage loop
has been removed. CPU equivalence and GPU exact-output checks passed. The final capacity
results in README.md use this vectorized path.

## Decode profiling

`src/utils/profiller.py` profiles the actual `DecodeWorker.step_batch()` path on
one GPU, with no changes to normal benchmark execution. Prefill runs before
measurement. A separate timing pass uses CUDA events; a second pass captures a
PyTorch CPU/CUDA trace, tensor shapes and memory events. Temporary ranges label
KV append/gather/padding, KV reads, attention, model forward and greedy sampling.
The harness checks that traced and untraced generated tokens match and releases
all KV blocks. This is a decode microprofile, excluding Ray, NCCL and coordinator
overhead, rather than an end-to-end capacity benchmark.

```bash
modal run src/Modal/run_profiler.py \
  --workload-dir workloads/baseline --batch-size 64 \
  --warmup-steps 4 --timing-steps 16 --trace-steps 2 \
  --num-blocks 2048 --output results/decode-profile-b64
```

This allocates one A100-80GB and incurs GPU charges. Results are saved locally:

- `report.json`: stream elapsed CUDA-event timing per batch, wall time, memory,
  kernel event count, output consistency and cleanup.
- `operators.txt`: operators ranked by self CUDA time. Compare copy/concatenation,
  padding, matrix operations and synchronization costs.
- `decode.trace.json.gz`: compressed Chrome trace. Decompress it and open the
  resulting JSON in Perfetto or Chrome tracing to inspect CPU launch gaps and
  GPU kernels alongside the labeled ranges.

Profiler overhead affects trace timings; use the separate timing pass for an
untraced comparison. CUDA event durations include stream scheduling gaps, so they
are not pure kernel execution time. Nested trace ranges overlap and must not be
summed. A batch contains one next-token prediction per active request.
