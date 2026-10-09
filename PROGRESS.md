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

Status: core checkpoint complete; additional assignment examples remain.

- Implemented `src/distengine/hf_reference.py` with prompt, output-length,
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

Remaining assignment work: capture the explanation and sentence-completion
examples with the pinned revision, and write answers to the stage 1 learning
questions. The README documents the model-running command.

## Stage 2 — Understand TinyLlama

Status: complete.

- Added [TinyLlama architecture notes](docs/tinyllama.md) covering embeddings,
  decoder blocks, RMSNorm, GQA, RoPE, causal attention, SwiGLU, and logits.
- Documented model dimensions, Q/K/V shapes, and planned per-layer KV-cache shapes.

## Stage 4 — Model implementation

Status: implementation complete; execution and correctness verification pending.

- Implemented embeddings, LM head, linear projections, RMSNorm, SwiGLU, RoPE,
  and full-sequence causal grouped-query attention in PyTorch.
- Assembled 22 decoder blocks and the final normalization in `models/llama.py`.
- Added pinned checkpoint loading with strict parameter-name and shape matching.
- Fixed config typing and read the RoPE base from the installed Transformers
  version's `rope_parameters` dictionary.
- Scoped the downloaded-model ignore rule to `/models/`, keeping model source
  files under `src/distengine/models/` available for tracking.

The user-reported run on October 8, 2026 loaded HF weights but failed during
custom model construction because `config.rope_theta` was unavailable. The
config access has been fixed; a successful rerun has not yet been reported.

## Stage 5 — Manual generation

Status: implemented; successful execution and HF comparison pending.

- Added greedy sampling and a manual generation loop without HF `generate()`.
- Added `manual_generate.py` with prompt and maximum-output-length arguments.
- Reuses the pinned tokenizer and chat template; currently supports one unpadded
  request and recomputes the full sequence without a KV cache.
- Documented the run command in the README. No new inference run or tests were
  performed for this documentation update.

## Roadmap

| Stage | Focus | Completion checkpoint | Status |
| --- | --- | --- | --- |
| 0 | Project setup | uv project, dependencies, package, README, PROGRESS, and .gitignore. | Complete |
| 1 | TinyLlama reference | Load TinyLlama-1.1B with Hugging Face and generate reference text. | Core complete; assignment examples pending |
| 2 | Understand TinyLlama | Document embeddings, layers, GQA, RoPE, attention, FFN, logits, and KV shapes. | Complete |
| 3 | Request / sequence | Prompt tokens, generated tokens, status, and max output length. | Pending |
| 4 | Model implementation | Minimal TinyLlama model components in PyTorch. | Implemented; verification pending |
| 5 | Manual generation | Greedy generation without HF generate(). | Implemented; verification pending |
| 6 | Basic KV cache | Contiguous prefill/decode KV cache; match HF output. | Pending |
| 7 | Paged KV cache | Fixed-size KV blocks and per-request block tables. | Pending |
| 8 | Block manager | Allocation, freeing, reuse, and available-block tracking. | Pending |
| 9 | Model runner | Prepare inputs, positions, and KV information; execute prefill/decode. | Pending |
| 10 | Basic scheduler | FIFO waiting/running/finished queues; no new continuous-batching research. | Pending |
| 11 | nano-vLLM-style engine | Connect LLMEngine, Scheduler, ModelRunner, and BlockManager on one GPU. | Pending |
| 12 | Separate prefill/decode roles | Explicit local PrefillWorker and DecodeWorker roles. | Pending |
| 13 | Two-GPU DistEngine | Prefill and decode on separate GPUs; confirm output correctness. | Pending |
| 14 | KV transfer | Simple worker transfer, initially GPU → CPU → GPU if necessary. | Pending |
| 15 | Prefill KV holding pool | Retain ready KV until decode admission; bounded capacity. | Pending |
| 16 | Decode-pull admission | Decode pulls a request and KV only when capacity is available. | Pending |
| 17 | Backpressure | Stop prefill admission when the holding pool is full; queue upstream. | Pending |
| 18 | Multiple prefill workers | Independent prefill workers and request queues. | Pending |
| 19 | Prefill work stealing | Steal unstarted whole requests; enable/disable switch. | Pending |
| 20 | Multiple decode workers | Simple placement; keep each request on its assigned decode worker. | Pending |
| 21 | Ray orchestration | Thin GPU/process actors; inference stays in normal Python classes. | Pending |
| 22 | Modal deployment | Start with one prefill GPU and one decode GPU; scale as necessary. | Pending |
| 23 | Workload generator | Constant and Poisson arrivals with reproducible seeds. | Pending |
| 24 | Mixed request workload | Short, medium, and long prompts that imbalance prefill work. | Pending |
| 25 | Metrics instrumentation | TTFT, TPOT, E2E, prefill wait, throughput, goodput, GPU utilization, idle time, queue length, and KV occupancy. | Pending |
| 26 | Final benchmark | Compare exactly the same workload with and without stealing. | Pending |
| 27 | Cleanup + report | Simplify code; document architecture, results, limitations, and RDMA/FlashInfer/TP future work. | Pending |

## Verification

Passed locally on macOS with Python 3.11.9:

- `uv sync --locked --offline --no-cache`: lockfile and environment agree.
- `uv run --no-sync --no-cache distengine`: package command runs.
- `uv run --no-sync --no-cache python -m distengine`: module command runs.
- Imported `distengine`, `torch`, and `transformers` successfully.
- `uv build --no-cache`: produced wheel and source archive in `dist/`.
- `git check-ignore .venv dist`: environment and build artifacts are ignored.

No model download, inference, or GPU correctness checks were performed at stage 0.
