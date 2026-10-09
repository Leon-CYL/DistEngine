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

- The user ran `distengine.request` successfully and reran `distengine.verify`
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
  files under `src/distengine/models/` available for tracking.

The initial user-reported run on October 8, 2026 failed during custom model
construction because `config.rope_theta` was unavailable. After the config fix,
the user's October 8 screenshot confirmed successful model construction, strict
checkpoint loading, and inference for the factual prompt.

Verified from the user's October 8, 2026 terminal output of
`uv run python -m distengine.verify`, on MPS with float32 and the pinned checkpoint:

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

The user's October 8, 2026 `distengine.verify` run passed all generation checks:

- Exact HF token agreement for factual (8 tokens), explanation (32 tokens), and
  sentence-completion (21 tokens) prompts.
- Repeatability, prompt preservation, output-limit stopping, and factual EOS
  stopping passed.

Additional local validation confirmed `generate_greedy()` raises `ValueError`
for an empty prompt, a batch size of two, a one-dimensional input, and zero or
negative output limits. All stage 4/5 verification checks are complete for the
current single-request, unpadded, full-sequence implementation. KV-cache
correctness remains part of stage 6.

## Roadmap

| Stage | Focus | Completion checkpoint | Status |
| --- | --- | --- | --- |
| 0 | Project setup | uv project, dependencies, package, README, PROGRESS, and .gitignore. | Complete |
| 1 | TinyLlama reference | Load TinyLlama-1.1B with Hugging Face and generate reference text. | Complete |
| 2 | Understand TinyLlama | Document embeddings, layers, GQA, RoPE, attention, FFN, logits, and KV shapes. | Complete |
| 3 | Request / sequence | Prompt tokens, generated tokens, status, and max output length. | Complete |
| 4 | Model implementation | Minimal TinyLlama model components in PyTorch. | Complete |
| 5 | Manual generation | Greedy generation without HF generate(). | Complete |
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
