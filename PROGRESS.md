# DistEngine progress

## Stage 0 — Project setup

Status: complete.

- Initialized a Python 3.11 uv project with a Hatchling build backend.
- Created the installable `src/distengine` package and command-line entry point.
- Added PyTorch and Transformers for the upcoming TinyLlama reference/model stages.
- Added a reproducible dependency lockfile and `.gitignore`.
- Documented setup, editable installation, dependency management, and packaging.
- Kept model loading and inference implementation for later stages.

## Roadmap

| Stage | Focus | Completion checkpoint | Status |
| --- | --- | --- | --- |
| 0 | Project setup | uv project, dependencies, package, README, PROGRESS, and .gitignore. | Complete |
| 1 | TinyLlama reference | Load TinyLlama-1.1B with Hugging Face and generate reference text. | Pending |
| 2 | Understand TinyLlama | Document embeddings, layers, GQA, RoPE, attention, FFN, logits, and KV shapes. | Pending |
| 3 | Request / sequence | Prompt tokens, generated tokens, status, and max output length. | Pending |
| 4 | Model implementation | Minimal TinyLlama model components in PyTorch. | Pending |
| 5 | Manual generation | Greedy generation without HF generate(). | Pending |
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
