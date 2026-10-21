# DistEngine

DistEngine is a distributed inference engine for TinyLlama, inspired by [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin). It runs prefill and decode independently across GPU workers, with paged KV storage, continuous decode batching, and optional work stealing to balance prefill queues. Ray coordinates the workers, and Modal runs the GPU benchmark.


### Baseline: batch 64 without work stealing

| Metric | Result |
| --- | ---: |
| Decode batch | 64 |
| Average output throughput | **744.96 tokens/s** |
| Average request throughput | 11.64 requests/s |
| Average runtime | 21.994 s |


### Final result: batch 64 with work stealing

| Metric | Result |
| --- | ---: |
| Decode batch | 64 |
| Average output throughput | **905.48 tokens/s** |
| Average request throughput | 14.15 requests/s |
| Average runtime | 18.095 s |

### 1. Install and authenticate

```bash
uv sync --locked
uv tool install modal
modal setup
```


### 2. Run the benchmark

Final optimized configuration with work stealing:

```bash
modal run src/Modal/run_capacity.py --output results/optimized-final
```

Baseline without work stealing:

```bash
modal run src/Modal/run_capacity.py --no-stealing --output results/baseline-final
```

Both commands use four A100-80GB GPUs, NCCL, vectorized KV updates, batch 64, 256 simultaneous requests, 64 output tokens and three passes. GPU runs incur charges. Use a new output directory for each run.

## View results

Results are saved **on your computer**:

- `results/optimized-final/baseline.json`: full optimized report.
- `results/baseline-final/baseline.json`: full no-stealing baseline report.
- `run-00.json`, `run-01.json` and `run-02.json` in each directory: individual passes.


## References

Research papers behind the project's design and related LLM serving techniques:

- [PagedAttention (vLLM): Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference](https://arxiv.org/abs/2403.02310)
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670)
