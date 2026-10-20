"""TinyLlama baseline and work-stealing comparisons on four A100s."""

import math
import json
from pathlib import Path

import modal

app = modal.App("distengine-benchmark")
cache = modal.Volume.from_name("distengine-hf-cache", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11")
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    image = (
        image.uv_sync(str(root))
        .env({"PYTHONPATH": "/app/src", "HF_HOME": "/cache/huggingface",
              "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"})
        .add_local_dir(root / "src", "/app/src", ignore=["**/__pycache__/**", "**/*.pyc"])
    )


@app.function(image=image, gpu="A100-80GB:4", cpu=8, memory=65536, timeout=600,
              volumes={"/cache": cache})
def benchmark(files, payload, provenance, repeats, sample_interval, thresholds, engine_config):
    from utils.benchmark import run_benchmark
    from distengine.metric import LatencyLimits
    from utils.freeze_workload import MODEL_ID, REVISION
    manifest = json.loads(files["manifest.json"])
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise ValueError("Frozen inputs do not match the pinned model")
    report = run_benchmark(payload, repeats=repeats,
                           interval_s=sample_interval, limits=LatencyLimits(**thresholds),
                           provenance=provenance, frozen_files=files, **engine_config)
    cache.commit()
    return report


@app.local_entrypoint()
def main(repeats: int = 3, sample_interval: float = .5,
         decode_batch_size: int = 4,
         ttft_limit: float = 2.0, tpot_limit: float = .1, e2e_limit: float = 10.0,
         output: str = "results/benchmark", workload_dir: str = "workloads/baseline",
         num_blocks: int = 128, max_held_requests: int = 4, warmup_requests: int = 4,
         max_run_seconds: float = 0.0, stealing: bool = False,
         compare_stealing: bool = False):
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from utils.benchmark_io import frozen_inputs, source_identity, save_report
    if compare_stealing and stealing:
        raise ValueError("Use --compare-stealing without --stealing")
    if Path(output).exists():
        raise ValueError("Choose a new output directory to preserve earlier runs")
    if any(n <= 0 for n in (repeats, decode_batch_size, num_blocks, max_held_requests, warmup_requests)):
        raise ValueError("Repeats, batch size, KV capacity and warm-up count must be positive")
    if not math.isfinite(sample_interval) or sample_interval <= 0:
        raise ValueError("Sampling interval must be finite and positive")
    if not math.isfinite(max_run_seconds) or max_run_seconds < 0:
        raise ValueError("Measured time limit must be nonnegative; 0 disables it")
    thresholds = dict(ttft_s=ttft_limit, tpot_s=tpot_limit, e2e_s=e2e_limit)
    if any(not math.isfinite(v) or v <= 0 for v in thresholds.values()):
        raise ValueError("Goodput limits must be finite and positive")
    files = frozen_inputs(root, workload_dir)
    payload = json.loads(files["mixed-v1.json"])
    engine_config = dict(prefill_workers=2, decode_workers=2, stealing_enabled=stealing,
                         compare_stealing=compare_stealing, decode_batch_size=decode_batch_size,
                         num_blocks=num_blocks, block_size=16, max_held_requests=max_held_requests,
                         warmup_requests=warmup_requests, gpu_family="A100", min_gpu_memory_gib=75,
                         max_run_s=max_run_seconds or None)
    print("A100-80GB x4: 2 prefill + 2 decode; float32; "
          f"{len(payload['requests'])} requests; {payload['rate_rps']} RPS; "
          f"decode batch up to {decode_batch_size}; {repeats} repeats/mode")
    print("Overall function timeout: 600s, including setup and all passes. "
          "GPU, CPU/memory and startup billing are additional to measured runtime.")
    report = benchmark.remote(files, payload, source_identity(root), repeats, sample_interval, thresholds, engine_config)
    save_report(report, output)
    print(f"Report saved: {output}/baseline.json")
