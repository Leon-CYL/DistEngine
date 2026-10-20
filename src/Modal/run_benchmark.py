"""Stage 22: four-A100-80GB TinyLlama prefill work-stealing comparison."""

import math
import json
from pathlib import Path

import modal

app = modal.App("distengine-stage-21-baseline")
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
def benchmark(files, provenance, repeats, sample_interval, thresholds, engine_config):
    from benchmark_stage21 import run_benchmark
    from distengine.metric import LatencyLimits
    from freeze_workload import MODEL_ID, REVISION
    manifest = json.loads(files["manifest.json"])
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise ValueError("Frozen inputs do not match the pinned model")
    report = run_benchmark(json.loads(files["mixed-v1.json"]), repeats=repeats,
                           interval_s=sample_interval, limits=LatencyLimits(**thresholds),
                           provenance=provenance, frozen_files=files, **engine_config)
    cache.commit()
    return report


@app.local_entrypoint()
def main(repeats: int = 3, sample_interval: float = .5,
         ttft_limit: float = 2.0, tpot_limit: float = .1, e2e_limit: float = 10.0,
         output: str = "",
         workload_dir: str = "workloads/a100-200-rps5", num_blocks: int = 128,
         max_held_requests: int = 4, warmup_requests: int = 4,
         max_run_seconds: float = 0.0, decode_batch_size: int = 4, stealing: bool = False,
         compare_stealing: bool = False):
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from benchmark_io import frozen_inputs, source_identity, save_report
    if not output:
        mode = "paired" if compare_stealing else "stealing" if stealing else "no-stealing"
        output = f"results/stage22-a100-80gb-200-rps5-{mode}"
    if Path(output).exists():
        raise ValueError("Choose a new output directory to preserve earlier runs")
    if compare_stealing and stealing:
        raise ValueError("Use --compare-stealing without --stealing")
    measured_passes = repeats * (2 if compare_stealing else 1)
    if repeats <= 0 or sample_interval <= 0:
        raise ValueError("Repeats and sample interval must be positive")
    thresholds = dict(ttft_s=ttft_limit, tpot_s=tpot_limit, e2e_s=e2e_limit)
    if any(not math.isfinite(v) or v <= 0 for v in thresholds.values()):
        raise ValueError("Goodput limits must be finite and positive")
    if not math.isfinite(max_run_seconds) or max_run_seconds < 0:
        raise ValueError("Measured time limit must be nonnegative; 0 disables it")
    if max_run_seconds > 0 and measured_passes * max_run_seconds > 270:
        raise ValueError("Measured budget exceeds 270 seconds; reduce repeats or time per run")
    engine_config = dict(prefill_workers=2, decode_workers=2, stealing_enabled=stealing, compare_stealing=compare_stealing, decode_batch_size=decode_batch_size, num_blocks=num_blocks, block_size=16,
                         max_held_requests=max_held_requests, warmup_requests=warmup_requests,
                         gpu_family="A100", min_gpu_memory_gib=75, max_run_s=max_run_seconds or None)
    for value in (decode_batch_size, num_blocks, max_held_requests, warmup_requests):
        if value <= 0:
            raise ValueError("Engine capacities and warm-up count must be positive")
    files = frozen_inputs(root, workload_dir)
    payload = json.loads(files["mixed-v1.json"])
    for request in payload["requests"]:
        if len(request["prompt_token_ids"]) + request["max_new_tokens"] - 1 > num_blocks * 16:
            raise ValueError("Workload exceeds configured per-worker KV capacity")
    print("GPU-only estimate at published base rate $0.000694/GPU/s: "
          "$0.17/minute for four GPUs; 600s function timeout corresponds to $1.67 GPU-only. "
          "Startup, CPU/memory and other billing are extra; this is not a dollar cap.")
    print(f"A100-80GB x4: 2 prefill + 2 decode; {len(payload['requests'])} requests; "
          f"{payload['rate_rps']} RPS; stealing={'paired off/on' if compare_stealing else stealing}; independent scheduling; decode batch up to {decode_batch_size}/worker; {num_blocks} KV blocks/worker; {measured_passes} measured passes; "
          f"measured limit/repeat: {str(max_run_seconds) + 's' if max_run_seconds else 'disabled (drain all requests)'}")
    report = benchmark.remote(files, source_identity(root), repeats, sample_interval, thresholds, engine_config)
    save_report(report, output)
    print(f"Baseline saved: {output}/baseline.json")
