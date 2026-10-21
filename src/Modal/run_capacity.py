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
        .run_commands(
            "/.uv/uv pip install --python /.uv/.venv/bin/python 'cupy-cuda13x[ctk]==14.2.0'"
        )
        .env(
            {
                "PYTHONPATH": "/app/src",
                "HF_HOME": "/cache/huggingface",
                "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            }
        )
        .add_local_dir(root / "src", "/app/src", ignore=["**/__pycache__/**", "**/*.pyc"])
    )


@app.function(
    image=image, gpu="A100-80GB:4", cpu=8, memory=65536, timeout=600, volumes={"/cache": cache}
)
def benchmark(files, payload, provenance, repeats, sample_interval, thresholds, engine_config):
    from utils.benchmark import run_benchmark
    from distengine.metric import LatencyLimits
    from utils.freeze_workload import MODEL_ID, REVISION

    manifest = json.loads(files["manifest.json"])
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise ValueError("Frozen inputs do not match the pinned model")
    report = run_benchmark(
        payload,
        repeats=repeats,
        interval_s=sample_interval,
        limits=LatencyLimits(**thresholds),
        provenance=provenance,
        frozen_files=files,
        **engine_config,
    )
    cache.commit()
    return report


@app.local_entrypoint()
def main(
    workload_dir: str = "workloads/baseline",
    num_requests: int = 256,
    output_tokens: int = 64,
    batch_sizes: str = "64",
    repeats: int = 3,
    max_run_seconds: float = 90.0,
    num_blocks: int = 2048,
    stealing: bool = True,
    output: str = "results/capacity-final",
):
    """Measure a simultaneous backlog, reusing four GPU actors across the sweep."""
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from utils.benchmark_io import frozen_inputs, source_identity, save_report
    from utils.capacity import make_backlog, summarize_capacity

    batches = [int(value) for value in batch_sizes.split(",")]
    if not batches or len(set(batches)) != len(batches) or min(batches) <= 0:
        raise ValueError("Provide distinct positive batch sizes")
    if (
        repeats <= 0
        or num_blocks <= 0
        or not math.isfinite(max_run_seconds)
        or max_run_seconds <= 0
    ):
        raise ValueError("Repeats, KV blocks and measured deadline must be positive")
    if repeats * len(batches) * max_run_seconds > 300:
        raise ValueError("Keep the measured sweep budget at most 300s to allow setup within 600s")
    if Path(output).exists():
        raise ValueError("Choose a new output directory")
    files = frozen_inputs(root, workload_dir)
    payload = make_backlog(json.loads(files["mixed-v1.json"]), num_requests, output_tokens)
    print(
        f"Capacity: 4 A100-80GB GPUs, float32/nccl; {num_requests} simultaneous requests; "
        f"{output_tokens} output tokens; batches {batches}; stealing={stealing}"
    )
    print(
        "GPU runs incur charges. Setup/reference/warm-up are excluded from measured throughput; "
        "600s overall timeout includes all work. Partial runs are not full-drain results."
    )
    report = benchmark.remote(
        files,
        payload,
        source_identity(root),
        repeats,
        0.5,
        dict(ttft_s=2.0, tpot_s=0.1, e2e_s=10.0),
        dict(
            prefill_workers=2,
            decode_workers=2,
            stealing_enabled=stealing,
            batch_sizes=batches,
            decode_batch_size=batches[0],
            num_blocks=num_blocks,
            block_size=16,
            max_held_requests=4,
            warmup_requests=4,
            gpu_family="A100",
            min_gpu_memory_gib=75,
            max_run_s=max_run_seconds,
        ),
    )
    report["capacity_workload"] = dict(
        arrivals="all at zero",
        num_requests=num_requests,
        output_tokens=output_tokens,
        eos_disabled=True,
        source_workload=workload_dir,
    )
    report["capacity_summary"] = summarize_capacity(report)
    save_report(report, output)
    print("Batch | completed passes | mean observed output tokens/s")
    for row in report["capacity_summary"]:
        print(
            f"{row['batch_size']:5} | {row['complete_passes']}/{row['passes']} | "
            f"{row['observed_tokens_per_s']:.2f}"
        )
    print(f"Report saved: {output}/baseline.json")
    if report["correctness_failed"]:
        raise RuntimeError("Output mismatch: diagnostic report saved")
