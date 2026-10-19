"""Stage 21: repeated four-T4 TinyLlama baseline with stealing disabled."""

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


@app.function(image=image, gpu="T4:4", cpu=8, memory=65536, timeout=3600,
              volumes={"/cache": cache})
def benchmark(files, provenance, repeats, sample_interval, thresholds):
    from benchmark_stage21 import run_benchmark
    from distengine.metric import LatencyLimits
    from freeze_workload import MODEL_ID, REVISION
    manifest = json.loads(files["manifest.json"])
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise ValueError("Frozen inputs do not match the pinned model")
    report = run_benchmark(json.loads(files["mixed-v1.json"]), repeats=repeats,
                           interval_s=sample_interval, limits=LatencyLimits(**thresholds),
                           provenance=provenance, frozen_files=files)
    cache.commit()
    return report


@app.local_entrypoint()
def main(repeats: int = 3, sample_interval: float = .25,
         ttft_limit: float = 2.0, tpot_limit: float = .1, e2e_limit: float = 10.0,
         output: str = "results/stage21-baseline"):
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from benchmark_io import frozen_inputs, source_identity, save_report
    if Path(output).exists():
        raise ValueError("Choose a new output directory to preserve earlier runs")
    if repeats <= 0 or sample_interval <= 0:
        raise ValueError("Repeats and sample interval must be positive")
    thresholds = dict(ttft_s=ttft_limit, tpot_s=tpot_limit, e2e_s=e2e_limit)
    if any(not math.isfinite(v) or v <= 0 for v in thresholds.values()):
        raise ValueError("Goodput limits must be finite and positive")
    files = frozen_inputs(root)
    report = benchmark.remote(files, source_identity(root), repeats, sample_interval, thresholds)
    save_report(report, output)
    print(f"Baseline saved: {output}/baseline.json")
