"""TinyLlama baseline and work-stealing comparisons on four A100s."""

import math
import json
from pathlib import Path

import modal

app = modal.App("distengine-decode-profiler")
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
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=32768,
    timeout=600,
    volumes={"/cache": cache},
)
def measure(prompts, config):
    import torch
    from utils.load_model import load_model
    from utils.profiller import profile_decode

    artifacts = profile_decode(load_model(torch.device("cuda:0")), prompts, **config)
    cache.commit()
    return artifacts


@app.local_entrypoint()
def main(
    workload_dir: str = "workloads/baseline",
    batch_size: int = 64,
    warmup_steps: int = 4,
    timing_steps: int = 16,
    trace_steps: int = 2,
    num_blocks: int = 2048,
    output: str = "results/decode-profile",
):
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from utils.benchmark_io import frozen_inputs, source_identity

    if Path(output).exists():
        raise ValueError("Choose a new output directory")
    config = dict(
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        timing_steps=timing_steps,
        trace_steps=trace_steps,
        num_blocks=num_blocks,
    )
    if any(value <= 0 for value in config.values()) or trace_steps > timing_steps:
        raise ValueError("Use positive sizes and trace_steps <= timing_steps")
    files = frozen_inputs(root, workload_dir)
    payload = json.loads(files["mixed-v1.json"])
    prompts = [item["prompt_token_ids"] for item in payload["requests"]]
    provenance = source_identity(root)
    print("One A100-80GB; float32; isolated decode microprofile. GPU runs incur charges.")
    artifacts = measure.remote(prompts, config)
    report = json.loads(artifacts["report.json"])
    report["provenance"] = provenance
    artifacts["report.json"] = json.dumps(report, indent=2).encode()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    for name, data in artifacts.items():
        (directory / name).write_bytes(data)
    print(artifacts["operators.txt"].decode())
    print(f"Profile saved locally: {output}")
