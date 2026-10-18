"""Stage 16: queued backpressure on two Modal T4 GPUs."""

import json
from pathlib import Path

import modal


app = modal.App("distengine-stage-16")
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


@app.function(image=image, gpu="T4:2", cpu=4, memory=32768, timeout=1800,
              volumes={"/cache": cache})
def verify(checkpoint: bool) -> dict:
    from verify_backpressure import main as check_cpu
    from verify_gpu_backpressure import main

    check_cpu()
    result = main(checkpoint=checkpoint)
    cache.commit()
    return result


@app.local_entrypoint()
def main(checkpoint: bool = True, output: str = "results/stage16-backpressure.json"):
    result = verify.remote(checkpoint)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Stage 16 two-GPU backpressure passed: {destination}")
