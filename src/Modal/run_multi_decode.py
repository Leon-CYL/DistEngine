"""Stage 22 moved before benchmarking: two prefill and two decode T4 GPUs."""

import json
from pathlib import Path

import modal

app = modal.App("distengine-stage-22")
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


@app.function(image=image, gpu="T4:4", cpu=8, memory=65536, timeout=1800,
              volumes={"/cache": cache})
def verify(checkpoint: bool) -> dict:
    from verify_multi_decode import main as ordinary_cpu
    from verify_multi_decode_ray import main as ray_cpu
    from verify_gpu_multi_decode import main
    ordinary_cpu()
    ray_cpu()
    result = main(checkpoint=checkpoint)
    cache.commit()
    return result


@app.local_entrypoint()
def main(checkpoint: bool = True, output: str = "results/stage22-multi-decode.json"):
    result = verify.remote(checkpoint)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Stage 22 four-GPU multi-decode passed: {destination}")
