"""Run stage 10 on one Modal GPU: modal run src/Modal/run_engine.py."""

import json
from pathlib import Path

import modal


app = modal.App("distengine-stage-10")
cache = modal.Volume.from_name("distengine-hf-cache", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11")
# Modal reimports this module as /root/run_engine.py in the container.
# Only the local process can resolve and upload the project checkout.
if modal.is_local():
    project_root = Path(__file__).resolve().parents[2]
    image = (
        image.uv_sync(str(project_root))
        .env({"PYTHONPATH": "/app/src", "HF_HOME": "/cache/huggingface"})
        .add_local_dir(project_root / "src", "/app/src",
                       ignore=["**/__pycache__/**", "**/*.pyc"])
    )


@app.function(image=image, gpu="T4", memory=16384, timeout=1800,
              volumes={"/cache": cache})
def run_engine(prompt: str, max_new_tokens: int) -> dict:
    import platform

    import torch
    import transformers
    from transformers import AutoTokenizer

    from distengine.engine import LLMEngine
    from generation import generate_greedy
    from load_model import MODEL_ID, REVISION, load_model
    from verify_engine import check_empty, main as verify_cpu

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Stage 10 requires exactly one visible NVIDIA CUDA GPU")
    device = torch.device("cuda:0")
    # Preserve the float32 correctness baseline used in earlier stages.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verify_cpu()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = load_model(device)
    if any(p.device != device for p in model.parameters()):
        raise RuntimeError("All model parameters must be on cuda:0")
    prompts = [
        "What is the capital of France?",
        "Explain why the sky is blue in two sentences.",
        "Complete this sentence: The best way to learn programming is",
    ]
    if prompt not in prompts:
        prompts.append(prompt)
    with torch.inference_mode():
        inputs = []
        limits = []
        for text in prompts:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs.append(tokenizer(formatted, add_special_tokens=False,
                                    return_tensors="pt")["input_ids"].to(device))
            limits.append(max_new_tokens if text == prompt else 32)
        expected = [generate_greedy(model, ids, limit, tokenizer.eos_token_id)[0].tolist()
                    for ids, limit in zip(inputs, limits)]
        block_size = 16
        num_blocks = max((ids.shape[1] + limit - 1 + block_size - 1) // block_size
                         for ids, limit in zip(inputs, limits))
        engine = LLMEngine(model, num_blocks, block_size)
        assert engine.storage.k_cache.device == device
        requests = [engine.add_request(str(i), ids[0].tolist(), limit, tokenizer.eos_token_id)
                    for i, (ids, limit) in enumerate(zip(inputs, limits))]
        assert engine.run_until_complete() == requests
        results = []
        for text, request, baseline in zip(prompts, requests, expected):
            assert request.token_ids == baseline, f"CUDA baseline mismatch: {text}"
            result = {
                "prompt": text,
                "generated_token_ids": request.generated_token_ids,
                "text": tokenizer.decode(request.generated_token_ids, skip_special_tokens=True),
                "finish_reason": request.reason,
            }
            results.append(result)
            print(f"PASS: CUDA engine exact baseline agreement: {text}\n{result['text']}", flush=True)
        check_empty(engine)
        # Exercise the same engine again after its block pool is released.
        repeated = engine.add_request("repeat", inputs[0][0].tolist(), limits[0], tokenizer.eos_token_id)
        assert engine.run_until_complete() == [repeated]
        assert repeated.token_ids == expected[0]
        check_empty(engine)
        torch.cuda.synchronize()
    cache.commit()
    return {
        "stage": 10, "status": "passed", "model": MODEL_ID, "revision": REVISION,
        "device": str(device), "gpu": torch.cuda.get_device_name(0), "gpu_count": 1,
        "dtype": str(next(model.parameters()).dtype), "python": platform.python_version(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "results": results,
    }


@app.local_entrypoint()
def main(prompt: str = "What is the capital of France?", max_new_tokens: int = 32,
         output: str = "results/stage10-modal.json"):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    result = run_engine.remote(prompt, max_new_tokens)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Stage 10 single-GPU checks passed. Results saved to {destination}")
