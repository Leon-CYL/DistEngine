"""Four-GPU correctness checkpoint: two prefill and two decode actors."""

import gc
import os
import platform
import sys

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
import torch

from distengine.ray_workers import PrefillActor, DecodeActor, build_test_model
from generation import generate_greedy
from verify_multi_decode_ray import exercise


@torch.inference_mode()
def main(checkpoint=True):
    if ray.is_initialized():
        raise RuntimeError("Run with a fresh Ray runtime")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("Verification requires exactly four NVIDIA GPUs")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if checkpoint:
        from transformers import AutoTokenizer
        from load_model import MODEL_ID, REVISION, load_model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        texts = ["What is the capital of France?",
                 "Explain why the sky is blue in two sentences.",
                 "Complete this sentence: The best way to learn programming is"]
        inputs = []
        for text in texts:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
            )
            inputs.append(tokenizer(formatted, add_special_tokens=False)["input_ids"])
        limit, eos = 32, tokenizer.eos_token_id
        model = load_model(torch.device("cuda:0"))
        model_id, revision = MODEL_ID, REVISION
    else:
        texts = ["three-token prompt", "one-token prompt", "four-token prompt"]
        inputs = [[10, 20, 30], [40], [10, 20, 30, 40]]
        limit, eos = 8, -1
        model = build_test_model(0).to("cuda:0")
        model_id, revision = "deterministic-small-llama", None
    baselines = [generate_greedy(model, torch.tensor([ids], device="cuda:0"), limit, eos)[0].tolist()
                 for ids in inputs]
    assert all(len(output) - len(ids) >= 4 for output, ids in zip(baselines, inputs))
    torch.cuda.synchronize()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    block_size = 4
    blocks = max((len(ids) + limit - 2) // block_size + 1 for ids in inputs)
    assert all(2 * ((len(ids) + limit - 2) // block_size + 1) > blocks for ids in inputs)
    ray.init(address="local", num_cpus=4, num_gpus=4, include_dashboard=False,
             runtime_env={"py_executable": sys.executable})
    actors = []
    try:
        for _ in range(2):
            actors.append(PrefillActor.options(num_gpus=1).remote(
                num_blocks=blocks, block_size=block_size, device="cuda", checkpoint=checkpoint,
                max_held_requests=1,
            ))
        sources = actors.copy()
        for _ in range(2):
            actors.append(DecodeActor.options(num_gpus=1).remote(
                num_blocks=blocks, block_size=block_size, device="cuda", checkpoint=checkpoint,
            ))
        infos = ray.get([a.ready.remote() for a in actors], timeout=600)
        assert len({i["pid"] for i in infos}) == 4
        assert len({i["node_id"] for i in infos}) == 1
        ids = [set(i["ray_gpu_ids"]) for i in infos]
        assert all(len(group) == 1 for group in ids) and len(set.union(*ids)) == 4
        for info in infos:
            assert info["pid"] != os.getpid() and info["python_executable"] == sys.executable
            assert info["device"] == info["kv_device"] == "cuda:0"
            assert info["visible_gpu_count"] == 1
        print("PASS: four separate processes and disjoint model/KV GPU assignments", flush=True)
        result = exercise(sources, actors[2:], inputs, limit, eos, baselines)
        return {"stage": 22, "execution_order": "before-stage-20", "status": "multi-decode-passed",
                "checkpoint": checkpoint, "model": model_id, "revision": revision,
                "actors": infos, "python": platform.python_version(), "torch": torch.__version__,
                "ray": ray.__version__, "prompts": texts, **result}
    finally:
        for actor in actors:
            try:
                ray.get(actor.close.remote(), timeout=10)
            finally:
                ray.kill(actor, no_restart=True)
        ray.shutdown()
