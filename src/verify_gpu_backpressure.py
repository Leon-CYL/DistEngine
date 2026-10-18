"""Two-GPU RayCoordinator saturation, automatic resumption, and output checks."""

import gc
import os
import platform
import sys

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
import torch

from distengine.ray_coordinator import RayCoordinator
from distengine.ray_workers import PrefillActor, DecodeActor, build_test_model
from generation import generate_greedy
from verify_gpu_workers import check_placement, check_released


@torch.inference_mode()
def main(checkpoint: bool = True) -> dict:
    if ray.is_initialized():
        raise RuntimeError("Run with a fresh Ray runtime")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Verification requires exactly two NVIDIA GPUs")
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
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True)
            inputs.append(tokenizer(formatted, add_special_tokens=False)["input_ids"])
        limit, eos = 32, tokenizer.eos_token_id
        model = load_model(torch.device("cuda:0"))
        model_id, revision = MODEL_ID, REVISION
    else:
        texts = ["three-token prompt", "one-token prompt", "block-boundary prompt"]
        inputs = [[10, 20, 30], [40], [10, 20, 30, 40]]
        # Reserve at least two blocks even for the one-token prompt. With
        # limit=4 it needed one block and two requests fit the shared pool,
        # so that case never exercised the intended admission pressure.
        limit, eos = 8, -1
        model = build_test_model(0).to("cuda:0")
        model_id, revision = "deterministic-small-llama", None
    baselines = [generate_greedy(model, torch.tensor([ids], device="cuda:0"),
                                 limit, eos)[0].tolist() for ids in inputs]
    torch.cuda.synchronize()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    block_size = 4
    num_blocks = max((len(ids) + limit - 2) // block_size + 1 for ids in inputs)
    # Each admitted request must prevent the next request from fitting.
    assert all(2 * ((len(ids) + limit - 2) // block_size + 1) > num_blocks for ids in inputs)
    ray.init(address="local", num_cpus=2, num_gpus=2, include_dashboard=False,
             runtime_env={"py_executable": sys.executable})
    actors = []
    try:
        source = PrefillActor.options(num_gpus=1).remote(
            num_blocks=num_blocks, block_size=block_size, device="cuda",
            checkpoint=checkpoint, max_held_requests=1)
        actors.append(source)
        target = DecodeActor.options(num_gpus=1).remote(
            num_blocks=num_blocks, block_size=block_size, device="cuda", checkpoint=checkpoint)
        actors.append(target)
        infos = ray.get([a.ready.remote() for a in actors], timeout=600)
        check_placement(infos)
        reports = []
        for index, (text, ids, baseline) in enumerate(zip(texts, inputs, baselines)):
            assert len(baseline) - len(ids) >= 4, "Need unfinished decode to demonstrate saturation"
            c = RayCoordinator(source, target)
            request_ids = [f"case-{index}-{suffix}" for suffix in ("a", "b", "c")]
            for rid in request_ids:
                c.submit(rid, ids, limit, eos)
            assert c.step() == []
            assert c.step() == []
            before = ray.get(source.ready.remote(), timeout=30)
            assert c.step() == []
            assert ray.get(source.ready.remote(), timeout=30) == before
            assert [s.request_id for s in c.waiting] == [request_ids[2]]
            assert list(c.decode_ready) == [request_ids[0]]
            assert ray.get(source.peek_ready.remote(), timeout=30)["request_id"] == request_ids[1]
            results = []
            ticks = 3
            for _ in range(3 * limit + 10):
                if not c.has_pending():
                    break
                results.extend(c.step())
                ticks += 1
            assert not c.has_pending(), "Coordinator failed to drain within tick bound"
            assert [r["request_id"] for r in results] == request_ids
            assert all(r["prompt_token_ids"] + r["generated_token_ids"] == baseline for r in results)
            assert c.drain_completed() == []
            for info in ray.get([a.ready.remote() for a in actors], timeout=30):
                check_released(info)
            assert ray.get(source.peek_ready.remote(), timeout=30) is None
            reports.append({"prompt": text, "saturation_verified": True,
                            "ticks": ticks, "results": results})
            print(f"PASS: GPU saturation, upstream waiting, automatic resumption, baseline and cleanup: {text}", flush=True)

        # Close while one request decodes, one is held, and one waits upstream.
        c = RayCoordinator(source, target)
        for suffix in ("a", "b", "c"):
            c.submit(f"unfinished-{suffix}", inputs[0], limit, -1)
        c.step()
        c.step()
        ray.get([a.close.remote() for a in actors], timeout=60)
        for info in ray.get([a.ready.remote() for a in actors], timeout=30):
            assert info["closed"]
            check_released(info)
        print("PASS: closing GPU workers releases held and decoding KV", flush=True)
        return {"stage": 16, "status": "backpressure-passed", "checkpoint": checkpoint,
                "model": model_id, "revision": revision, "actors": infos,
                "python": platform.python_version(), "torch": torch.__version__,
                "ray": ray.__version__, "reports": reports}
    finally:
        for actor in actors:
            try:
                ray.get(actor.close.remote(), timeout=10)
            except Exception as exc:
                print(f"Cleanup warning: {exc}", flush=True)
            finally:
                ray.kill(actor, no_restart=True)
        ray.shutdown()
