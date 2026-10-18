"""Three-GPU multi-prefill queue isolation, admission, and output checks."""

import gc
import os
import platform
import sys

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
import torch

from distengine.multi_prefill_ray_coordinator import MultiPrefillRayCoordinator
from distengine.ray_workers import PrefillActor, DecodeActor, build_test_model
from generation import generate_greedy
from verify_gpu_workers import check_released


@torch.inference_mode()
def main(checkpoint: bool = True) -> dict:
    if ray.is_initialized():
        raise RuntimeError("Run with a fresh Ray runtime")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
        raise RuntimeError("Verification requires exactly three NVIDIA GPUs")
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
    ray.init(address="local", num_cpus=3, num_gpus=3, include_dashboard=False,
             runtime_env={"py_executable": sys.executable})
    actors = []
    try:
        sources = []
        for _ in range(2):
            actor = PrefillActor.options(num_gpus=1).remote(
                num_blocks=num_blocks, block_size=block_size, device="cuda",
                checkpoint=checkpoint, max_held_requests=1)
            sources.append(actor)
            actors.append(actor)
        target = DecodeActor.options(num_gpus=1).remote(
            num_blocks=num_blocks, block_size=block_size, device="cuda", checkpoint=checkpoint)
        actors.append(target)
        infos = ray.get([a.ready.remote() for a in actors], timeout=600)
        assert len({info["pid"] for info in infos}) == 3
        assert all(info["pid"] != os.getpid() for info in infos)
        assert len({info["node_id"] for info in infos}) == 1
        gpu_ids = [set(info["ray_gpu_ids"]) for info in infos]
        assert all(len(ids) == 1 for ids in gpu_ids)
        assert len(set.union(*gpu_ids)) == 3
        for info in infos:
            assert info["device"] == info["kv_device"] == "cuda:0"
            assert info["visible_gpu_count"] == 1
            assert info["python_executable"] == sys.executable
        print("PASS: three separate actor processes and disjoint GPU assignments", flush=True)
        c = MultiPrefillRayCoordinator(sources, target)
        # Two rounds of mixed prompts put three requests on each source.
        expected = {}
        descriptions = {}
        for i in range(6):
            case = i % len(inputs)
            rid = str(i)
            c.submit(rid, inputs[case], limit, eos)
            expected[rid] = baselines[case]
            descriptions[rid] = texts[case]
        assert [[s.request_id for s in q] for q in c.waiting] == [["0", "2", "4"], ["1", "3", "5"]]
        assert all(len(b) - len(ids) >= 4 for b, ids in zip(baselines, inputs))
        assert c.step() == []
        held = ray.get([s.peek_ready.remote() for s in sources], timeout=30)
        assert [d["request_id"] for d in held] == ["0", "1"]
        assert c.step() == []
        assert list(c.decode_ready) == ["1"]
        held = ray.get([s.peek_ready.remote() for s in sources], timeout=30)
        assert [d["request_id"] for d in held] == ["0", "3"]
        assert [[s.request_id for s in q] for q in c.waiting] == [["2", "4"], ["5"]]
        print("PASS: independent holding pools, source-1 admission and refill while source-0 remains held", flush=True)
        before = ray.get([s.ready.remote() for s in sources], timeout=30)
        assert c.step() == []
        assert ray.get([s.ready.remote() for s in sources], timeout=30) == before
        results = []
        ticks = 3
        for _ in range(6 * limit + 20):
            if not c.has_pending():
                break
            results.extend(c.step())
            ticks += 1
        assert not c.has_pending(), "Multi-prefill scheduler failed to drain"
        assert len(results) == 6 and len({r["request_id"] for r in results}) == 6
        for result in results:
            rid = result["request_id"]
            assert result["prompt_token_ids"] + result["generated_token_ids"] == expected[rid]
        assert c.drain_completed() == []
        for info in ray.get([a.ready.remote() for a in actors], timeout=30):
            check_released(info)
        assert ray.get([s.peek_ready.remote() for s in sources], timeout=30) == [None, None]
        print("PASS: all six mixed-prompt requests match baseline, complete once and release all KV", flush=True)
        # Both sources retain work while the target owns an unfinished request.
        for i in range(6):
            c.submit(f"unfinished-{i}", inputs[i % len(inputs)], limit, -1)
        c.step()
        c.step()
        ray.get([a.close.remote() for a in actors], timeout=60)
        for info in ray.get([a.ready.remote() for a in actors], timeout=30):
            assert info["closed"]
            check_released(info)
        print("PASS: three GPU workers close with unfinished held and decode work", flush=True)
        return {"stage": 17, "status": "multi-prefill-passed", "checkpoint": checkpoint,
                "model": model_id, "revision": revision, "actors": infos,
                "python": platform.python_version(), "torch": torch.__version__,
                "ray": ray.__version__, "ticks": ticks,
                "results": [{"prompt": descriptions[r["request_id"]], **r} for r in results]}
    finally:
        for actor in actors:
            try:
                ray.get(actor.close.remote(), timeout=10)
            except Exception as exc:
                print(f"Cleanup warning: {exc}", flush=True)
            finally:
                ray.kill(actor, no_restart=True)
        ray.shutdown()
