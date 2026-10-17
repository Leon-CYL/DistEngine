"""Stage 12: verify separate GPU placement and independent worker execution."""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import sys
from typing import cast

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
from ray.actor import ActorHandle, ActorProxy
import torch

from distengine.ray_workers import (
    PrefillActor, PrefillActorImpl, DecodeActor, DecodeActorImpl, build_test_model,
)
from generation import generate_greedy


def check_placement(infos):
    first, second = infos
    assert first["pid"] != second["pid"]
    assert first["pid"] != os.getpid() and second["pid"] != os.getpid()
    assert first["node_id"] == second["node_id"]
    first_ids = set(first["ray_gpu_ids"])
    second_ids = set(second["ray_gpu_ids"])
    assert len(first_ids) == len(second_ids) == 1
    assert first_ids.isdisjoint(second_ids)
    for info in infos:
        assert info["device"] == info["kv_device"] == "cuda:0"
        assert info["visible_gpu_count"] == 1
        assert info["python_executable"] == sys.executable


def check_released(info):
    assert not info["active_request_ids"]
    assert info["num_free_blocks"] == info["num_blocks"]


@torch.inference_mode()
def main(checkpoint: bool = False, transfer: bool = False, admission: bool = False) -> dict:
    if admission:
        transfer = True
    from distengine.admission import pull_one_ray
    if ray.is_initialized():
        raise RuntimeError("Run this verification with a fresh Ray runtime")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Stage 12 verification requires exactly two NVIDIA GPUs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)

    if checkpoint:
        from transformers import AutoTokenizer
        from load_model import MODEL_ID, REVISION, load_model

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        prompts = [
            "What is the capital of France?",
            "Explain why the sky is blue in two sentences.",
            "Complete this sentence: The best way to learn programming is",
        ]
        inputs = []
        for text in prompts:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs.append(tokenizer(formatted, add_special_tokens=False)["input_ids"])
        limit, eos = 32, tokenizer.eos_token_id
        model = load_model(torch.device("cuda:0"))
        model_id, revision = MODEL_ID, REVISION
    else:
        prompts = ["three-token prompt", "one-token prompt", "block-boundary prompt"]
        inputs = [[10, 20, 30], [40], [10, 20, 30, 40]]
        limit, eos = 4, -1
        model = build_test_model(0).to("cuda:0")
        model_id, revision = "deterministic-small-llama", None

    expected = [generate_greedy(model, torch.tensor([ids], device="cuda:0"),
                                limit, eos)[0].tolist() for ids in inputs]
    torch.cuda.synchronize()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    block_size = 4
    num_blocks = max((len(ids) + limit - 1 + block_size - 1) // block_size for ids in inputs)

    ray.init(address="local", num_cpus=2, num_gpus=2, include_dashboard=False,
             runtime_env={"py_executable": sys.executable})
    actors: list[ActorProxy[PrefillActorImpl] | ActorProxy[DecodeActorImpl]] = []
    try:
        prefill: ActorProxy[PrefillActorImpl] = PrefillActor.options(num_gpus=1).remote(
            num_blocks=num_blocks, block_size=block_size, device="cuda", checkpoint=checkpoint)
        actors.append(prefill)
        decode: ActorProxy[DecodeActorImpl] = DecodeActor.options(num_gpus=1).remote(
            num_blocks=num_blocks, block_size=block_size, device="cuda", checkpoint=checkpoint)
        actors.append(decode)
        infos = ray.get([prefill.ready.remote(), decode.ready.remote()], timeout=600)
        check_placement(infos)
        print("PASS: separate processes and GPUs; model and KV placement", flush=True)

        results = []
        admission_count = 0
        for index, (text, ids, baseline) in enumerate(zip(prompts, inputs, expected)):
            request_id = f"request-{index}"
            first = ray.get(prefill.prefill.remote(request_id, ids, limit, eos), timeout=180)
            if transfer:
                if first["status"] == "FINISHED":
                    result = first
                else:
                    if admission:
                        descriptor = ray.get(prefill.peek_ready.remote(), timeout=30)
                        assert descriptor == {"request_id": request_id,
                                              "num_prompt_tokens": len(ids),
                                              "max_new_tokens": limit}
                        required = (len(ids) + limit - 1 + block_size - 1) // block_size
                        # Occupy enough blocks that the FIFO head cannot fit.
                        occupied_tokens = (num_blocks - required) * block_size + 1
                        ray.get(decode.prepare_local_fixture.remote(
                            "occupied", [ids[0]] * occupied_tokens, 2, -1), timeout=180)
                        before_source, before_target = ray.get(
                            [prefill.ready.remote(), decode.ready.remote()], timeout=30)
                        assert pull_one_ray(prefill, decode) is None
                        after_source, after_target = ray.get(
                            [prefill.ready.remote(), decode.ready.remote()], timeout=30)
                        assert before_source == after_source and before_target == after_target
                        assert ray.get(prefill.peek_ready.remote(), timeout=30) == descriptor
                        ray.get(decode.release.remote("occupied"), timeout=30)
                        result = pull_one_ray(prefill, decode)
                        assert result is not None and result["request_id"] == request_id
                        assert ray.get(prefill.peek_ready.remote(), timeout=30) is None
                        reserved = ray.get(decode.ready.remote(), timeout=30)
                        assert reserved["num_free_blocks"] == num_blocks - required
                        admission_count += 1
                    else:
                        payload_ref = prefill.export_request.remote(request_id)
                        result = ray.get(decode.import_request.remote(payload_ref), timeout=180)
                        # ACK: destination copied every layer before source release.
                        ray.get(prefill.release.remote(request_id), timeout=30)
            else:
                # Stage 12 keeps its independent execution check.
                result = ray.get(decode.prepare_local_fixture.remote(
                    request_id, ids, limit, eos), timeout=180)
                if first["status"] != "FINISHED":
                    ray.get(prefill.release.remote(request_id), timeout=30)
            assert first["generated_token_ids"] == baseline[len(ids):len(ids) + 1]
            while result["status"] != "FINISHED":
                result = ray.get(decode.step.remote(request_id), timeout=180)
                if admission and result["status"] != "FINISHED":
                    current = ray.get(decode.ready.remote(), timeout=30)
                    assert current["num_free_blocks"] == num_blocks - required
            assert result["prompt_token_ids"] + result["generated_token_ids"] == baseline
            for info in ray.get([actor.ready.remote() for actor in actors], timeout=30):
                check_released(info)
            results.append({"prompt": text, **result})
            mode = ("decode-pull denial/retry and reserved handoff" if admission else
                    "end-to-end KV handoff" if transfer else "independent prefill/decode")
            print(f"PASS: GPU {mode}, baseline agreement and release: {text}", flush=True)

        # Both roles must release unfinished work when closed.
        ray.get(prefill.prefill.remote("unfinished", inputs[0], limit, -1), timeout=180)
        if transfer:
            payload_ref = prefill.export_request.remote("unfinished")
            ray.get(decode.import_request.remote(payload_ref, reserve_full=admission), timeout=180)
        else:
            ray.get(decode.prepare_local_fixture.remote("unfinished", inputs[0], limit, -1), timeout=180)
        ray.get([actor.close.remote() for actor in actors], timeout=30)
        for info in ray.get([actor.ready.remote() for actor in actors], timeout=30):
            assert info["closed"]
            check_released(info)
        if admission:
            assert admission_count > 0, "No unfinished request exercised decode-pull admission"
        print("PASS: GPU worker close releases unfinished requests", flush=True)
        return {
            "stage": 15 if admission else 13 if transfer else 12,
            "status": ("decode-pull-admission-passed" if admission else
                       "end-to-end-transfer-passed" if transfer else "placement-and-independent-execution-passed"),
            "decode_pull_verified": admission,
            "admitted_requests": admission_count,
            "kv_transfer_verified": transfer, "checkpoint": checkpoint,
            "model": model_id, "revision": revision,
            "python": platform.python_version(), "torch": torch.__version__,
            "ray": ray.__version__, "actors": infos, "results": results,
        }
    finally:
        for actor in actors:
            try:
                ray.get(actor.close.remote(), timeout=10)
            except Exception as exc:
                print(f"Cleanup warning: {exc}", flush=True)
            finally:
                ray.kill(cast(ActorHandle, actor), no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/stage12-placement.json"))
    args = parser.parse_args()
    report = main(checkpoint=args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")
