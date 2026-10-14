"""CPU KV handoff correctness, validation, failure atomicity, and real Ray dispatch."""

import os
import sys
from typing import cast
from unittest.mock import patch

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
from ray.actor import ActorHandle
import torch

from distengine.paged_kv_cache import PagedKVCache
from distengine.ray_workers import PrefillActor, DecodeActor, build_test_model
from distengine.workers import PrefillWorker, DecodeWorker
from generation import generate_greedy
from verify_runner import raises


def empty(worker):
    assert not worker.requests and not worker.runner.caches
    assert not worker.runner.block_manager.request_blocks
    assert worker.info()["num_free_blocks"] == worker.info()["num_blocks"]


def finish(worker, request_id):
    while request_id in worker.requests:
        result = worker.step(request_id)
    return result


def check_local():
    torch.set_num_threads(1)
    source = PrefillWorker(build_test_model(0), 8, 4)
    # Different block sizes and occupied blocks force independent destination maps.
    target = DecodeWorker(build_test_model(0), 16, 2)
    try:
        target.prepare_local_fixture("other", [50], 2, -1)
        for index, prompt in enumerate(([10], [10, 20, 30, 40], [10, 20, 30, 40, 50])):
            rid = f"handoff-{index}"
            source.prefill(rid, prompt, 4, -1)
            payload = source.export_request(rid)
            assert payload["keys"].device.type == "cpu"
            assert payload["keys"].shape[3] == len(prompt)
            with patch.object(target.runner.model, "forward", side_effect=AssertionError("No recompute")):
                target.import_request(payload)
            for layer in range(2):
                k, v = target.runner.caches[rid].read(layer)
                assert torch.equal(k, payload["keys"][layer])
                assert torch.equal(v, payload["values"][layer])
            raises(ValueError, lambda: target.import_request(payload))
            source.release(rid)  # Only after successful destination import.
            source.prefill("reuse", [60], 1, -1)
            # Source reuse and caller mutation must not alter imported KV.
            payload["keys"].zero_()
            payload["values"].zero_()
            result = finish(target, rid)
            baseline = generate_greedy(target.runner.model, torch.tensor([prompt]), 4, -1)[0].tolist()
            assert result["prompt_token_ids"] + result["generated_token_ids"] == baseline
        finish(target, "other")
        empty(source)
        empty(target)

        source.prefill("validation", [10, 20, 30], 4, -1)
        good = source.export_request("validation")
        for field, value in [
            ("version", 2), ("model_identity", "wrong"), ("cache_length", 2),
            ("generated_token_ids", []), ("generated_token_ids", [True]),
            ("prompt_token_ids", [64]), ("max_new_tokens", True),
            ("request_id", []), ("keys", good["keys"].double()),
            ("keys", good["keys"][:, :, :, :-1]),
            ("values", torch.full_like(good["values"], float("nan"))),
        ]:
            raises(ValueError, lambda: target.import_request({**good, field: value}))
            empty(target)
        wrong_model = DecodeWorker(build_test_model(1), 8, 4)
        try:
            raises(ValueError, lambda: wrong_model.import_request(good))
            empty(wrong_model)
        finally:
            wrong_model.close()
        # Temporary capacity failure must preserve source ownership.
        small = DecodeWorker(build_test_model(0), 2, 4)
        try:
            small.prepare_local_fixture("occupied", [10] * 7, 2, -1)
            raises(RuntimeError, lambda: small.import_request(good))
            assert "validation" in source.requests
            small.release("occupied")
            small.import_request(good)
            finish(small, "validation")
            empty(small)
        finally:
            small.close()

        # Fail after first layer copy. Destination must publish no partial cache.
        append = PagedKVCache.append
        def fail_second(cache, layer, k, v):
            if layer == 1:
                raise RuntimeError("Injected import copy failure")
            return append(cache, layer, k, v)
        with patch.object(PagedKVCache, "append", fail_second):
            raises(RuntimeError, lambda: target.import_request(good))
        empty(target)
        target.import_request(good)
        source.release("validation")
        finish(target, "validation")
        empty(target)

        first = source.prefill("finished", [10], 1, -1)
        assert first["status"] == "FINISHED"
        raises(KeyError, lambda: source.export_request("finished"))
        eos = first["generated_token_ids"][0]
        assert source.prefill("eos", [10], 4, eos)["reason"] == "eos"
        empty(source)
        print("PASS: CPU handoff, remapping, no recompute, isolation, validation, retry and rollback")
    finally:
        source.close()
        target.close()


def check_ray():
    ray.init(address="local", num_cpus=2, include_dashboard=False,
             runtime_env={"py_executable": sys.executable})
    actors = []
    try:
        source = PrefillActor.remote()
        actors.append(source)
        target = DecodeActor.remote()
        actors.append(target)
        ray.get([source.ready.remote(), target.ready.remote()], timeout=60)
        prompt = [10, 20, 30, 40, 50]
        first = ray.get(source.prefill.remote("ray-handoff", prompt, 4, -1), timeout=30)
        assert first["status"] == "RUNNING"
        payload_ref = source.export_request.remote("ray-handoff")
        result = ray.get(target.import_request.remote(payload_ref), timeout=30)
        ray.get(source.release.remote("ray-handoff"), timeout=30)
        while result["status"] != "FINISHED":
            result = ray.get(target.step.remote("ray-handoff"), timeout=30)
        model = build_test_model(0)
        expected = generate_greedy(model, torch.tensor([prompt]), 4, -1)[0].tolist()
        assert result["prompt_token_ids"] + result["generated_token_ids"] == expected
        for info in ray.get([actor.ready.remote() for actor in actors], timeout=30):
            assert not info["active_request_ids"]
            assert info["num_free_blocks"] == info["num_blocks"]
        print("PASS: real Ray CPU payload handoff, continuation and cleanup")
    finally:
        for actor in actors:
            try:
                ray.get(actor.close.remote(), timeout=10)
            finally:
                ray.kill(cast(ActorHandle, actor), no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    check_local()
    check_ray()
