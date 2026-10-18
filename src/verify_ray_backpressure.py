"""Real Ray CPU saturation and automatic resumption through RayCoordinator."""

import os
import sys

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
import torch

from distengine.ray_coordinator import RayCoordinator
from distengine.ray_workers import PrefillActor, DecodeActor, build_test_model
from generation import generate_greedy


def main():
    ray.init(address="local", num_cpus=2, include_dashboard=False,
             runtime_env={"py_executable": sys.executable})
    actors = []
    try:
        source = PrefillActor.remote(num_blocks=4, max_held_requests=1)
        actors.append(source)
        target = DecodeActor.remote(num_blocks=2)
        actors.append(target)
        ray.get([a.ready.remote() for a in actors], timeout=60)
        c = RayCoordinator(source, target)
        for rid in ("a", "b", "c"):
            c.submit(rid, [10, 20, 30], 4, -1)
        assert c.step() == []
        assert c.step() == []
        assert c.step() == []
        assert [s.request_id for s in c.waiting] == ["c"]
        assert list(c.decode_ready) == ["a"]
        assert ray.get(source.peek_ready.remote(), timeout=30)["request_id"] == "b"
        results = []
        for _ in range(30):
            if not c.has_pending():
                break
            results.extend(c.step())
        assert not c.has_pending()
        assert [r["request_id"] for r in results] == ["a", "b", "c"]
        model = build_test_model(0)
        expected = generate_greedy(model, torch.tensor([[10, 20, 30]]), 4, -1)[0].tolist()
        assert all(r["prompt_token_ids"] + r["generated_token_ids"] == expected for r in results)
        assert c.drain_completed() == []
        for info in ray.get([a.ready.remote() for a in actors], timeout=30):
            assert not info["active_request_ids"]
            assert info["num_free_blocks"] == info["num_blocks"]
        print("PASS: real Ray upstream waiting, decode progress, automatic resumption, outputs and cleanup")
    finally:
        for actor in actors:
            try:
                ray.get(actor.close.remote(), timeout=10)
            finally:
                ray.kill(actor, no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
