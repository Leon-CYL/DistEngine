import os
import sys
from typing import cast

# This verification starts actors on this machine only. Reuse the installed
# project environment instead of uv rebuilding it in every Ray session.
# Ray reads this setting during import, so it must precede `import ray`.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
from ray.actor import ActorHandle, ActorProxy
from ray.exceptions import RayTaskError
import torch

from distengine.ray_workers import (
    PrefillActor,
    PrefillActorImpl,
    DecodeActor,
    DecodeActorImpl,
    build_test_model,
)
from generation import generate_greedy


def main():
    prompt = [10, 20, 30]
    limit = 4
    eos = -1

    model = build_test_model(seed=0)
    expected = generate_greedy(
        model,
        torch.tensor([prompt]),
        limit,
        eos,
    )[0].tolist()

    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        runtime_env={"py_executable": sys.executable},
    )

    actors: list[ActorProxy[PrefillActorImpl] | ActorProxy[DecodeActorImpl]] = []

    try:
        prefill: ActorProxy[PrefillActorImpl] = PrefillActor.remote()
        actors.append(prefill)

        decode: ActorProxy[DecodeActorImpl] = DecodeActor.remote()
        actors.append(decode)

        # Submit both checks before waiting.
        infos = ray.get(
            [
                prefill.ready.remote(),
                decode.ready.remote(),
            ],
            timeout=60,
        )

        driver_pid = os.getpid()
        prefill_pid = infos[0]["pid"]
        decode_pid = infos[1]["pid"]

        assert prefill_pid != driver_pid
        assert decode_pid != driver_pid
        assert prefill_pid != decode_pid
        assert all(info["python_executable"] == sys.executable for info in infos)

        print("PASS: actors run in separate processes")

        first = ray.get(
            prefill.prefill.remote("prefill-A", prompt, limit, eos),
            timeout=30,
        )

        assert first["generated_token_ids"] == [
            expected[len(prompt)]
        ]

        ray.get(prefill.release.remote("prefill-A"), timeout=30)

        info = ray.get(prefill.ready.remote(), timeout=30)
        assert info["active_request_ids"] == []
        assert info["num_free_blocks"] == info["num_blocks"]

        print("PASS: remote prefill and explicit release")

        # Until stage 13, prepare KV inside the decode actor.
        result = ray.get(
            decode.prepare_local_fixture.remote(
                "decode-A", prompt, limit, eos
            ),
            timeout=30,
        )

        while result["status"] != "FINISHED":
            result = ray.get(
                decode.step.remote("decode-A"),
                timeout=30,
            )

        actual = (
            result["prompt_token_ids"]
            + result["generated_token_ids"]
        )

        assert actual == expected
        assert result["reason"] == "length"

        info = ray.get(decode.ready.remote(), timeout=30)
        assert info["active_request_ids"] == []
        assert info["num_free_blocks"] == info["num_blocks"]

        print("PASS: remote decode, baseline agreement, block release")

        # Verify that a worker exception reaches the driver.
        try:
            ray.get(decode.step.remote("missing"), timeout=30)
        except RayTaskError as exc:
            assert isinstance(exc.cause, KeyError)
        else:
            raise AssertionError("Expected a remote KeyError")

        # An ordinary method error should not kill the actor.
        info = ray.get(decode.ready.remote(), timeout=30)
        assert not info["closed"]

        print("PASS: remote exception propagation and actor survival")

        # Leave work active to verify close() releases it.
        ray.get(
            prefill.prefill.remote("unfinished", prompt, limit, eos),
            timeout=30,
        )

        ray.get(
            [actor.close.remote() for actor in actors],
            timeout=30,
        )

        closed_infos = ray.get(
            [actor.ready.remote() for actor in actors],
            timeout=30,
        )

        for info in closed_infos:
            assert info["closed"]
            assert info["active_request_ids"] == []
            assert info["num_free_blocks"] == info["num_blocks"]

        print("PASS: close releases unfinished requests")
        print("All Ray worker checks passed.")

    finally:
        # Try application cleanup for every actor, even after a failure.
        for actor in actors:
            try:
                ray.get(actor.close.remote(), timeout=10)
            except Exception as exc:
                print(f"Cleanup warning: {exc}")
            finally:
                # ActorProxy includes a class type for editor method inference;
                # values returned by .remote() are ActorHandle objects at runtime.
                ray.kill(cast(ActorHandle, actor), no_restart=True)

        ray.shutdown()


if __name__ == "__main__":
    main()
