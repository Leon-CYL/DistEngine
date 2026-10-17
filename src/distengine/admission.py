from typing import cast

import ray
from ray import ObjectRef

from .workers import PrefillWorker, DecodeWorker

def pull_one_ray(source, target) -> dict | None:
    """Run one serial admission attempt through Ray actors."""
    # Actor handles are dynamic; specify single-object refs so ray.get's
    # overloads do not infer a list result in editors.
    descriptor_ref = cast(ObjectRef[dict | None], source.peek_ready.remote())
    descriptor = ray.get(descriptor_ref)

    if descriptor is None:
        return None

    capacity_ref = cast(
        ObjectRef[bool],
        target.can_admit.remote(
            descriptor["num_prompt_tokens"],
            descriptor["max_new_tokens"],
        )
    )
    can_admit = ray.get(capacity_ref)

    if not can_admit:
        return None

    request_id = descriptor["request_id"]
    payload_ref = source.export_request.remote(request_id)

    result_ref = cast(
        ObjectRef[dict],
        target.import_request.remote(
            payload_ref,
            reserve_full=True,
        )
    )
    result = ray.get(result_ref)

    ray.get(source.release.remote(request_id))
    return result

def pull_one(
    source: PrefillWorker,
    target: DecodeWorker,
) -> dict | None:
    """Admit the oldest held request if decode capacity is available."""
    descriptor = source.peek_ready()

    if descriptor is None:
        return None

    if not target.can_admit(
        descriptor["num_prompt_tokens"],
        descriptor["max_new_tokens"],
    ):
        return None

    request_id = descriptor["request_id"]
    payload = source.export_request(request_id)

    result = target.import_request(
        payload,
        reserve_full=True,
    )

    source.release(request_id)
    return result
