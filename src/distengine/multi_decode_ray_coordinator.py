"""Ray orchestration for multiple prefill and multiple decode actors."""

import ray

from .multi_decode_coordinator import _MultiDecodeScheduling
from .multi_prefill_ray_coordinator import MultiPrefillRayCoordinator


class MultiDecodeRayCoordinator(_MultiDecodeScheduling, MultiPrefillRayCoordinator):
    def __init__(self, sources, targets):
        sources = list(sources)
        self.targets = list(targets)
        if not self.targets:
            raise ValueError("At least one decode actor is required")
        super().__init__(sources, self.targets[0])
        extra = ray.get([target.ready.remote() for target in self.targets[1:]])
        self._target_metadata = [self._metadata[-1], *extra]
        all_infos = [*self._metadata, *extra]
        if len({info["actor_id"] for info in all_infos}) != len(all_infos):
            raise ValueError("All actors must be distinct")
        for info in all_infos:
            if info["closed"]:
                raise RuntimeError("Worker is closed")
            if info["active_request_ids"] or info["num_free_blocks"] != info["num_blocks"]:
                raise ValueError("Coordinator requires fresh actors")
        if any(info["role"] != "DecodeWorker" for info in self._target_metadata) or any(
            info["role"] != "PrefillWorker" for info in self._metadata[:-1]
        ):
            raise ValueError("Actors must match their prefill/decode roles")
        self._init_decode()

    def _submission_targets(self):
        return self._target_metadata

    def _check_open(self):
        infos = super()._check_open()
        extra = ray.get([target.ready.remote() for target in self.targets[1:]])
        if any(info["closed"] for info in extra):
            raise RuntimeError("Worker is closed")
        # Preserve the parent metadata layout (sources, first target).
        return infos

    def _ensure_all_open(self):
        self._check_open()

    @staticmethod
    def _dispatch(worker, method, *args, **kwargs):
        return getattr(worker, method).remote(*args, **kwargs)

    @staticmethod
    def _resolve(ref):
        return ray.get(ref)

    def _call(self, worker, method, *args, **kwargs):
        return self._resolve(self._dispatch(worker, method, *args, **kwargs))

    def _info(self, worker):
        return self._call(worker, "ready")

    def _export(self, source, request_id):
        # Ray passes the payload object directly to the destination actor.
        return source.export_request.remote(request_id)

    def _prefill_all(self):
        self._prefill_sources()

    def _has_held(self):
        return any(item is not None for item in ray.get([
            source.peek_ready.remote() for source in self.sources
        ]))
