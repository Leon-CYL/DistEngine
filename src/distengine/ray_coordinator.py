from collections import deque
from typing import cast

import ray
from ray import ObjectRef

from .admission import pull_one_ray
from .coordinator import Submission


class RayCoordinator:
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self._metadata = self._check_open()
        
        for info in self._metadata:
            if info["active_request_ids"] or info["num_free_blocks"] != info["num_blocks"]:
                raise ValueError("Coordinator requires fresh workers")
            
        if ray.get(cast(ObjectRef[dict | None], source.peek_ready.remote())) is not None:
            raise ValueError("Coordinator requires an empty holding pool")
        
        self.waiting: deque[Submission] = deque()
        self.decode_ready: deque[str] = deque()
        self.seen_ids: set[str] = set()
        self.completed: deque[dict] = deque()

    def _check_open(self) -> list[dict]:
        infos = ray.get([
            cast(ObjectRef[dict], self.source.ready.remote()),
            cast(ObjectRef[dict], self.target.ready.remote()),
        ])
        if any(info["closed"] for info in infos):
            raise RuntimeError("Worker is closed")
        return infos

    def submit(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int,
    ) -> None:
        """Validate and enqueue a copied submission without allocating KV."""
        self._check_open()

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        
        if request_id in self.seen_ids:
            raise ValueError("Request ID has already been submitted")
        
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            raise ValueError("prompt_token_ids must be a nonempty list")

        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")

        if type(eos_token_id) is not int:
            raise ValueError("eos_token_id must be an integer")
        
        for token_id in prompt_token_ids:
            if type(token_id) is not int:
                raise ValueError("Token IDs must be integers")

            for info in self._metadata:
                if not 0 <= token_id < info["vocab_size"]:
                    raise ValueError("Token ID is outside the model vocabulary")

        maximum = len(prompt_token_ids) + max_new_tokens - 1
        for info in self._metadata:
            if maximum > info["num_blocks"] * info["block_size"]:
                raise ValueError("Request exceeds worker KV capacity")

        submission = Submission(
            request_id=request_id,
            prompt_token_ids=tuple(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        self.waiting.append(submission)
        self.seen_ids.add(request_id)

    def _decode_one(self) -> dict | None:
        if not self.decode_ready:
            return None
        request_id = self.decode_ready[0]
        try:
            result = ray.get(cast(ObjectRef[dict], self.target.step.remote(request_id)))
        except Exception:
            # Preserve the original failure if the actor is unreachable.
            try:
                info = ray.get(cast(ObjectRef[dict], self.target.ready.remote()))
            except Exception:
                info = None
            if info is not None and request_id not in info["active_request_ids"]:
                self.decode_ready.popleft()
            raise
        self.decode_ready.popleft()
        if result["status"] != "FINISHED":
            self.decode_ready.append(request_id)
            return None
        return result

    def _pull_one(self) -> None:
        result = pull_one_ray(self.source, self.target)
        if result is not None:
            self.decode_ready.append(result["request_id"])

    def _prefill_one(self) -> dict | None:
        if not self.waiting:
            return None
        submission = self.waiting[0]
        available = ray.get(cast(ObjectRef[bool], self.source.can_prefill.remote(
            len(submission.prompt_token_ids), submission.max_new_tokens)))
        if not available:
            return None
        result = ray.get(cast(ObjectRef[dict], self.source.prefill.remote(
            submission.request_id, list(submission.prompt_token_ids),
            submission.max_new_tokens, submission.eos_token_id)))
        self.waiting.popleft()
        return result if result["status"] == "FINISHED" else None

    def step(self) -> list[dict]:
        self._check_open()
        result = self._decode_one()
        if result is not None:
            self.completed.append(result)
        self._pull_one()
        result = self._prefill_one()
        if result is not None:
            self.completed.append(result)
        return self.drain_completed()

    def drain_completed(self) -> list[dict]:
        results = list(self.completed)
        self.completed.clear()
        return results

    def has_pending(self) -> bool:
        if self.waiting or self.decode_ready:
            return True
        return ray.get(cast(ObjectRef[dict | None], self.source.peek_ready.remote())) is not None
