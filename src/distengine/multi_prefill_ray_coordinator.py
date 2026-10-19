"""Serial driver with concurrent prefills across exclusively owned Ray actors."""

from collections import deque
from typing import cast

import ray
from ray import ObjectRef

from .admission import pull_one_ray
from .coordinator import Submission


class MultiPrefillRayCoordinator:
    def __init__(self, sources, target):
        if not sources:
            raise ValueError("At least one prefill actor is required")
        self.sources = list(sources)
        self.target = target
        self._metadata = self._check_open()
        # Actor IDs also catch different handles referring to the same actor.
        identities = [info["actor_id"] for info in self._metadata]
        if len(set(identities)) != len(identities):
            raise ValueError("All worker actors must be distinct")
        for info in self._metadata:
            if info["active_request_ids"] or info["num_free_blocks"] != info["num_blocks"]:
                raise ValueError("Coordinator requires fresh workers")
        held = ray.get([cast(ObjectRef[dict | None], s.peek_ready.remote()) for s in self.sources])
        if any(item is not None for item in held):
            raise ValueError("Coordinator requires empty holding pools")
        self.waiting: list[deque[Submission]] = [deque() for _ in self.sources]
        self.decode_ready: deque[str] = deque()
        self.completed: deque[dict] = deque()
        self.seen_ids: set[str] = set()
        self.next_submit = 0
        self.next_pull = 0

    def _check_open(self) -> list[dict]:
        infos = ray.get([cast(ObjectRef[dict], a.ready.remote())
                         for a in [*self.sources, self.target]])
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
        metadata = (self._metadata[self.next_submit], *self._submission_targets())

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

            for info in metadata:
                if not 0 <= token_id < info["vocab_size"]:
                    raise ValueError("Token ID is outside the model vocabulary")

        maximum = len(prompt_token_ids) + max_new_tokens - 1
        for info in metadata:
            if maximum > info["num_blocks"] * info["block_size"]:
                raise ValueError("Request exceeds worker KV capacity")

        submission = Submission(
            request_id=request_id,
            prompt_token_ids=tuple(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        self.waiting[self.next_submit].append(submission)
        self.seen_ids.add(request_id)
        self.next_submit = (self.next_submit + 1) % len(self.sources)

    def _submission_targets(self):
        return (self._metadata[-1],)

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
        count = len(self.sources)
        start = self.next_pull
        for offset in range(count):
            index = (start + offset) % count
            result = pull_one_ray(self.sources[index], self.target)
            if result is not None:
                self.decode_ready.append(result["request_id"])
                self.next_pull = (index + 1) % count
                return
        self.next_pull = (start + 1) % count

    def _prefill_sources(self) -> None:
        """Dispatch eligible actors first, then reconcile every submitted task."""
        pending = []
        first_error = None
        for index, source in enumerate(self.sources):
            if not self.waiting[index]:
                continue
            submission = self.waiting[index][0]
            try:
                available = ray.get(cast(ObjectRef[bool], source.can_prefill.remote(
                    len(submission.prompt_token_ids), submission.max_new_tokens)))
                if available:
                    ref = source.prefill.remote(
                        submission.request_id, list(submission.prompt_token_ids),
                        submission.max_new_tokens, submission.eos_token_id)
                    pending.append((index, cast(ObjectRef[dict], ref)))
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        # No prefill ray.get occurs until every eligible source was dispatched.
        # A failure cannot skip reconciliation of another actor's success.
        for index, ref in pending:
            try:
                result = ray.get(ref)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            else:
                self.waiting[index].popleft()
                if result["status"] == "FINISHED":
                    self.completed.append(result)
        if first_error is not None:
            raise first_error

    def step(self) -> list[dict]:
        self._check_open()
        result = self._decode_one()
        if result is not None:
            self.completed.append(result)
        self._pull_one()
        self._prefill_sources()
        return self.drain_completed()

    def drain_completed(self) -> list[dict]:
        results = list(self.completed)
        self.completed.clear()
        return results

    def has_pending(self) -> bool:
        if self.decode_ready or any(self.waiting):
            return True
        held = ray.get([cast(ObjectRef[dict | None], s.peek_ready.remote()) for s in self.sources])
        return any(item is not None for item in held)
