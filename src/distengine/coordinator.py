from collections import deque
from dataclasses import dataclass

from .admission import pull_one
from .workers import PrefillWorker, DecodeWorker


@dataclass(frozen=True)
class Submission:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int


class Coordinator:
    def __init__(
        self,
        source: PrefillWorker,
        target: DecodeWorker,
    ):
        for worker in (source, target):
            worker._ensure_open()
            if (worker.requests or worker.runner.caches
                    or worker.runner.block_manager.request_blocks):
                raise ValueError("Coordinator requires fresh workers")
        if len(source.prefill_pool):
            raise ValueError("Coordinator requires an empty holding pool")

        self.source = source
        self.target = target

        self.waiting: deque[Submission] = deque()
        self.decode_ready: deque[str] = deque()
        self.seen_ids: set[str] = set()
        self.completed: deque[dict] = deque()
        
    def submit(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int,
    ) -> None:
        """Validate and enqueue a copied submission without allocating KV."""
        self.source._ensure_open()
        self.target._ensure_open()

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

            for worker in (self.source, self.target):
                if not 0 <= token_id < worker.runner.model.config.vocab_size:
                    raise ValueError("Token ID is outside the model vocabulary")

        maximum = len(prompt_token_ids) + max_new_tokens - 1
        for worker in (self.source, self.target):
            manager = worker.runner.block_manager
            if maximum > manager.num_blocks * manager.block_size:
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
            result = self.target.step(request_id)
        except Exception:
            if request_id not in self.target.requests:
                self.decode_ready.popleft()
            raise

        self.decode_ready.popleft()
        if result["status"] != "FINISHED":
            self.decode_ready.append(request_id)
            return None

        return result
    
    def _pull_one(self) -> None:
        result = pull_one(self.source, self.target)

        if result is not None:
            self.decode_ready.append(result["request_id"])
            
    def _prefill_one(self) -> dict | None:
        if not self.waiting:
            return None

        submission = self.waiting[0]

        if not self.source.can_prefill(
            len(submission.prompt_token_ids),
            submission.max_new_tokens,
        ):
            return None

        result = self.source.prefill(
            submission.request_id,
            list(submission.prompt_token_ids),
            submission.max_new_tokens,
            submission.eos_token_id,
        )

        self.waiting.popleft()

        if result["status"] == "FINISHED":
            return result

        return None

    def step(self) -> list[dict]:
        """Advance decode, pull, then prefill; preserve completions on error."""
        self.source._ensure_open()
        self.target._ensure_open()

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
        return bool(
            self.waiting
            or len(self.source.prefill_pool)
            or self.decode_ready
        )
