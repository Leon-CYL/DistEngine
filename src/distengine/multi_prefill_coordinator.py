from collections import deque

from .admission import pull_one
from .coordinator import Submission
from .workers import PrefillWorker, DecodeWorker


class MultiPrefillCoordinator:
    def __init__(self, sources: list[PrefillWorker], target: DecodeWorker):
        if not sources:
            raise ValueError("At least one prefill worker is required")
        
        if len({id(source) for source in sources}) != len(sources):
            raise ValueError("Prefill workers must be distinct")
        
        if any(source is target for source in sources):
            raise ValueError("Decode worker must be distinct from prefill workers")
        
        if len({id(source.prefill_pool) for source in sources}) != len(sources):
            raise ValueError("Prefill workers must have independent holding pools")
        
        self.sources = list(sources)
        self.target = target
        
        self._ensure_open()
        
        for worker in (*self.sources, target):
            if (worker.requests or worker.runner.caches
                    or worker.runner.block_manager.request_blocks):
                raise ValueError("Coordinator requires fresh workers")
            
        if any(len(source.prefill_pool) for source in self.sources):
            raise ValueError("Coordinator requires empty holding pools")
        
        self.waiting: list[deque[Submission]] = [deque() for _ in self.sources]
        self.decode_ready: deque[str] = deque()
        self.completed: deque[dict] = deque()
        self.seen_ids: set[str] = set()
        self.next_submit = 0
        self.next_pull = 0

    def _ensure_open(self) -> None:
        for worker in (*self.sources, self.target):
            worker._ensure_open()

    def submit(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int,
    ) -> None:
        """Validate and enqueue a copied submission without allocating KV."""
        self._ensure_open()
        source = self.sources[self.next_submit]

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

            for worker in (source, self.target):
                if not 0 <= token_id < worker.runner.model.config.vocab_size:
                    raise ValueError("Token ID is outside the model vocabulary")

        maximum = len(prompt_token_ids) + max_new_tokens - 1
        for worker in (source, self.target):
            manager = worker.runner.block_manager
            if maximum > manager.num_blocks * manager.block_size:
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
        count = len(self.sources)
        start = self.next_pull
        for offset in range(count):
            index = (start + offset) % count
            result = pull_one(self.sources[index], self.target)
            if result is not None:
                self.decode_ready.append(result["request_id"])
                self.next_pull = (index + 1) % count
                return
        self.next_pull = (start + 1) % count

    def _prefill_one(self, source_index: int) -> dict | None:
        queue = self.waiting[source_index]
        source = self.sources[source_index]
        if not queue:
            return None

        submission = queue[0]

        if not source.can_prefill(
            len(submission.prompt_token_ids),
            submission.max_new_tokens,
        ):
            return None

        result = source.prefill(
            submission.request_id,
            list(submission.prompt_token_ids),
            submission.max_new_tokens,
            submission.eos_token_id,
        )

        queue.popleft()

        if result["status"] == "FINISHED":
            return result

        return None

    def step(self) -> list[dict]:
        """Decode once, pull at most once, then try prefill on each source.

        Resolve every source's prefill attempt before propagating the first
        prefill error, so a failing source does not prevent another's progress.
        Completions survive any later operation failure.
        """
        self._ensure_open()
        result = self._decode_one()
        if result is not None:
            self.completed.append(result)
        self._pull_one()
        first_error = None
        for index in range(len(self.sources)):
            try:
                result = self._prefill_one(index)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            else:
                if result is not None:
                    self.completed.append(result)
        if first_error is not None:
            raise first_error
        return self.drain_completed()

    def drain_completed(self) -> list[dict]:
        results = list(self.completed)
        self.completed.clear()
        return results

    def has_pending(self) -> bool:
        return bool(self.decode_ready or any(self.waiting)
                    or any(len(source.prefill_pool) for source in self.sources))
