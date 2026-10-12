from collections import deque

from .request import Request, RequestStatus
from .model_runner import ModelRunner


class Scheduler:
    """Sequential FIFO scheduling with exclusive ownership of a fresh runner.

    Failed requests leave scheduling with their current status and reserved ID;
    they are not recorded as successfully finished.
    """

    def __init__(self, runner: ModelRunner) -> None:
        if runner.caches:
            raise ValueError("Scheduler requires a runner with no active caches")
        if runner.block_manager.request_blocks:
            raise ValueError("Scheduler requires a block manager with no registered requests")
        self.runner = runner
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()
        self.finished: list[Request] = []
        self.request_ids: set[str] = set()

    def _validate_waiting(self, request: Request) -> None:
        self.runner._validate_request(request)
        if request.status != RequestStatus.WAITING:
            raise ValueError("Request must be WAITING")
        if request.generated_token_ids:
            raise ValueError("Waiting request must have no generated tokens")
        manager = self.runner.block_manager
        max_cached_tokens = request.num_prompt_tokens + request.max_new_tokens - 1
        required_blocks = (max_cached_tokens + manager.block_size - 1) // manager.block_size
        if required_blocks > manager.num_blocks:
            raise ValueError("Request's maximum cached history exceeds the block pool")

    def add_request(self, request: Request) -> None:
        self._validate_waiting(request)
        if request.request_id in self.request_ids:
            raise ValueError("Request ID has already been submitted")
        self.waiting.append(request)
        self.request_ids.add(request.request_id)

    @property
    def has_pending_requests(self) -> bool:
        return bool(self.waiting or self.running)

    def step(self) -> Request | None:
        if self.running:
            request = self.running[0]
            operation = self.runner.decode
        elif self.waiting:
            request = self.waiting[0]
            self._validate_waiting(request)
            self.waiting.popleft()
            request.start()
            self.running.append(request)
            operation = self.runner.prefill
        else:
            return None

        try:
            operation(request)
        except Exception:
            # Runner failures may already have released their own cache.
            self.running.popleft()
            if request.request_id in self.runner.caches:
                self.runner.release(request)
            raise

        if request.is_finished:
            self._finish_request(request)
        return request

    def _finish_request(self, request: Request) -> None:
        if request.status != RequestStatus.FINISHED:
            raise ValueError("Request must be FINISHED")
        if not self.running or self.running[0] is not request:
            raise ValueError("Request is not the active request")
        self.runner.release(request)
        self.running.popleft()
        self.finished.append(request)
