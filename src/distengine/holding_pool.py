class PrefillHoldingPool:
    def __init__(self, max_held_requests: int):
        if type(max_held_requests) is not int or max_held_requests <= 0:
            raise ValueError("max_held_requests must be a positive integer")

        self._max_held_requests = max_held_requests
        self._ready: dict[str, None] = {}

    def is_full(self) -> bool:
        return len(self._ready) >= self._max_held_requests

    def add(self, request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")

        if request_id in self._ready:
            raise ValueError("Request is already held")

        if self.is_full():
            raise RuntimeError("Prefill holding pool is full")

        self._ready[request_id] = None

    def peek(self) -> str | None:
        return next(iter(self._ready), None)

    def remove(self, request_id: str) -> None:
        del self._ready[request_id]

    def __len__(self) -> int:
        return len(self._ready)

    def request_ids(self) -> list[str]:
        return list(self._ready)
