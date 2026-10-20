from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from collections.abc import Callable
from typing import Literal, Protocol

from .metric import MetricsRecorder


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    arrival_offset_s: float
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int


def generate_workload(
    *,
    arrival_mode: Literal["constant", "poisson"],
    rate_rps: float,
    num_requests: int,
    seed: int,
    prompt_token_ids: list[int] | tuple[int, ...],
    max_new_tokens: int,
    eos_token_id: int,
) -> list[ScheduledRequest]:
    """Generate arrivals in seconds relative to replay start.

    Constant arrivals start at zero. Poisson arrivals start after the first
    exponential gap. Randomness is local and reproducible for a given seed.
    """
    if arrival_mode not in ("constant", "poisson"):
        raise ValueError("arrival_mode must be 'constant' or 'poisson'")

    if type(num_requests) is not int or num_requests < 0:
        raise ValueError("num_requests must be a nonnegative integer")

    if type(rate_rps) not in (int, float):
        raise ValueError("rate_rps must be a finite positive number")

    try:
        valid_rate = math.isfinite(rate_rps) and rate_rps > 0
    except OverflowError:
        valid_rate = False
        
    if not valid_rate:
        raise ValueError("rate_rps must be a finite positive number")

    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    if not isinstance(prompt_token_ids, (list, tuple)) or not prompt_token_ids:
        raise ValueError("prompt_token_ids must be a nonempty list or tuple")
    
    if any(type(token) is not int or token < 0 for token in prompt_token_ids):
        raise ValueError("Prompt token IDs must be nonnegative integers")

    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    
    if type(eos_token_id) is not int:
        raise ValueError("eos_token_id must be an integer")

    prompt = tuple(prompt_token_ids)
    rng = random.Random(seed)
    requests = []
    arrival_offset_s = 0.0

    for index in range(num_requests):
        if arrival_mode == "constant":
            arrival_offset_s = index / rate_rps
        else:
            arrival_offset_s += rng.expovariate(rate_rps)

        if not math.isfinite(arrival_offset_s):
            raise ValueError("rate_rps produces nonfinite arrival offsets")

        requests.append(ScheduledRequest(
            request_id=f"request-{index:06d}",
            arrival_offset_s=arrival_offset_s,
            prompt_token_ids=prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        ))

    return requests


@dataclass(frozen=True)
class Workload:
    schema_version: int
    arrival_mode: Literal["constant", "poisson"]
    rate_rps: float
    seed: int
    requests: tuple[ScheduledRequest, ...]


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _workload_from_dict(payload: object) -> Workload:
    """Validate serialized data without regenerating its arrival schedule."""
    fields = {"schema_version", "arrival_mode", "rate_rps", "seed", "requests"}
    
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("Workload must contain exactly the supported fields")
    
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported workload schema_version; expected 1")
    
    if payload["arrival_mode"] not in ("constant", "poisson"):
        raise ValueError("arrival_mode must be 'constant' or 'poisson'")
    
    rate = payload["rate_rps"]
    
    if not _finite_number(rate) or rate <= 0:
        raise ValueError("rate_rps must be a finite positive number")
    
    if type(payload["seed"]) is not int:
        raise ValueError("seed must be an integer")
    
    if not isinstance(payload["requests"], (list, tuple)):
        raise ValueError("requests must be an array")

    request_fields = {
        "request_id", "arrival_offset_s", "prompt_token_ids",
        "max_new_tokens", "eos_token_id",
    }
    
    requests = []
    seen_ids = set()
    previous_offset = 0.0
    
    for index, item in enumerate(payload["requests"]):
        prefix = f"Request {index}"
        if not isinstance(item, dict) or set(item) != request_fields:
            raise ValueError(f"{prefix} must contain exactly the supported fields")
        
        request_id = item["request_id"]
        
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"{prefix}: request_id must be a nonempty string")
        
        if request_id in seen_ids:
            raise ValueError(f"{prefix}: duplicate request_id {request_id!r}")
        
        offset = item["arrival_offset_s"]
        
        if not _finite_number(offset) or offset < previous_offset:
            raise ValueError(f"{prefix}: arrival offsets must be finite, nonnegative and ordered")
        
        prompt = item["prompt_token_ids"]
        if not isinstance(prompt, (list, tuple)) or not prompt:
            raise ValueError(f"{prefix}: prompt_token_ids must be a nonempty array")
        
        if any(type(token) is not int or token < 0 for token in prompt):
            raise ValueError(f"{prefix}: prompt tokens must be nonnegative integers")
        
        limit = item["max_new_tokens"]
        
        if type(limit) is not int or limit <= 0:
            raise ValueError(f"{prefix}: max_new_tokens must be a positive integer")
        
        if type(item["eos_token_id"]) is not int:
            raise ValueError(f"{prefix}: eos_token_id must be an integer")
        
        requests.append(ScheduledRequest(
            request_id, offset, tuple(prompt), limit, item["eos_token_id"],
        ))
        
        seen_ids.add(request_id)
        previous_offset = offset

    return Workload(1, payload["arrival_mode"], rate, payload["seed"], tuple(requests))


def save_workload(workload: Workload, path: str | Path) -> None:
    """Validate and save a workload, replacing an existing file at path."""
    if not isinstance(workload, Workload):
        raise ValueError("workload must be a Workload")
    
    payload = asdict(workload)
    _workload_from_dict(payload)
    text = json.dumps(payload, indent=2, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def load_workload(path: str | Path) -> Workload:
    """Load validated request inputs and offsets exactly as stored."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _workload_from_dict(payload)


class WorkloadCoordinator(Protocol):
    """Coordinator operations required by schedule replay."""

    def submit(
        self, request_id: str, prompt_token_ids: list[int],
        max_new_tokens: int, eos_token_id: int,
    ) -> None: ...

    def step(self) -> list[dict]: ...

    def has_pending(self) -> bool: ...


def replay_workload(
    workload: Workload,
    coordinator: WorkloadCoordinator,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    metrics: MetricsRecorder | None = None,
    sample: Callable[[], None] | None = None,
    deadline: float | None = None,
) -> list[dict]:
    """Submit scheduled arrivals and drain a fresh, exclusively owned coordinator.

    Deadlines use seconds relative to replay start. Blocking coordinator calls
    can delay submission; all overdue arrivals are submitted before the next
    step. Completion dictionaries are returned in coordinator delivery order.
    Errors propagate without retrying submissions or closing caller-owned workers.
    Injected clocks must be monotonic, and sleep must advance or wait on that clock.
    An optional recorder must be shared with the coordinator. Its clock and
    origin define replay time, and return freezes its report window. An optional
    absolute deadline stops between blocking calls and may leave pending work;
    the caller must close workers. It cannot interrupt an in-flight model/Ray call.
    """
    if not isinstance(workload, Workload):
        raise ValueError("workload must be a Workload")
    # Validate before submitting anything, including directly constructed data.
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite clock timestamp")
    workload = _workload_from_dict(asdict(workload))
    if coordinator.has_pending():
        raise ValueError("Replay requires a coordinator with no pending work")

    coordinator_metrics = getattr(coordinator, "metrics", None)
    if metrics is None:
        metrics = coordinator_metrics
    if metrics is not None:
        if metrics is not coordinator_metrics:
            raise ValueError("Replay and coordinator must share the same recorder")
        if metrics.requests or metrics.finished_at is not None:
            raise ValueError("Replay requires a fresh metrics recorder")
        if clock is not time.perf_counter and clock != metrics.clock:
            raise ValueError("Replay and metrics must share the same clock")
        clock = metrics.clock

    next_request = 0
    completed = []
    start = clock() if metrics is None else metrics.started_at
    while next_request < len(workload.requests) or coordinator.has_pending():
        if deadline is not None and clock() >= deadline:
            break
        while next_request < len(workload.requests):
            request = workload.requests[next_request]
            if deadline is not None and clock() >= deadline:
                break
            if request.arrival_offset_s > clock() - start:
                break
            if metrics is not None:
                metrics.submitted(request.request_id, request.arrival_offset_s)
            try:
                coordinator.submit(
                    request.request_id, list(request.prompt_token_ids),
                    request.max_new_tokens, request.eos_token_id,
                )
            except Exception as exc:
                if metrics is not None:
                    metrics.failed(request.request_id, exc)
                raise
            next_request += 1

        if sample is not None:
            sample()

        if deadline is not None and clock() >= deadline:
            break
        if coordinator.has_pending():
            try:
                completed.extend(coordinator.step())
            except Exception as exc:
                if metrics is not None:
                    metrics.orchestration_failed(exc)
                raise
        elif next_request < len(workload.requests):
            remaining = workload.requests[next_request].arrival_offset_s - (clock() - start)
            if deadline is not None:
                remaining = min(remaining, max(0.0, deadline - clock()))
            if remaining > 0:
                sleep(remaining)

    if metrics is not None:
        metrics.finish()
    return completed
