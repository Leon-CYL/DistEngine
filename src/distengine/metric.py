from dataclasses import asdict, dataclass, field
import math
from typing import Callable
import time


@dataclass(frozen=True)
class LatencyLimits:
    ttft_s: float
    tpot_s: float
    e2e_s: float

    def __post_init__(self):
        for value in (self.ttft_s, self.tpot_s, self.e2e_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Latency limits must be finite and positive")


@dataclass
class RequestTiming:
    request_id: str
    scheduled_s: float
    submitted_s: float

    prefill_dispatched_s: float | None = None
    token_observed_s: list[float] = field(default_factory=list)

    finished_s: float | None = None
    error: str | None = None

    def measurements(self) -> dict:
        tokens = self.token_observed_s
        first = tokens[0] if tokens else None
        last = tokens[-1] if tokens else None

        return {
            "request_id": self.request_id,
            "output_tokens": len(tokens),
            "submission_delay_s": self.submitted_s - self.scheduled_s,
            "prefill_wait_s": (
                self.prefill_dispatched_s - self.submitted_s
                if self.prefill_dispatched_s is not None
                else None
            ),
            "ttft_s": (first - self.submitted_s if first is not None else None),
            "e2e_s": (self.finished_s - self.submitted_s if self.finished_s is not None else None),
            "tpot_s": ((last - first) / (len(tokens) - 1) if len(tokens) > 1 else None),
            "error": self.error,
        }


def percentile(values: list[float], fraction: float) -> float | None:
    """Linearly interpolate between sorted observations."""
    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)

    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class MetricsRecorder:
    """Driver-observed request events; create after worker setup and warm-up.

    Replay shares started_at/clock and freezes the report window on success.
    Actor execution and GPU utilization require separate instrumentation.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.perf_counter,
        *,
        started_at: float | None = None,
    ):
        self.clock = clock
        self.started_at = clock() if started_at is None else started_at
        self.requests: dict[str, RequestTiming] = {}
        self.finished_at: float | None = None
        self.orchestration_errors: list[dict] = []

    def elapsed(self) -> float:
        return (self.clock() if self.finished_at is None else self.finished_at) - self.started_at

    def _ensure_open(self):
        if self.finished_at is not None:
            raise ValueError("Measurement already ended")

    def finish(self) -> None:
        """Freeze the observation window after draining replay."""
        self._ensure_open()
        self.finished_at = self.clock()

    def orchestration_failed(self, error: Exception) -> None:
        """An orchestration error does not necessarily retire a request."""
        self._ensure_open()
        self.orchestration_errors.append(
            {
                "observed_s": self.elapsed(),
                "error": f"{type(error).__name__}: {error}",
            }
        )

    def submitted(self, request_id: str, scheduled_s: float) -> None:
        self._ensure_open()
        if request_id in self.requests:
            raise ValueError("Request already recorded")

        now = self.elapsed()
        if not math.isfinite(scheduled_s) or scheduled_s < 0:
            raise ValueError("Scheduled arrival must be finite and nonnegative")
        if scheduled_s > now:
            raise ValueError("Request submitted before its scheduled arrival")

        self.requests[request_id] = RequestTiming(
            request_id=request_id,
            scheduled_s=scheduled_s,
            submitted_s=now,
        )

    def prefill_dispatched(self, request_id: str) -> None:
        self._ensure_open()
        timing = self.requests[request_id]

        if timing.finished_s is not None:
            raise ValueError("Request already ended")
        if timing.prefill_dispatched_s is not None:
            raise ValueError("Prefill already dispatched")

        timing.prefill_dispatched_s = self.elapsed()

    def observe(
        self,
        request_id: str,
        *,
        generated_tokens: int,
        finished: bool,
    ) -> None:
        """Observe a successful prefill or decode result."""
        self._ensure_open()
        timing = self.requests[request_id]

        if timing.finished_s is not None:
            raise ValueError("Request already ended")
        if timing.prefill_dispatched_s is None:
            raise ValueError("Prefill dispatch has not been recorded")
        if type(generated_tokens) is not int:
            raise ValueError("Token count must be an integer")

        previous = len(timing.token_observed_s)

        # Current engine calls produce exactly one new token per result.
        if generated_tokens != previous + 1:
            raise ValueError("Expected exactly one new token")

        now = self.elapsed()
        timing.token_observed_s.append(now)

        if finished:
            timing.finished_s = now

    def failed(self, request_id: str, error: Exception) -> None:
        self._ensure_open()
        timing = self.requests[request_id]

        if timing.finished_s is not None:
            raise ValueError("Request already ended")

        timing.finished_s = self.elapsed()
        timing.error = f"{type(error).__name__}: {error}"

    def report(self, limits: LatencyLimits) -> dict:
        """Snapshot raw events and aggregate successful completed requests."""
        duration = self.elapsed()
        rows = [r.measurements() for r in self.requests.values()]

        successful = [
            row
            for row in rows
            if row["e2e_s"] is not None and row["error"] is None and row["output_tokens"] > 0
        ]

        good = [
            row
            for row in successful
            if row["ttft_s"] <= limits.ttft_s
            and row["e2e_s"] <= limits.e2e_s
            and (row["tpot_s"] is None or row["tpot_s"] <= limits.tpot_s)
        ]

        distributions = {}
        for name in (
            "submission_delay_s",
            "prefill_wait_s",
            "ttft_s",
            "tpot_s",
            "e2e_s",
        ):
            values = [row[name] for row in successful if row[name] is not None]
            distributions[name] = {
                "count": len(values),
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
            }

        return {
            "timing_scope": "driver-observed",
            "duration_s": duration,
            "submitted": len(rows),
            "completed": len(successful),
            "failed": sum(row["error"] is not None for row in rows),
            "unfinished": sum(row["e2e_s"] is None for row in rows),
            "request_throughput_rps": (len(successful) / duration if duration > 0 else None),
            "completed_output_tokens_per_s": (
                sum(row["output_tokens"] for row in successful) / duration if duration > 0 else None
            ),
            "goodput_rps": (len(good) / duration if duration > 0 else None),
            "goodput_limits": asdict(limits),
            "latencies": distributions,
            "requests": rows,
            "raw_timings": [asdict(timing) for timing in self.requests.values()],
            "orchestration_errors": [dict(item) for item in self.orchestration_errors],
        }


class _MetricsHooks:
    """Optional driver events shared by ordinary and Ray coordinators."""

    def _record_prefill_dispatch(self, request_id):
        if self.metrics is not None:
            timing = self.metrics.requests[request_id]
            if timing.prefill_dispatched_s is None:
                self.metrics.prefill_dispatched(request_id)

    def _record_result(self, result):
        if self.metrics is not None:
            self.metrics.observe(
                result["request_id"],
                generated_tokens=len(result["generated_token_ids"]),
                finished=result["status"] == "FINISHED",
            )

    def _record_retired(self, request_id, error):
        if self.metrics is not None:
            self.metrics.failed(request_id, error)

    def _record_submission(self, request_id):
        # Direct submit() calls have no external arrival schedule. Replay
        # records its scheduled arrival before calling submit().
        if self.metrics is not None and request_id not in self.metrics.requests:
            self.metrics.submitted(request_id, self.metrics.elapsed())
