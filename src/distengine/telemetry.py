"""Optional worker activity, tick-boundary state and background GPU sampling."""

from functools import wraps
import subprocess
import threading
import time


class WorkerActivity:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.started = clock()
        self.stopped = None
        self.busy_s = 0.0
        self.operations = {}
        self.depth = 0

    def report(self):
        elapsed = (self.clock() if self.stopped is None else self.stopped) - self.started
        return {"window_s": elapsed, "busy_s": self.busy_s,
                "idle_s": max(0.0, elapsed - self.busy_s),
                "busy_fraction": self.busy_s / elapsed if elapsed > 0 else 0.0,
                "operations": {k: dict(v) for k, v in self.operations.items()},
                "scope": "outermost worker operations; idle includes waiting and driver overhead"}


def measured(operation):
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            activity = self.activity
            if activity is None or activity.stopped is not None:
                return method(self, *args, **kwargs)
            outer = activity.depth == 0
            if outer:
                self._sync_measurement()
                start = activity.clock()
            activity.depth += 1
            try:
                return method(self, *args, **kwargs)
            finally:
                activity.depth -= 1
                if outer:
                    self._sync_measurement()
                    duration = activity.clock() - start
                    activity.busy_s += duration
                    row = activity.operations.setdefault(operation, {"calls": 0, "seconds": 0.0})
                    row["calls"] += 1
                    row["seconds"] += duration
        return wrapped
    return decorate


class SystemSampler:
    """Samples state between driver ticks; collection cost is part of replay."""
    def __init__(self, coordinator, fetch_workers, *, interval_s=0.25, clock=time.perf_counter):
        if interval_s <= 0:
            raise ValueError("Sampling interval must be positive")
        self.coordinator = coordinator
        self.fetch_workers = fetch_workers
        self.interval_s = interval_s
        self.clock = clock
        self.started_at = clock()
        self.next_sample = self.started_at
        self.samples = []

    def sample(self, force=False):
        now = self.clock()
        if not force and now < self.next_sample:
            return
        cached = getattr(self.coordinator, "cached_workers", None)
        workers = cached() if cached is not None and not force else self.fetch_workers()
        c = self.coordinator
        self.samples.append({
            "observed_s": now - self.started_at,
            "collection_s": self.clock() - now,
            "upstream_queue_lengths": [len(q) for q in c.waiting],
            "decode_queue_lengths": [len(q) for q in c.decode_ready],
            "pending_releases": len(c.pending_releases),
            "workers": workers,
        })
        self.next_sample = self.clock() + self.interval_s


class GPUSampler:
    """Poll nvidia-smi independently of blocking Ray calls. Never invent CPU data."""
    def __init__(self, *, interval_s=0.25, clock=time.perf_counter, query=None):
        if interval_s <= 0:
            raise ValueError("Sampling interval must be positive")
        self.interval_s = interval_s
        self.clock = clock
        self.query = query or self._query
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()

    @staticmethod
    def _query():
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, check=True, timeout=3)
        rows = []
        for line in result.stdout.strip().splitlines():
            index, uuid, utilization, used, total = [v.strip() for v in line.split(",")]
            rows.append({"index": int(index), "uuid": uuid,
                         "utilization_percent": float(utilization),
                         "memory_used_mib": float(used), "memory_total_mib": float(total)})
        if not rows:
            raise RuntimeError("nvidia-smi returned no GPUs")
        return rows

    def start(self, started_at):
        self.started_at = started_at
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                started = self.clock()
                gpus = self.query()
                self.samples.append({"observed_s": self.clock() - self.started_at,
                                     "collection_s": self.clock() - started, "gpus": gpus})
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self.stop_event.wait(self.interval_s)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("GPU sampler did not stop")

    def report(self, duration_s):
        samples = [s for s in self.samples if 0 <= s["observed_s"] <= duration_s]
        values = {}
        for sample in samples:
            for gpu in sample["gpus"]:
                values.setdefault(gpu.get("uuid", str(gpu["index"])), []).append(gpu["utilization_percent"])
        return {"samples": samples,
                "utilization_by_gpu": {key: {"sample_mean_percent": sum(v) / len(v),
                                             "sample_max_percent": max(v), "count": len(v)}
                                       for key, v in values.items()},
                "errors": list(self.errors), "interval_s": self.interval_s,
                "scope": "nvidia-smi utilization windows; GPU memory includes weights and allocator caches"}
