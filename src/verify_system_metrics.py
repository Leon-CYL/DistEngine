"""Deterministic worker/state/GPU telemetry checks and repeated CPU smoke run."""

from dataclasses import asdict
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import torch

from benchmark_stage21 import run_benchmark, smoke_workload, save_report
from distengine.telemetry import GPUSampler, SystemSampler, WorkerActivity, measured
from verify_workload import FakeClock


def check_activity():
    clock = FakeClock()
    class Worker:
        activity = WorkerActivity(clock)
        def _sync_measurement(self): pass
        @measured("outer")
        def outer(self):
            clock.now += 1
            self.inner()
        @measured("inner")
        def inner(self): clock.now += 2
        @measured("failure")
        def failure(self):
            clock.now += 1
            raise RuntimeError("expected")
    worker = Worker()
    worker.outer()
    try:
        worker.failure()
    except RuntimeError:
        pass
    clock.now += 2
    report = worker.activity.report()
    assert report["busy_s"] == 4 and report["idle_s"] == 2
    assert report["operations"]["outer"] == {"calls": 1, "seconds": 3}
    assert "inner" not in report["operations"] and worker.activity.depth == 0
    worker.activity.stopped = clock()
    clock.now += 10
    assert worker.activity.report() == report
    print("PASS: nested calls counted once, failure timing and frozen worker busy/idle windows")


def check_gpu():
    gpu = GPUSampler()
    with patch("subprocess.run") as run:
        run.return_value.stdout = "0, GPU-test, 42, 100, 1000\n"
        assert gpu._query() == [{"index": 0, "uuid": "GPU-test", "utilization_percent": 42.,
                                "memory_used_mib": 100., "memory_total_mib": 1000.}]
    def query():
        gpu.stop_event.set()
        return [{"index": 0, "utilization_percent": 42}]
    gpu.query = query
    gpu.start(gpu.clock())
    gpu.thread.join(timeout=2)
    gpu.stop()
    assert len(gpu.report(10)["samples"]) == 1 and not gpu.errors
    bad = GPUSampler()
    def failed_query():
        bad.stop_event.set()
        raise RuntimeError("unavailable")
    bad.query = failed_query
    bad.start(bad.clock())
    bad.thread.join(timeout=2)
    bad.stop()
    assert not bad.samples and bad.errors
    print("PASS: GPU parser, independent polling and explicit sampling failures")


def check_sampler():
    clock = FakeClock()
    class C:
        waiting = [[1, 2], []]
        decode_ready = [["a"], ["b"]]
        pending_releases = {}
    sampler = SystemSampler(C(), lambda: [{"kv_occupancy_fraction": .5}], clock=clock)
    sampler.sample(); sampler.sample()
    assert len(sampler.samples) == 1
    clock.now += .25
    sampler.sample()
    assert len(sampler.samples) == 2
    assert sampler.samples[0]["upstream_queue_lengths"] == [2, 0]
    assert sampler.samples[1]["decode_queue_lengths"] == [1, 1]
    print("PASS: interval-gated queue and KV snapshots")


def main():
    torch.set_num_threads(1)
    check_activity()
    check_gpu()
    check_sampler()
    report = run_benchmark(asdict(smoke_workload()), cpu=True, repeats=2, interval_s=.001)
    assert report["stealing_enabled"] is False and len(report["runs"]) == 2
    assert report["aggregate"]["request_throughput_rps"]["stdev"] is not None
    for run in report["runs"]:
        assert run["request_metrics"]["completed"] == 6
        assert run["gpu"]["available"] is False
        assert run["state_samples"]
        assert any(w["kv_blocks_used"] > 0 for s in run["state_samples"] for w in s["workers"])
        for worker in run["worker_activity"]:
            assert worker["activity"]["busy_s"] > 0
            assert worker["activity"]["idle_s"] >= 0 and worker["kv_blocks_used"] == 0
    with TemporaryDirectory() as directory:
        target = Path(directory) / "baseline"
        save_report(report, target)
        assert json.loads((target / "baseline.json").read_text())["status"] == "cpu-smoke-passed"
        assert len(list(target.glob("run-*.json"))) == 2
    print("PASS: repeated warm-up/replay, exact outputs, request/system metrics and saved baseline files")


if __name__ == "__main__":
    main()
