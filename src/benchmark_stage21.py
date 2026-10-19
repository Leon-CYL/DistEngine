"""Repeated stealing-disabled baseline with request, worker and GPU telemetry."""

import argparse
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import time

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import torch

from distengine.metric import LatencyLimits, MetricsRecorder
from distengine.multi_decode_coordinator import MultiDecodeCoordinator
from distengine.telemetry import GPUSampler, SystemSampler
from distengine.workload import Workload, _workload_from_dict, generate_workload, replay_workload
from generation import generate_greedy
from benchmark_io import source_identity, frozen_inputs, save_report



def smoke_workload():
    requests = generate_workload(arrival_mode="poisson", rate_rps=100,
                                num_requests=6, seed=42, prompt_token_ids=[10],
                                max_new_tokens=4, eos_token_id=-1)
    requests = tuple(replace(r, prompt_token_ids=tuple([10] * length))
                     for r, length in zip(requests, [1, 8, 16, 1, 8, 16]))
    return Workload(1, "poisson", 100, 42, requests)



def run_benchmark(workload_payload, *, cpu=False, repeats=3, interval_s=.25,
                  limits=None, provenance=None, frozen_files=None):
    if type(repeats) is not int or repeats <= 0 or interval_s <= 0:
        raise ValueError("Repeat count and sampling interval must be positive")
    limits = limits or LatencyLimits(2, .1, 10)
    workload = _workload_from_dict(workload_payload)
    if not workload.requests:
        raise ValueError("Benchmark requires requests")
    torch.set_num_threads(1)
    workers = []
    ray = None
    blocks, block_size = 64, 16
    keys = {(r.prompt_token_ids, r.max_new_tokens, r.eos_token_id) for r in workload.requests}
    for prompt, output, eos in keys:
        if len(prompt) + output - 1 > blocks * block_size:
            raise ValueError("Workload exceeds benchmark KV capacity")
    if cpu:
        from distengine.holding_pool import PrefillHoldingPool
        from distengine.workers import PrefillWorker, DecodeWorker
        from verify_workers import make_model
        for _ in range(2):
            workers.append(PrefillWorker(make_model(), blocks, block_size, PrefillHoldingPool(2)))
        for _ in range(2):
            workers.append(DecodeWorker(make_model(), blocks, block_size))
        model = workers[0].runner.model
        device = torch.device("cpu")
        model_id, revision = "deterministic-small-llama", None
    else:
        import ray as ray_module
        ray = ray_module
        if ray.is_initialized():
            raise RuntimeError("Benchmark requires a fresh Ray runtime")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
            raise RuntimeError("Benchmark requires exactly four visible NVIDIA GPUs")
        GPUSampler._query()  # Fail before loading weights if GPU polling is unavailable.
        from load_model import load_model, MODEL_ID, REVISION
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = load_model(device)
        model_id, revision = MODEL_ID, REVISION
    expected = {}
    with torch.inference_mode():
        for prompt, output, eos in sorted(keys):
            ids = generate_greedy(model, torch.tensor([prompt], device=device), output, eos)[0].tolist()
            expected[(prompt, output, eos)] = ids
    if not cpu:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    reports = []
    try:
        if not cpu:
            import sys
            from distengine.ray_workers import PrefillActor, DecodeActor
            from distengine.multi_decode_ray_coordinator import MultiDecodeRayCoordinator
            ray.init(address="local", num_cpus=4, num_gpus=4, include_dashboard=False,
                     runtime_env={"py_executable": sys.executable})
            for _ in range(2):
                workers.append(PrefillActor.options(num_gpus=1).remote(
                    num_blocks=blocks, block_size=block_size, device="cuda", checkpoint=True,
                    max_held_requests=2))
            for _ in range(2):
                workers.append(DecodeActor.options(num_gpus=1).remote(
                    num_blocks=blocks, block_size=block_size, device="cuda", checkpoint=True))
            placement = ray.get([w.ready.remote() for w in workers], timeout=600)
            if len({i["pid"] for i in placement}) != 4 or len({g for i in placement for g in i["ray_gpu_ids"]}) != 4:
                raise RuntimeError("Workers do not have independent processes and GPUs")
            if any(i["device"] != "cuda:0" or i["kv_device"] != "cuda:0" or
                   i["visible_gpu_count"] != 1 for i in placement):
                raise RuntimeError("Unexpected model/KV GPU placement")
        else:
            placement = [w.info() for w in workers]
        for run in range(repeats):
            factory = MultiDecodeCoordinator if cpu else MultiDecodeRayCoordinator
            warm = factory(workers[:2], workers[2:])
            longest = max(workload.requests, key=lambda r: len(r.prompt_token_ids))
            for i in range(4):
                warm.submit(f"warmup-{run}-{i}", list(longest.prompt_token_ids),
                            longest.max_new_tokens, longest.eos_token_id)
            for _ in range(4 * longest.max_new_tokens + 30):
                if not warm.has_pending():
                    break
                warm.step()
            if warm.has_pending():
                raise RuntimeError("Warm-up did not drain")
            c = factory(workers[:2], workers[2:])
            if cpu:
                for w in workers:
                    w.start_measurement()
                fetch = lambda stop=False: [w.telemetry(stop=stop) for w in workers]
            else:
                ray.get([w.start_measurement.remote() for w in workers])
                fetch = lambda stop=False: ray.get([w.telemetry.remote(stop=stop) for w in workers])
            sampler = SystemSampler(c, fetch, interval_s=interval_s)
            metrics = MetricsRecorder()
            c.metrics = metrics
            sampler.started_at = metrics.started_at
            gpu = None if cpu else GPUSampler(interval_s=interval_s)
            if gpu is not None:
                gpu.start(metrics.started_at)
            try:
                outputs = replay_workload(workload, c, metrics=metrics, sample=sampler.sample)
            finally:
                try:
                    final_workers = fetch(stop=True)
                finally:
                    if gpu is not None:
                        gpu.stop()
            sampler.sample(force=True)
            request_report = metrics.report(limits)
            if len(outputs) != len(workload.requests) or len({r["request_id"] for r in outputs}) != len(outputs):
                raise RuntimeError("Missing or duplicate request completions")
            by_id = {r["request_id"]: r for r in outputs}
            for request in workload.requests:
                result = by_id[request.request_id]
                if result["status"] != "FINISHED" or result["prompt_token_ids"] + result["generated_token_ids"] != expected[(request.prompt_token_ids, request.max_new_tokens, request.eos_token_id)]:
                    raise RuntimeError(f"Output mismatch: {request.request_id}")
            if any(w["active_request_ids"] or w["num_free_blocks"] != w["num_blocks"] or w["held_requests"] for w in final_workers):
                raise RuntimeError("Workers retained KV after replay")
            gpu_report = {"available": False, "reason": "CPU smoke test"} if gpu is None else gpu.report(request_report["duration_s"])
            if not cpu and (gpu_report["errors"] or not gpu_report["samples"]):
                raise RuntimeError(f"GPU sampling failed: {gpu_report['errors']}")
            reports.append({"run": run, "request_metrics": request_report,
                            "state_samples": sampler.samples, "worker_activity": final_workers,
                            "gpu": gpu_report, "decode_assignments": dict(c.decode_assignments),
                            "outputs": outputs, "output_correctness": "all match uncached greedy baseline"})
            print(f"PASS run {run + 1}/{repeats}: {len(outputs)} exact outputs, {request_report['duration_s']:.3f}s, full KV release", flush=True)
        aggregates = {}
        for name in ("request_throughput_rps", "completed_output_tokens_per_s", "goodput_rps"):
            values = [r["request_metrics"][name] for r in reports]
            aggregates[name] = {"values": values, "mean": statistics.mean(values),
                                "stdev": statistics.stdev(values) if len(values) > 1 else None}
        latency_variation = {}
        for metric in ("ttft_s", "tpot_s", "e2e_s", "prefill_wait_s", "submission_delay_s"):
            latency_variation[metric] = {}
            for quantile in ("p50", "p95", "p99"):
                values = [r["request_metrics"]["latencies"][metric][quantile] for r in reports]
                values = [v for v in values if v is not None]
                latency_variation[metric][quantile] = {
                    "values": values, "mean": statistics.mean(values) if values else None,
                    "stdev": statistics.stdev(values) if len(values) > 1 else None}
        return {"stage": 21, "status": "cpu-smoke-passed" if cpu else "baseline-passed",
                "stealing_enabled": False, "model": model_id, "revision": revision,
                "dtype": "float32", "python": platform.python_version(), "torch": torch.__version__,
                "ray": None if cpu else ray.__version__, "placement": placement,
                "config": {"prefill_workers": 2, "decode_workers": 2, "num_blocks": blocks,
                           "block_size": block_size, "max_held_requests": 2, "repeats": repeats,
                           "sample_interval_s": interval_s, "warmup_requests_per_run": 4,
                           "warmup_policy": "longest prompt, unchanged output limit/EOS",
                           "goodput_limits": asdict(limits)},
                "provenance": provenance, "frozen_files": frozen_files,
                "workload": asdict(workload), "runs": reports, "aggregate": aggregates,
                "latency_across_runs": latency_variation,
                "limitations": ["State samples are at tick boundaries and may miss transient peaks.",
                                "Polling/RPC and explicit CUDA synchronization overhead is included.",
                                "Worker idle is wall time outside measured operations, not GPU idle.",
                                "GPU utilization is nvidia-smi's sampled window, not per-kernel timing."]}
    finally:
        import sys
        already_failing = sys.exc_info()[0] is not None
        cleanup_errors = []
        for w in workers:
            try:
                if cpu:
                    w.close()
                else:
                    ray.get(w.close.remote(), timeout=10)
            except Exception as exc:
                cleanup_errors.append(exc)
            finally:
                if not cpu:
                    ray.kill(w, no_restart=True)
        if ray is not None and ray.is_initialized():
            ray.shutdown()
        if cleanup_errors and not already_failing:
            raise RuntimeError(f"Worker cleanup failed: {cleanup_errors}")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Small-model smoke check, not the GPU baseline")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/stage21-cpu-smoke"))
    args = parser.parse_args()
    if not args.cpu:
        parser.error("Use the Modal runner for the four-GPU baseline")
    if args.output.exists():
        parser.error("Choose a new output directory")
    root = Path(__file__).resolve().parents[1]
    report = run_benchmark(asdict(smoke_workload()), cpu=True, repeats=args.repeats,
                           interval_s=.01, provenance=source_identity(root))
    save_report(report, args.output)
    print(f"Saved {args.output / 'baseline.json'}")


if __name__ == "__main__":
    main()
