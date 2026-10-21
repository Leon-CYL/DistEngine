"""Repeated baseline/stealing benchmark with request, worker and GPU telemetry."""

import argparse
from dataclasses import asdict, replace
import gc
import math
import json
import os
from pathlib import Path
import platform
import statistics
import time

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import torch

from distengine.metric import LatencyLimits, MetricsRecorder
from distengine.coordinator import Coordinator
from distengine.telemetry import GPUSampler, SystemSampler
from distengine.workload import Workload, _workload_from_dict, generate_workload, replay_workload
from utils.generation import generate_greedy_cached
from utils.benchmark_io import source_identity, frozen_inputs, save_report


def smoke_workload():
    requests = generate_workload(
        arrival_mode="poisson",
        rate_rps=100,
        num_requests=6,
        seed=42,
        prompt_token_ids=[10],
        max_new_tokens=4,
        eos_token_id=-1,
    )
    requests = tuple(
        replace(r, prompt_token_ids=tuple([10] * length))
        for r, length in zip(requests, [1, 8, 16, 1, 8, 16])
    )
    return Workload(1, "poisson", 100, 42, requests)


def comparison_order(repeats):
    """Counterbalance adjacent pairs: off/on, on/off, off/on, ..."""
    return [
        mode
        for pair in range(repeats)
        for mode in ((False, True) if pair % 2 == 0 else (True, False))
    ]


def aggregate_runs(reports):
    aggregates = {}
    for name in (
        "request_throughput_rps",
        "completed_output_tokens_per_s",
        "observed_output_tokens_per_s",
        "goodput_rps",
    ):
        values = [r["request_metrics"][name] for r in reports]
        aggregates[name] = {
            "values": values,
            "mean": statistics.mean(values) if values else None,
            "stdev": statistics.stdev(values) if len(values) > 1 else None,
        }
    latency = {}
    for metric in ("ttft_s", "tpot_s", "e2e_s", "prefill_wait_s", "submission_delay_s"):
        latency[metric] = {}
        for quantile in ("p50", "p95", "p99"):
            values = [r["request_metrics"]["latencies"][metric][quantile] for r in reports]
            values = [v for v in values if v is not None]
            latency[metric][quantile] = {
                "values": values,
                "mean": statistics.mean(values) if values else None,
                "stdev": statistics.stdev(values) if len(values) > 1 else None,
            }
    return {
        "aggregate": aggregates,
        "latency_across_runs": latency,
        "executed_repeats": len(reports),
    }


def run_benchmark(
    workload_payload,
    *,
    cpu=False,
    repeats=3,
    interval_s=0.25,
    limits=None,
    provenance=None,
    frozen_files=None,
    num_blocks=64,
    block_size=16,
    max_held_requests=2,
    warmup_requests=4,
    gpu_family=None,
    min_gpu_memory_gib=0,
    max_run_s=None,
    decode_batch_size=1,
    prefill_workers=2,
    decode_workers=2,
    stealing_enabled=False,
    compare_stealing=False,
    batch_sizes=None,
):
    if type(repeats) is not int or repeats <= 0 or interval_s <= 0:
        raise ValueError("Repeat count and sampling interval must be positive")
    for name, value in (
        ("prefill_workers", prefill_workers),
        ("decode_workers", decode_workers),
        ("decode_batch_size", decode_batch_size),
        ("num_blocks", num_blocks),
        ("block_size", block_size),
        ("max_held_requests", max_held_requests),
        ("warmup_requests", warmup_requests),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_run_s is not None and (
        not isinstance(max_run_s, (int, float)) or not math.isfinite(max_run_s) or max_run_s <= 0
    ):
        raise ValueError("max_run_s must be finite and positive")
    if type(stealing_enabled) is not bool:
        raise ValueError("stealing_enabled must be a boolean")
    if type(compare_stealing) is not bool:
        raise ValueError("compare_stealing must be a boolean")
    if compare_stealing and stealing_enabled:
        raise ValueError("Choose compare_stealing or stealing_enabled, not both")
    kv_transport = "cpu" if cpu else "nccl"
    if batch_sizes is not None:
        if (
            compare_stealing
            or not batch_sizes
            or any(type(b) is not int or b <= 0 for b in batch_sizes)
        ):
            raise ValueError(
                "Batch sweep requires positive batch sizes and a single stealing policy"
            )
    total_workers = prefill_workers + decode_workers
    limits = limits or LatencyLimits(2, 0.1, 10)
    workload = _workload_from_dict(workload_payload)
    if not workload.requests:
        raise ValueError("Benchmark requires requests")
    torch.set_num_threads(1)
    workers = []
    ray = None
    blocks = num_blocks
    keys = {(r.prompt_token_ids, r.max_new_tokens, r.eos_token_id) for r in workload.requests}
    for prompt, output, eos in keys:
        if len(prompt) + output - 1 > blocks * block_size:
            raise ValueError("Workload exceeds benchmark KV capacity")
    if cpu:
        from distengine.holding_pool import PrefillHoldingPool
        from distengine.worker import PrefillWorker, DecodeWorker, build_cpu_model

        for _ in range(prefill_workers):
            workers.append(
                PrefillWorker(
                    build_cpu_model(0),
                    blocks,
                    block_size,
                    PrefillHoldingPool(max_held_requests),
                )
            )
        for _ in range(decode_workers):
            workers.append(DecodeWorker(build_cpu_model(0), blocks, block_size))
        model = workers[0].runner.model
        device = torch.device("cpu")
        model_id, revision = "deterministic-small-llama", None
    else:
        import ray as ray_module

        ray = ray_module
        if ray.is_initialized():
            raise RuntimeError("Benchmark requires a fresh Ray runtime")
        if not torch.cuda.is_available() or torch.cuda.device_count() != total_workers:
            raise RuntimeError(f"Benchmark requires exactly {total_workers} visible NVIDIA GPUs")
        GPUSampler._query()  # Fail before loading weights if GPU polling is unavailable.
        from utils.load_model import load_model, MODEL_ID, REVISION

        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = load_model(device)
        model_id, revision = MODEL_ID, REVISION
    expected = {}
    from distengine.paged_kv_cache import PagedKVCache

    config = model.config
    with torch.inference_mode():
        for prompt, output, eos in sorted(keys):
            required = (len(prompt) + output + block_size - 1) // block_size
            cache = PagedKVCache(
                num_layers=config.num_hidden_layers,
                num_kv_heads=config.num_key_value_heads or config.num_attention_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                num_physical_blocks=required,
                block_size=block_size,
                block_table=list(range(required)),
                device=device,
                dtype=torch.float32,
            )
            ids = generate_greedy_cached(
                model, torch.tensor([prompt], device=device), output, eos, cache
            )[0].tolist()
            expected[(prompt, output, eos)] = ids
    del cache
    if not cpu:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    reports = []
    cleanup_state = []
    collective_group = None
    transfer_checks = []
    gpu_topology = None
    gpu_topology_error = None
    try:
        if not cpu:
            import sys
            from distengine.worker import PrefillActor, DecodeActor

            ray.init(
                address="local",
                num_cpus=total_workers,
                num_gpus=total_workers,
                include_dashboard=False,
                runtime_env={"py_executable": sys.executable},
            )
            for _ in range(prefill_workers):
                workers.append(
                    PrefillActor.options(num_gpus=1).remote(
                        num_blocks=blocks,
                        block_size=block_size,
                        device="cuda",
                        checkpoint=True,
                        max_held_requests=max_held_requests,
                    )
                )
            for _ in range(decode_workers):
                workers.append(
                    DecodeActor.options(num_gpus=1).remote(
                        num_blocks=blocks,
                        block_size=block_size,
                        device="cuda",
                        checkpoint=True,
                    )
                )
            placement = ray.get([w.ready.remote() for w in workers], timeout=600)
            if (
                len({i["pid"] for i in placement}) != total_workers
                or len({g for i in placement for g in i["ray_gpu_ids"]}) != total_workers
            ):
                raise RuntimeError("Workers do not have independent processes and GPUs")
            if any(
                i["device"] != "cuda:0" or i["kv_device"] != "cuda:0" or i["visible_gpu_count"] != 1
                for i in placement
            ):
                raise RuntimeError("Unexpected model/KV GPU placement")
            if gpu_family is not None and any(gpu_family not in i["gpu_name"] for i in placement):
                raise RuntimeError(f"Expected {gpu_family} GPUs")
            if any(i.get("gpu_memory_bytes", 0) < min_gpu_memory_gib * 1024**3 for i in placement):
                raise RuntimeError("GPU memory does not match the requested baseline profile")
            if kv_transport == "nccl":
                import subprocess
                from ray.experimental.collective import create_collective_group

                if len({info["node_id"] for info in placement}) != 1:
                    raise RuntimeError("This benchmark expects all four GPU actors on one node")
                try:
                    topology = subprocess.run(
                        ["nvidia-smi", "topo", "-m"],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    )
                    gpu_topology = topology.stdout
                    if topology.returncode:
                        gpu_topology_error = (
                            f"nvidia-smi topo exited {topology.returncode}: {topology.stderr}"
                        )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    gpu_topology_error = str(exc)
                collective_group = create_collective_group(workers, backend="nccl")
                # Exercise each source/target path before timing; do not ray.get
                # GPU exports on the driver or stage them through the object store.
                for source in range(prefill_workers):
                    for target in range(prefill_workers, total_workers):
                        probe = (
                            workers[source].transfer_probe.options(tensor_transport="nccl").remote()
                        )
                        check = ray.get(
                            workers[target].validate_transfer_probe.remote(probe), timeout=60
                        )
                        if check["received_device"] != "cuda:0":
                            raise RuntimeError("NCCL probe did not arrive on CUDA")
                        transfer_checks.append(dict(source=source, target=target, **check))
                        del probe
                print("PASS: NCCL CUDA tensor checks for every prefill/decode pair", flush=True)
        else:
            placement = [w.info() for w in workers]
        modes = comparison_order(repeats) if compare_stealing else [stealing_enabled] * repeats
        plan = [(mode, batch) for batch in (batch_sizes or [decode_batch_size]) for mode in modes]
        for run, (mode, batch) in enumerate(plan):
            factory = Coordinator
            scheduling = dict(decode_batch_size=batch)
            scheduling["stealing_enabled"] = mode
            warm = factory(workers[:prefill_workers], workers[prefill_workers:], **scheduling)
            longest = max(workload.requests, key=lambda r: len(r.prompt_token_ids))
            for i in range(warmup_requests):
                warm.submit(
                    f"warmup-{run}-{i}",
                    list(longest.prompt_token_ids),
                    longest.max_new_tokens,
                    longest.eos_token_id,
                )
            warmup_deadline = time.perf_counter() + 60
            while warm.has_pending() and time.perf_counter() < warmup_deadline:
                warm.step()
            if warm.has_pending():
                raise RuntimeError("Warm-up did not drain")
            c = factory(workers[:prefill_workers], workers[prefill_workers:], **scheduling)
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
                outputs = replay_workload(
                    workload,
                    c,
                    metrics=metrics,
                    sample=sampler.sample,
                    deadline=None if max_run_s is None else metrics.started_at + max_run_s,
                )
            finally:
                try:
                    final_workers = fetch(stop=True)
                finally:
                    if gpu is not None:
                        gpu.stop()
            sampler.sample(force=True)
            request_report = metrics.report(limits)
            observed_tokens = sum(len(t.token_observed_s) for t in metrics.requests.values())
            request_report["observed_output_tokens"] = observed_tokens
            request_report["observed_output_tokens_per_s"] = (
                observed_tokens / request_report["duration_s"]
            )
            complete = len(outputs) == len(workload.requests) and not c.has_pending()
            if len({r["request_id"] for r in outputs}) != len(outputs):
                raise RuntimeError("Duplicate request completions")
            if not complete and max_run_s is None:
                raise RuntimeError("Missing request completions")
            validation = {"checked_sequences": 0, "mismatches": []}
            samples = list(outputs)
            requests = {r.request_id: r for r in workload.requests}
            for result in samples:
                request = requests[result["request_id"]]
                reference = expected[
                    (request.prompt_token_ids, request.max_new_tokens, request.eos_token_id)
                ]
                actual = result["prompt_token_ids"] + result["generated_token_ids"]
                finished = result["status"] == "FINISHED"
                mismatch = (
                    result["prompt_token_ids"] != list(request.prompt_token_ids)
                    or actual != reference[: len(actual)]
                    or (finished and len(actual) != len(reference))
                )
                validation["checked_sequences"] += 1
                if mismatch:
                    first = next(
                        (i for i, (a, b) in enumerate(zip(actual, reference)) if a != b),
                        min(len(actual), len(reference)),
                    )
                    validation["mismatches"].append(
                        {
                            "request_id": request.request_id,
                            "first_different_index": first,
                            "actual_token": actual[first] if first < len(actual) else None,
                            "expected_token": reference[first] if first < len(reference) else None,
                        }
                    )
            validation["status"] = (
                "failed"
                if validation["mismatches"]
                else "passed"
                if validation["checked_sequences"]
                else "not-checked"
            )
            if complete and any(
                w["active_request_ids"]
                or w["num_free_blocks"] != w["num_blocks"]
                or w["held_requests"]
                for w in final_workers
            ):
                raise RuntimeError("Workers retained KV after replay")
            if complete:
                sent = sum(w["kv_export_bytes"] for w in final_workers)
                received = sum(w["kv_import_bytes"] for w in final_workers)
                if sent != received:
                    raise RuntimeError("KV export/import byte counts disagree")
            gpu_report = (
                {"available": False, "reason": "CPU smoke test"}
                if gpu is None
                else gpu.report(request_report["duration_s"])
            )
            if not cpu and (gpu_report["errors"] or not gpu_report["samples"]):
                raise RuntimeError(f"GPU sampling failed: {gpu_report['errors']}")
            reports.append(
                {
                    "run": run,
                    "decode_batch_size": batch,
                    "stealing_enabled": mode,
                    "complete": complete,
                    "stop_reason": "drained" if complete else "measured-time limit",
                    "not_submitted": len(workload.requests) - request_report["submitted"],
                    "request_metrics": request_report,
                    "state_samples": sampler.samples,
                    "worker_activity": final_workers,
                    "gpu": gpu_report,
                    "decode_assignments": dict(c.decode_assignments),
                    "kv_handoffs": list(c.handoff_events),
                    "prefill_assignments": dict(getattr(c, "prefill_assignments", {})),
                    "steal_events": list(getattr(c, "steal_events", [])),
                    "steal_count": len(getattr(c, "steal_events", [])),
                    "outputs": outputs,
                    "validation": validation,
                    "output_correctness": validation["status"],
                }
            )
            label = "MISMATCH" if validation["mismatches"] else "PASS" if complete else "PARTIAL"
            print(
                f"{label} run {run + 1}/{len(plan)}: batch={batch}, {len(outputs)}/{len(workload.requests)} completed, "
                f"{request_report['duration_s']:.3f}s; {observed_tokens} tokens "
                f"({request_report['observed_output_tokens_per_s']:.2f} tokens/s)",
                flush=True,
            )
            if validation["mismatches"]:
                break  # Keep diagnostics and avoid paying for more invalid measurements.
            if not complete:
                break
        summary = aggregate_runs(reports)
        comparison = (
            {
                name: aggregate_runs([r for r in reports if r["stealing_enabled"] == enabled])
                for name, enabled in (("no-stealing", False), ("stealing", True))
            }
            if compare_stealing
            else None
        )
        correctness_failed = any(r["validation"]["status"] == "failed" for r in reports)
        return {
            "capacity_by_batch": {
                str(batch): aggregate_runs([r for r in reports if r["decode_batch_size"] == batch])
                for batch in batch_sizes
            }
            if batch_sizes
            else None,
            "stage": 22 if compare_stealing else 21,
            "correctness_failed": correctness_failed,
            "status": "correctness-failed"
            if correctness_failed
            else "time-limited"
            if not all(r["complete"] for r in reports)
            else (
                "cpu-smoke-passed"
                if cpu
                else ("calibration-passed" if max_run_s is not None else "baseline-passed")
            ),
            "purpose": "paired-stealing-comparison"
            if compare_stealing
            else "calibration"
            if max_run_s is not None
            else "baseline",
            "complete": all(r["complete"] for r in reports),
            "stealing_enabled": None if compare_stealing else stealing_enabled,
            "comparison": comparison,
            "comparison_order": modes if compare_stealing else None,
            "model": model_id,
            "revision": revision,
            "dtype": "float32",
            "reference": "float32 cached greedy",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ray": None if cpu else ray.__version__,
            "placement": placement,
            "transfer_checks": transfer_checks,
            "gpu_topology": gpu_topology,
            "gpu_topology_error": gpu_topology_error,
            "config": {
                "kv_transport": kv_transport,
                "prefill_workers": prefill_workers,
                "decode_workers": decode_workers,
                "decode_batch_size": decode_batch_size,
                "num_blocks": blocks,
                "block_size": block_size,
                "max_held_requests": max_held_requests,
                "repeats": repeats,
                "sample_interval_s": interval_s,
                "warmup_requests_per_run": warmup_requests,
                "max_run_s": max_run_s,
                "executed_repeats": len(reports),
                "warmup_policy": "longest prompt, unchanged output limit/EOS",
                "compare_stealing": compare_stealing,
                "planned_measured_passes": len(plan),
                "batch_sizes": batch_sizes,
                "goodput_limits": asdict(limits),
                "required_gpu_family": gpu_family,
                "min_gpu_memory_gib": min_gpu_memory_gib,
                "execution_policy": (
                    "independent; event-driven actors"
                    if not cpu
                    else "same multi-worker scheduler; synchronous ordinary workers"
                ),
            },
            "provenance": provenance,
            "frozen_files": frozen_files,
            "workload": asdict(workload),
            "runs": reports,
            "aggregate": {} if compare_stealing else summary["aggregate"],
            "cleanup": cleanup_state,
            "latency_across_runs": {} if compare_stealing else summary["latency_across_runs"],
            "limitations": [
                (
                    "Periodic worker state is cached from completed operations; snapshot ages and in-flight flags are included."
                    if not cpu
                    else "State samples are at tick boundaries and may miss transient peaks."
                ),
                "Polling/RPC and explicit CUDA synchronization overhead is included.",
                "Worker idle is wall time outside measured operations, not GPU idle. NCCL receive time precedes import execution; kv_handoffs records end-to-end driver-observed handoff time.",
                "GPU utilization is nvidia-smi's sampled window, not per-kernel timing.",
                "Time-limited reports censor unfinished requests; completed-only latencies are not a full baseline.",
                "Soft deadlines are checked between calls and may overrun one blocking operation.",
            ],
        }
    finally:
        import sys

        already_failing = sys.exc_info()[0] is not None
        cleanup_errors = []
        for w in workers:
            try:
                if cpu:
                    w.close()
                    closed = w.telemetry()
                else:
                    ray.get(w.close.remote(), timeout=10)
                    closed = ray.get(w.telemetry.remote(), timeout=10)
                cleanup_state.append(closed)
                if (
                    closed["active_request_ids"]
                    or closed["kv_blocks_used"]
                    or closed["held_requests"]
                ):
                    raise RuntimeError("Worker retained KV after close")
            except Exception as exc:
                cleanup_errors.append(exc)
        if ray is not None and ray.is_initialized():
            try:
                if collective_group is not None:
                    from ray.experimental.collective import destroy_collective_group

                    destroy_collective_group(collective_group)
            except Exception as exc:
                cleanup_errors.append(exc)
            finally:
                for w in workers:
                    ray.kill(w, no_restart=True)
                ray.shutdown()
        if cleanup_errors and not already_failing:
            raise RuntimeError(f"Worker cleanup failed: {cleanup_errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu", action="store_true", help="Small-model smoke check, not the GPU baseline"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/stage21-cpu-smoke"))
    parser.add_argument("--decode-batch-size", type=int, default=1)
    args = parser.parse_args()
    if not args.cpu:
        parser.error("Use the Modal runner for the GPU benchmark")
    if args.output.exists():
        parser.error("Choose a new output directory")
    root = Path(__file__).resolve().parents[2]
    report = run_benchmark(
        asdict(smoke_workload()),
        cpu=True,
        repeats=args.repeats,
        interval_s=0.01,
        provenance=source_identity(root),
        decode_batch_size=args.decode_batch_size,
    )
    save_report(report, args.output)
    print(f"Saved {args.output / 'baseline.json'}")


if __name__ == "__main__":
    main()
