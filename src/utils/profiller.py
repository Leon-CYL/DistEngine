"""Opt-in decode microprofile; never enabled in normal benchmarks."""

from contextlib import ExitStack, contextmanager
import gzip
import json
from pathlib import Path
import statistics
import tempfile
import time
from unittest.mock import patch

import torch

from distengine.worker import DecodeWorker
from distengine.request import Request
from distengine.paged_kv_cache import BatchedPagedKVCache, PagedKVCache
from layers.attention import Attention


@contextmanager
def decode_ranges(model, sampler):
    """Temporary trace labels; restore methods even when profiling fails."""

    def wrap(original, label):
        def call(*args, **kwargs):
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)

        return call

    with ExitStack() as stack:
        for owner, method, label in (
            (BatchedPagedKVCache, "append", "decode/kv_append_gather_pad"),
            (PagedKVCache, "read", "decode/kv_read"),
            (Attention, "forward", "decode/attention"),
            (model, "forward", "decode/model_forward"),
            (type(sampler), "forward", "decode/greedy_sampling"),
        ):
            stack.enter_context(patch.object(owner, method, wrap(getattr(owner, method), label)))
        yield


def profile_decode(
    model,
    prompts,
    *,
    batch_size=64,
    warmup_steps=4,
    timing_steps=16,
    trace_steps=2,
    num_blocks=2048,
):
    """Profile the actual worker step_batch path on one model/GPU.

    CUDA-event intervals include default-stream scheduling gaps, not just kernel
    execution. PyTorch trace and event timing use separate, identical workloads.
    """
    for value in (batch_size, warmup_steps, timing_steps, trace_steps, num_blocks):
        if type(value) is not int or value <= 0:
            raise ValueError("Profile sizes and step counts must be positive integers")
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("Provide nonempty prompts")
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    steps = max(timing_steps, trace_steps)
    output_length = 1 + warmup_steps + steps + 1
    selected = [list(prompts[i % len(prompts)]) for i in range(batch_size)]
    required = sum((len(p) + output_length - 2 + 16) // 16 for p in selected)
    if required > num_blocks:
        raise ValueError(f"Profile requires {required} KV blocks; configured {num_blocks}")
    worker = DecodeWorker(model, num_blocks, 16)
    ids = [f"profile-{i}" for i in range(batch_size)]

    def sync():
        if cuda:
            torch.cuda.synchronize(device)

    def prepare():
        for rid, prompt in zip(ids, selected):
            request = Request(rid, prompt, output_length, -1)
            request.start()
            worker.runner.prefill(request)
            worker.requests[rid] = request
        for _ in range(warmup_steps):
            worker.step_batch(ids)
        sync()

    def release():
        for rid in list(worker.requests):
            worker.release(rid)
        assert worker.runner.block_manager.num_free_blocks == num_blocks

    try:
        prepare()
        pairs = (
            [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(timing_steps)
            ]
            if cuda
            else []
        )
        sync()
        start = time.perf_counter()
        for index in range(timing_steps):
            if cuda:
                begin, end = pairs[index]
                begin.record()
            worker.step_batch(ids)
            if cuda:
                end.record()
        sync()
        elapsed = time.perf_counter() - start
        event_ms = [begin.elapsed_time(end) for begin, end in pairs]
        expected = (
            [
                worker.requests[rid].generated_token_ids[: 1 + warmup_steps + trace_steps]
                for rid in ids
            ]
            if trace_steps <= timing_steps
            else None
        )
        release()
        prepare()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        with decode_ranges(model, worker.runner.sampler):
            with torch.profiler.profile(
                activities=activities, record_shapes=True, profile_memory=True, with_stack=False
            ) as prof:
                for _ in range(trace_steps):
                    with torch.profiler.record_function("decode/worker_step_batch"):
                        worker.step_batch(ids)
                    prof.step()
                sync()
        actual = [worker.requests[rid].generated_token_ids[:] for rid in ids]
        if expected is not None and actual != expected:
            raise RuntimeError("Profiled and unprofiled decode outputs differ")
        table = prof.key_averages().table(
            sort_by="self_cuda_time_total" if cuda else "self_cpu_time_total", row_limit=40
        )
        with tempfile.TemporaryDirectory() as temp:
            trace_path = Path(temp) / "decode.json"
            prof.export_chrome_trace(str(trace_path))
            trace = gzip.compress(trace_path.read_bytes())
        report = dict(
            cuda_kernel_events=sum(
                event.device_type == torch.autograd.DeviceType.CUDA for event in prof.events()
            ),
            scope="single decode worker; excludes prefill, transfer, Ray and coordinator",
            device=str(device),
            dtype=str(next(model.parameters()).dtype),
            gpu_name=torch.cuda.get_device_name(device) if cuda else None,
            torch_version=torch.__version__,
            batch_size=batch_size,
            prompt_lengths=[len(p) for p in selected],
            warmup_steps=warmup_steps,
            timing_steps=timing_steps,
            trace_steps=trace_steps,
            num_blocks=num_blocks,
            timing_wall_s=elapsed,
            decode_tokens=timing_steps * batch_size,
            worker_decode_tokens_per_s=timing_steps * batch_size / elapsed,
            cuda_event_ms=event_ms,
            mean_cuda_event_ms=statistics.mean(event_ms) if event_ms else None,
            median_cuda_event_ms=statistics.median(event_ms) if event_ms else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
            profile_matches_unprofiled=actual == expected if expected is not None else None,
            caveats=[
                "Profiler overhead affects the trace, not the separate timing pass.",
                "CUDA events measure stream elapsed time, including gaps; not pure kernel time.",
                "Context length grows each step; this is not full-engine throughput.",
            ],
        )
        release()
        report["kv_cleanup_passed"] = True
        return {
            "report.json": json.dumps(report, indent=2).encode(),
            "operators.txt": table.encode(),
            "decode.trace.json.gz": trace,
        }
    finally:
        worker.close()
