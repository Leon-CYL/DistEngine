"""Real CPU Ray checks for independent phases, ownership, backpressure and failures."""
import os
os.environ['RAY_ENABLE_UV_RUN_RUNTIME_ENV'] = '0'
import sys
import time

import ray
import torch

from distengine.independent_ray_coordinator import IndependentRayCoordinator
from distengine.metric import MetricsRecorder
from distengine.ray_workers import PrefillActor, DecodeActor, PrefillActorImpl, DecodeActorImpl, build_test_model, scheduled_state
from distengine.telemetry import SystemSampler
from distengine.workload import replay_workload
from benchmark_stage21 import smoke_workload
from generation import generate_greedy


@ray.remote(num_cpus=0)
class Gate:
    def __init__(self):
        self.started = False
        self.open = False

    def enter(self):
        self.started = True

    def status(self):
        return self.started, self.open

    def release(self):
        self.open = True


class GatedPrefill(PrefillActorImpl):
    def __init__(self, gate):
        super().__init__(num_blocks=24, block_size=4, max_held_requests=2)
        self.gate = gate

    def scheduled(self, operation, *args, **kwargs):
        if operation == 'prefill' and args[0] == 'slow':
            ray.get(self.gate.enter.remote())
            while not ray.get(self.gate.status.remote())[1]:
                time.sleep(.005)
        return super().scheduled(operation, *args, **kwargs)


class FailingPrefill(PrefillActorImpl):
    def __init__(self):
        super().__init__(num_blocks=16, block_size=4, max_held_requests=2)
        self.fail_once = True

    def scheduled(self, operation, *args, **kwargs):
        if operation == 'release' and self.fail_once:
            self.fail_once = False
            return {'result': None, 'error': RuntimeError('release retry'),
                    'state': scheduled_state(self.worker)}
        return super().scheduled(operation, *args, **kwargs)


class FailingDecode(DecodeActorImpl):
    def __init__(self):
        super().__init__(num_blocks=16, block_size=4)
        self.fail_once = True

    def scheduled(self, operation, *args, **kwargs):
        if operation == 'step_batch' and self.fail_once:
            from unittest.mock import patch
            self.fail_once = False
            with patch.object(self.worker.runner.model.model.layers[1], 'forward',
                              side_effect=RuntimeError('partial decode failure')):
                return super().scheduled(operation, *args, **kwargs)
        return super().scheduled(operation, *args, **kwargs)


def drain(c, results, timeout=20):
    end = time.perf_counter() + timeout
    while c.has_pending() and time.perf_counter() < end:
        results.extend(c.step())
    assert not c.has_pending(), 'Independent coordinator did not drain'


def close(workers):
    ray.get([w.close.remote() for w in workers], timeout=20)
    infos = ray.get([w.ready.remote() for w in workers], timeout=10)
    assert all(i['closed'] and not i['active_request_ids'] and i['num_free_blocks'] == i['num_blocks'] for i in infos)
    for w in workers:
        ray.kill(w)


def check_overlap():
    gate = Gate.remote()
    sources = [ray.remote(num_cpus=1)(GatedPrefill).remote(gate),
               PrefillActor.remote(num_blocks=24, block_size=4, max_held_requests=2)]
    targets = [DecodeActor.remote(num_blocks=24, block_size=4) for _ in range(2)]
    workers = sources + targets
    results = []
    metrics = MetricsRecorder()
    try:
        c = IndependentRayCoordinator(sources, targets, metrics=metrics, decode_batch_size=4)
        c.submit('fast-a', [10], 24, -1)
        c.submit('fast-b', [20, 30], 24, -1)
        end = time.perf_counter() + 15
        while not all(len(metrics.requests[r].token_observed_s) >= 2 for r in ('fast-a', 'fast-b')):
            assert time.perf_counter() < end
            results.extend(c.step())
        c.submit('slow', [10] * 9, 8, -1)
        c.submit('immediate-peer', [40], 1, -1)
        end = time.perf_counter() + 15
        while not ray.get(gate.status.remote())[0]:
            assert time.perf_counter() < end
            results.extend(c.step())
        # A sample must not enqueue an RPC behind the blocked prefill.
        sampler = SystemSampler(c, lambda: (_ for _ in ()).throw(AssertionError('blocking telemetry RPC')))
        sampler.sample()
        assert sampler.samples[0]['workers'][0]['operation_in_flight']
        end = time.perf_counter() + 15
        required = {'fast-a', 'fast-b', 'immediate-peer'}
        while not required.issubset({r['request_id'] for r in results}):
            assert time.perf_counter() < end, 'Decode/peer prefill blocked on gated source'
            results.extend(c.step())
        assert len(metrics.requests['slow'].token_observed_s) == 0
        assert not ray.get(gate.status.remote())[1]
        print('PASS: both decoders and peer prefill finish while another prefill remains gated; telemetry never waits', flush=True)
        ray.get(gate.release.remote())
        drain(c, results)
        assert len(results) == 4
        model = build_test_model(0)
        for result in results:
            limit = 24 if result['request_id'].startswith('fast') else 8 if result['request_id'] == 'slow' else 1
            expected = generate_greedy(model, torch.tensor([result['prompt_token_ids']]), limit, -1)[0].tolist()
            assert result['prompt_token_ids'] + result['generated_token_ids'] == expected
            timing = metrics.requests[result['request_id']]
            assert len(timing.token_observed_s) == len(result['generated_token_ids'])
            assert timing.finished_s == timing.token_observed_s[-1]
        assert len(c.decode_assignments) == 3 and not c.pending_releases
        metrics.finish()
    finally:
        ray.get(gate.release.remote())
        close(workers)
        ray.kill(gate)
    print('PASS: exact outputs, immediate completion, per-token metrics, acknowledgements and full cleanup', flush=True)


def check_pressure_and_replay(workers_per_role=2):
    sources = [PrefillActor.remote(num_blocks=6, block_size=4, max_held_requests=1) for _ in range(workers_per_role)]
    targets = [DecodeActor.remote(num_blocks=5, block_size=4) for _ in range(workers_per_role)]
    workers = sources + targets
    try:
        c = IndependentRayCoordinator(sources, targets, decode_batch_size=4)
        results = []
        for i in range(18):
            c.submit(str(i), [10] * (i % 4 + 1), 8, -1)
        owners = {}
        saw_held = saw_upstream = False
        end = time.perf_counter() + 20
        while c.has_pending() and time.perf_counter() < end:
            results.extend(c.step())
            assert all(c.decode_assignments[rid] == owner for rid, owner in owners.items())
            owners.update(c.decode_assignments)
            snapshots = c.cached_workers()
            assert all(s['held_requests'] <= 1 for s in snapshots[:workers_per_role])
            assert all(0 <= s['num_free_blocks'] <= s['num_blocks'] for s in snapshots)
            saw_held |= any(s['held_requests'] for s in snapshots[:workers_per_role])
            saw_upstream |= any(c.waiting)
        assert not c.has_pending() and len(results) == len({r['request_id'] for r in results}) == 18
        assert saw_held and saw_upstream and set(owners.values()) == set(range(workers_per_role))
        model = build_test_model(0)
        for r in results:
            expected = generate_greedy(model, torch.tensor([r['prompt_token_ids']]), 8, -1)[0].tolist()
            assert r['prompt_token_ids'] + r['generated_token_ids'] == expected
        # Fresh coordinator on the same drained actors: warm-up/repeat lifecycle.
        c = IndependentRayCoordinator(sources, targets, metrics=MetricsRecorder(), decode_batch_size=1)
        outputs = replay_workload(smoke_workload(), c)
        assert len(outputs) == 6 and not c.has_pending() and c.metrics.finished_at is not None
    finally:
        close(workers)
    print('PASS: bounded holding/KV pressure, stable ownership, exact mixed outputs, replay and actor reuse', flush=True)


def check_failures():
    sources = [ray.remote(num_cpus=1)(FailingPrefill).remote(),
               PrefillActor.remote(num_blocks=16, block_size=4)]
    targets = [ray.remote(num_cpus=1)(FailingDecode).remote(),
               DecodeActor.remote(num_blocks=16, block_size=4)]
    workers = sources + targets
    metrics = MetricsRecorder()
    try:
        c = IndependentRayCoordinator(sources, targets, metrics=metrics, decode_batch_size=4)
        for i in range(6):
            c.submit('failure-' + str(i), [10], 6, -1)
        results = []
        errors = []
        end = time.perf_counter() + 20
        while c.has_pending() and time.perf_counter() < end:
            try:
                results.extend(c.step())
            except RuntimeError as exc:
                errors.append(str(exc))
        results.extend(c.drain_completed())
        assert not c.has_pending()
        assert any('release retry' in e for e in errors)
        assert any('partial decode failure' in e for e in errors)
        failed = [r for r in metrics.requests.values() if r.error is not None]
        assert failed and len(results) + len(failed) == 6
        assert len({r['request_id'] for r in results}) == len(results)
        assert not c.pending_releases and not c._transfers
        assert c.decode_assignments and set(c.decode_assignments.values()) == {0, 1}
    finally:
        close(workers)
    print('PASS: failed source release retries without duplicate import; damaged decode rows retire and peers drain', flush=True)


def check_deadline():
    gate = Gate.remote()
    source = ray.remote(num_cpus=1)(GatedPrefill).remote(gate)
    target = DecodeActor.remote(num_blocks=24, block_size=4)
    try:
        from distengine.workload import Workload, ScheduledRequest
        metrics = MetricsRecorder()
        c = IndependentRayCoordinator([source], [target], metrics=metrics)
        workload = Workload(1, 'constant', 1, 42,
                            (ScheduledRequest('slow', 0, (10,), 8, -1),))
        start = time.perf_counter()
        outputs = replay_workload(workload, c, deadline=start + .1)
        assert not outputs and c.has_pending() and metrics.finished_at is not None
        assert time.perf_counter() - start < 2
    finally:
        ray.get(gate.release.remote())
        close([source, target])
        ray.kill(gate)
    print('PASS: replay deadline returns with prefill in flight; unfinished work closes and frees KV', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers-per-role', type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ray.init(address='local', num_cpus=2 * args.workers_per_role, include_dashboard=False,
             runtime_env={'py_executable': sys.executable})
    try:
        check_overlap()
        check_pressure_and_replay(args.workers_per_role)
        check_failures()
        check_deadline()
    finally:
        ray.shutdown()
