"""Deterministic policy checks and real Ray stealing-disabled/enabled replay."""
import os
os.environ['RAY_ENABLE_UV_RUN_RUNTIME_ENV'] = '0'
from collections import deque
import sys
import time

import ray
import torch

from distengine.coordinator import Submission
from distengine.independent_ray_coordinator import IndependentRayCoordinator
from distengine.metric import MetricsRecorder
from distengine.ray_workers import PrefillActor, DecodeActor, build_test_model
from generation import generate_greedy
from verify_independent import Gate, GatedPrefill, close
from verify_runner import raises


def check_policy():
    c = object.__new__(IndependentRayCoordinator)
    c.stealing_enabled = True
    c.steal_events = []
    c.metrics = None
    c._busy = {0: 'prefill-ref'}
    c._prefill_started = {'active'}
    c._states = [dict(closed=False,block_size=4,num_blocks=8,num_free_blocks=8,
                      held_requests=0,max_held_requests=2,vocab_size=64) for _ in range(3)]
    def request(rid, length=2, output=4):
        return Submission(rid, tuple([10] * length), output, -1)
    active, tail, other = request('active'), request('tail', 8), request('other')
    c.waiting = [deque([active, tail]), deque(), deque([other])]
    c._steal_prefill(1)
    assert list(c.waiting[0]) == [active] and list(c.waiting[1]) == [tail]
    assert c.steal_events[0]['from_source'] == 0
    assert active.request_id in c._prefill_started
    # Receivers with existing work or an outstanding operation cannot steal.
    c._steal_prefill(1)
    c._busy[1] = 'busy'; c.waiting[1].clear(); c._steal_prefill(1)
    assert len(c.steal_events) == 1
    del c._busy[1]
    for key,value in [('num_free_blocks',0),('held_requests',2),('closed',True),('num_blocks',1)]:
        before=c._states[1][key]; c._states[1][key]=value
        c.waiting[2]=deque([request('large',8,8)])
        c._steal_prefill(1)
        assert not c.waiting[1]
        c._states[1][key]=before
    c.stealing_enabled=False; c._steal_prefill(1)
    assert not c.waiting[1]
    raises(ValueError, lambda: IndependentRayCoordinator([], [], stealing_enabled=1))
    print('PASS: token-weighted donor selection, unstarted tail only, idle/empty receiver, capacity and disabled guards',flush=True)


def exercise(enabled):
    gate=Gate.remote()
    sources=[ray.remote(num_cpus=1)(GatedPrefill).remote(gate),
             PrefillActor.remote(num_blocks=24,block_size=4,max_held_requests=2)]
    targets=[DecodeActor.remote(num_blocks=24,block_size=4) for _ in range(2)]
    workers=sources+targets
    results=[]
    metrics=MetricsRecorder()
    specifications=[('slow',[10]*9,8),('peer',[20],1),('queued-a',[30]*3,4),
                    ('peer2',[40],1),('queued-b',[50]*5,4),('peer3',[60],1)]
    try:
        c=IndependentRayCoordinator(sources,targets,metrics=metrics,decode_batch_size=4,
                                    stealing_enabled=enabled)
        for rid,prompt,limit in specifications:c.submit(rid,prompt,limit,-1)
        deadline=time.perf_counter()+15
        while not ray.get(gate.status.remote())[0]:
            assert time.perf_counter()<deadline
            results.extend(c.step())
        expected={'peer','peer2','peer3'} | ({'queued-a','queued-b'} if enabled else set())
        while not expected.issubset({r['request_id'] for r in results}):
            assert time.perf_counter()<deadline
            results.extend(c.step())
        if enabled:
            assert {e['request_id'] for e in c.steal_events} == {'queued-a','queued-b'}
            assert all(e['from_source']==0 and e['to_source']==1 for e in c.steal_events)
            assert c.prefill_assignments['slow']==0
            assert c.prefill_assignments['queued-a']==c.prefill_assignments['queued-b']==1
        else:
            assert not c.steal_events
            assert not ({'queued-a','queued-b'} & {r['request_id'] for r in results})
        # Neither the in-flight prompt nor held/decode KV can be stolen.
        assert not metrics.requests['slow'].token_observed_s
        assert all(e['request_id']!='slow' for e in c.steal_events)
        owners=dict(c.decode_assignments)
        ray.get(gate.release.remote())
        while c.has_pending() and time.perf_counter()<deadline:
            results.extend(c.step())
            assert all(c.decode_assignments[rid]==owner for rid,owner in owners.items())
        assert not c.has_pending() and len(results)==len({r['request_id'] for r in results})==6
        assert len(c.prefill_assignments)==6
        assert not c.pending_releases and not c._transfers
        model=build_test_model(0)
        by_id={r['request_id']:r for r in results}
        for rid,prompt,limit in specifications:
            expected_ids=generate_greedy(model,torch.tensor([prompt]),limit,-1)[0].tolist()
            assert prompt+by_id[rid]['generated_token_ids']==expected_ids
            t=metrics.requests[rid]
            assert len(t.token_observed_s)==limit and t.finished_s==t.token_observed_s[-1]
            assert t.error is None
        metrics.finish()
        recorded = {rid:r['generated_token_ids'] for rid,r in by_id.items()}
        # Replay the same reproducible arrival schedule on drained actors.
        from benchmark_stage21 import smoke_workload
        from distengine.workload import replay_workload
        replay_metrics = MetricsRecorder()
        replay = IndependentRayCoordinator(sources, targets, metrics=replay_metrics,
                                           decode_batch_size=4, stealing_enabled=enabled)
        outputs = replay_workload(smoke_workload(), replay)
        assert len(outputs) == 6 and not replay.has_pending()
        assert replay_metrics.finished_at is not None
        for result in outputs:
            rid = result['request_id']
            assert len(replay_metrics.requests[rid].token_observed_s) == len(result['generated_token_ids'])
            recorded[rid] = result['generated_token_ids']
        return recorded
    finally:
        ray.get(gate.release.remote())
        close(workers)
        ray.kill(gate)


if __name__=='__main__':
    torch.set_num_threads(1)
    check_policy()
    ray.init(address='local',num_cpus=4,include_dashboard=False,runtime_env={'py_executable':sys.executable})
    try:
        off=exercise(False)
        on=exercise(True)
        assert off==on
        print('PASS: enabled stealing progresses while donor is gated; disabled/enabled exact outputs, metrics, fixed decode ownership and KV cleanup',flush=True)
    finally:
        ray.shutdown()
