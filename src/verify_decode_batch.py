"""CPU correctness and failure checks for continuous decode batching; --ray adds actors."""
import argparse
from unittest.mock import patch

import torch

from distengine.holding_pool import PrefillHoldingPool
from distengine.metric import MetricsRecorder
from distengine.multi_decode_coordinator import MultiDecodeCoordinator
from distengine.workers import DecodeWorker, PrefillWorker
from generation import generate_greedy
from verify_runner import raises
from verify_workers import make_model, check_empty


def check_worker():
    worker = DecodeWorker(make_model(), 48, 4)
    prompts = [[10], [20, 30, 40], [12] * 9, [50, 60]]
    limits = [3, 7, 5, 8]
    expected = {}
    try:
        for i, (prompt, limit) in enumerate(zip(prompts, limits)):
            expected[str(i)] = generate_greedy(worker.runner.model, torch.tensor([prompt]), limit, -1)[0].tolist()
            worker.prepare_local_fixture(str(i), prompt, limit, -1)
        before = worker.info()
        raises(ValueError, lambda: worker.step_batch([]))
        raises(ValueError, lambda: worker.step_batch(['0', '0']))
        raises(KeyError, lambda: worker.step_batch(['0', 'missing']))
        assert before == worker.info()
        forward_batches = []
        hook = worker.runner.model.register_forward_pre_hook(lambda model, args: forward_batches.append(args[0].shape[0]))
        completed = {}
        try:
            while worker.requests:
                ids = list(worker.requests)
                count = len(forward_batches)
                results = worker.step_batch(ids)
                assert len(forward_batches) == count + 1 and forward_batches[-1] == len(ids)
                for result in results:
                    if result['status'] == 'FINISHED':
                        completed[result['request_id']] = result
        finally:
            hook.remove()
        for rid, result in completed.items():
            assert result['prompt_token_ids'] + result['generated_token_ids'] == expected[rid]
        assert len(completed) == 4 and 4 in forward_batches and 1 in forward_batches
        check_empty(worker)
        # EOS in one row releases only that row.
        eos = expected['0'][2]  # Second generated token for the one-token prompt.
        worker.prepare_local_fixture('eos', prompts[0], 8, eos)
        worker.prepare_local_fixture('peer', prompts[1], 8, -1)
        result = worker.step_batch(['eos', 'peer'])
        assert result[0]['reason'] == 'eos' and 'peer' in worker.requests
    finally:
        worker.close()
    check_empty(worker)
    print('PASS: one forward per batch, unequal histories, block growth, shrinking batches, EOS and cleanup')


def check_failure():
    worker = DecodeWorker(make_model(), 2, 4)
    try:
        for rid in ('a', 'b'):
            worker.prepare_local_fixture(rid, [10] * 4, 4, -1)
        tables = {rid: blocks.copy() for rid, blocks in worker.runner.block_manager.request_blocks.items()}
        with patch.object(worker.runner.model, 'forward', side_effect=AssertionError('must not execute')):
            raises(RuntimeError, lambda: worker.step_batch(['a', 'b']))
        assert worker.runner.block_manager.request_blocks == tables
        assert all(c.sequence_length == 4 for c in worker.runner.caches.values())
    finally:
        worker.close()
    worker = DecodeWorker(make_model(), 12, 4)
    try:
        for rid in ('a', 'b', 'unrelated'):
            worker.prepare_local_fixture(rid, [10], 4, -1)
        # Fail after the first attention layer has already written every row.
        with patch.object(worker.runner.model.model.layers[1], 'forward', side_effect=RuntimeError('partial write')):
            raises(RuntimeError, lambda: worker.step_batch(['a', 'b']))
        assert set(worker.requests) == {'unrelated'}
        worker.step('unrelated')
    finally:
        worker.close()
    check_empty(worker)
    print('PASS: atomic capacity rejection, partial-forward batch retirement and unrelated-cache survival')


def check_coordinator(use_ray=False):
    ray = None
    metrics = MetricsRecorder()
    if use_ray:
        import ray
        from distengine.ray_workers import PrefillActor, DecodeActor
        from distengine.multi_decode_ray_coordinator import MultiDecodeRayCoordinator
        ray.init(num_cpus=4, include_dashboard=False)
        sources = [PrefillActor.remote(num_blocks=48, block_size=4, max_held_requests=4) for _ in range(2)]
        targets = [DecodeActor.remote(num_blocks=48, block_size=4) for _ in range(2)]
        coordinator = MultiDecodeRayCoordinator(sources, targets, decode_batch_size=4, metrics=metrics)
        for w in targets:
            ray.get(w.start_measurement.remote())
    else:
        sources = [PrefillWorker(make_model(), 48, 4, PrefillHoldingPool(4)) for _ in range(2)]
        targets = [DecodeWorker(make_model(), 48, 4) for _ in range(2)]
        coordinator = MultiDecodeCoordinator(sources, targets, decode_batch_size=4, metrics=metrics)
        for w in targets:
            w.start_measurement()
    try:
        for i in range(12):
            coordinator.submit(str(i), [10] * (i % 5 + 1), 8, -1)
        results = []
        for _ in range(150):
            if not coordinator.has_pending():
                break
            owners = dict(coordinator.decode_assignments)
            results.extend(coordinator.step())
            assert all(coordinator.decode_assignments[rid] == owner for rid, owner in owners.items())
        assert not coordinator.has_pending() and len(results) == 12
        model = make_model()
        for result in results:
            expected = generate_greedy(model, torch.tensor([result['prompt_token_ids']]), 8, -1)[0].tolist()
            assert result['prompt_token_ids'] + result['generated_token_ids'] == expected
        for worker in targets:
            info = ray.get(worker.telemetry.remote()) if use_ray else worker.telemetry()
            assert any(int(size) > 1 for size in info['decode_batch_sizes'])
            assert not info['active_request_ids'] and info['num_free_blocks'] == info['num_blocks']
        assert set(coordinator.decode_assignments.values()) == {0, 1}
        metrics.finish()
        assert len(metrics.requests) == 12
        for result in results:
            timing = metrics.requests[result['request_id']]
            assert len(timing.token_observed_s) == len(result['generated_token_ids'])
            assert timing.finished_s == timing.token_observed_s[-1]
            assert timing.error is None
    finally:
        for worker in (*sources, *targets):
            if use_ray:
                ray.get(worker.close.remote())
                ray.kill(worker)
            else:
                worker.close()
        if ray:
            ray.shutdown()
    print(f'PASS: {"Ray" if use_ray else "local"} dynamic admission, both batched decoders, sticky ownership and exact outputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ray', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    check_worker()
    check_failure()
    check_coordinator()
    if args.ray:
        check_coordinator(True)
