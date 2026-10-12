"""CPU-only sequential FIFO scheduler verification."""
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from generation import generate_greedy
from models.llama import LlamaForCausalLM
from distengine.request import Request, RequestStatus
from distengine.scheduler import Scheduler
from verify_runner import make_runner, raises


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32,
    )).eval()
    runner = make_runner(model, 2)
    scheduler = Scheduler(runner)
    assert scheduler.step() is None and not scheduler.has_pending_requests
    requests = [Request('A', [10, 20, 30, 40], 4, -1),
                Request('B', [50, 60], 1, -1),
                Request('C', [10, 20, 30], 3, -1)]
    expected = {r.request_id: generate_greedy(model, torch.tensor([r.prompt_token_ids]),
                                            r.max_new_tokens, r.eos_token_id)[0].tolist()
                for r in requests}
    for request in requests:
        scheduler.add_request(request)
    before = (list(scheduler.waiting), scheduler.request_ids.copy())
    raises(ValueError, lambda: scheduler.add_request(Request('A', [10], 1, -1)))
    raises(ValueError, lambda: scheduler.add_request(Request('huge', [10] * 9, 1, -1)))
    running = Request('invalid-state', [10], 1, -1)
    running.start()
    raises(ValueError, lambda: scheduler.add_request(running))
    generated = Request('invalid-tokens', [10], 1, -1)
    generated.generated_token_ids.append(20)
    raises(ValueError, lambda: scheduler.add_request(generated))
    assert (list(scheduler.waiting), scheduler.request_ids) == before
    order = []
    with patch.object(runner, 'prefill', wraps=runner.prefill) as prefill, \
         patch.object(runner, 'decode', wraps=runner.decode) as decode:
        while scheduler.has_pending_requests:
            calls = prefill.call_count + decode.call_count
            request = scheduler.step()
            assert prefill.call_count + decode.call_count == calls + 1
            assert request is not None
            order.append(request.request_id)
            assert len(scheduler.running) <= 1
            if request.is_finished:
                assert request.request_id not in runner.caches
                assert runner.block_manager.num_free_blocks == 2
            else:
                assert scheduler.running[0] is request
        calls = prefill.call_count + decode.call_count
        assert scheduler.step() is None
        assert prefill.call_count + decode.call_count == calls
    assert order == ['A'] * 4 + ['B'] + ['C'] * 3
    assert scheduler.finished == requests
    for request in requests:
        assert request.token_ids == expected[request.request_id]
    raises(ValueError, lambda: scheduler.add_request(Request('A', [10], 1, -1)))
    print('PASS: FIFO, one call per step, exact generation, rejection, release/reuse, empty steps')

    eos = int(model(torch.tensor([[10]]))[0, -1].argmax().item())
    request = Request('eos', [10], 4, eos)
    scheduler.add_request(request)
    assert scheduler.step() is request and request.reason == 'eos'
    assert runner.block_manager.num_free_blocks == 2
    runner.block_manager.allocate('registered-empty', 0)
    raises(ValueError, lambda: Scheduler(runner))
    runner.block_manager.free('registered-empty')
    active = Request('active', [10], 4, -1)
    active.start()
    runner.prefill(active)
    raises(ValueError, lambda: Scheduler(runner))
    runner.release(active)
    print('PASS: EOS stopping and nonempty runner/manager rejection')

    # Inject failures both before execution and after partial KV writes.
    for phase in ('prefill', 'decode'):
        for partial in (False, True):
            runner = make_runner(model, 2)
            scheduler = Scheduler(runner)
            failed = Request('failed', [10, 20, 30, 40], 3, -1)
            later = Request('later', [50], 1, -1)
            scheduler.add_request(failed)
            scheduler.add_request(later)
            if phase == 'decode':
                scheduler.step()
            def partial_forward(input_ids, positions=None, cache=None):
                k = torch.zeros(1, cache.num_kv_heads, input_ids.shape[1], cache.head_dim)
                cache.append(0, k, k)
                raise RuntimeError('Injected partial-write failure')
            if partial:
                context = patch.object(model, 'forward', side_effect=partial_forward)
            else:
                context = patch.object(runner, phase, side_effect=RuntimeError('Injected failure'))
            with context:
                raises(RuntimeError, scheduler.step)
            assert not scheduler.running and list(scheduler.waiting) == [later]
            assert not scheduler.finished
            assert failed.status == RequestStatus.RUNNING
            assert 'failed' in scheduler.request_ids
            assert not runner.caches and not runner.block_manager.request_blocks
            assert runner.block_manager.num_free_blocks == 2
            assert scheduler.step() is later and later.is_finished
            assert scheduler.finished == [later] and not scheduler.has_pending_requests
            assert runner.block_manager.num_free_blocks == 2
    print('PASS: prefill/decode failure cleanup and subsequent waiting-request execution')
    print('All scheduler checks passed.')


if __name__ == '__main__':
    main()
