"""Small CPU assertions for ModelRunner; no model download required."""
import torch
from transformers import LlamaConfig

from .block_manager import BlockManager
from .generation import generate_greedy
from .model_runner import ModelRunner
from .models.llama import LlamaForCausalLM
from .paged_kv_cache import KVStorage
from .request import Request


def raises(error_type, operation):
    try:
        operation()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def check_release(runner, request):
    owned = len(runner.block_manager.get_block_table(request.request_id))
    free_before = runner.block_manager.num_free_blocks
    tokens_before = request.token_ids
    status_before = request.status
    reason_before = request.reason
    runner.release(request)
    assert request.request_id not in runner.caches
    assert request.request_id not in runner.block_manager.request_blocks
    assert runner.block_manager.num_free_blocks == free_before + owned
    assert request.token_ids == tokens_before
    assert request.status == status_before
    assert request.reason == reason_before
    raises(KeyError, lambda: runner.release(request))


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=32)
    model = LlamaForCausalLM(config).eval()
    manager = BlockManager(8, 4)
    storage = KVStorage(2, 2, 4, 8, 4, torch.device('cpu'), torch.float32)
    runner = ModelRunner(model, manager, storage)
    ids, positions = runner.prepare_inputs([10, 20, 30], 0)
    assert ids.tolist() == [[10, 20, 30]]
    assert positions.tolist() == [[0, 1, 2]]
    ids, positions = runner.prepare_inputs([40], 3)
    assert ids.tolist() == [[40]] and positions.tolist() == [[3]]
    assert ids.dtype == positions.dtype == torch.long
    assert ids.device == positions.device == runner.device
    raises(ValueError, lambda: runner.prepare_inputs([], 0))
    raises(ValueError, lambda: runner.prepare_inputs([True], 0))
    raises(ValueError, lambda: runner.prepare_inputs([64], 0))
    raises(ValueError, lambda: runner.prepare_inputs([10], -1))
    print('PASS: input tensors, positions, and validation')

    requests = [Request('A', [10, 20, 30], 4, -1),
                Request('B', [40, 50, 60], 4, -1)]
    expected = {r.request_id: generate_greedy(model, torch.tensor([r.prompt_token_ids]),
                                            r.max_new_tokens, r.eos_token_id)[0].tolist()
                for r in requests}
    for r in requests:
        raises(ValueError, lambda: runner.prefill(r))
        r.start()
        raises(KeyError, lambda: runner.decode(r))
        token = runner.prefill(r)
        cache = runner.caches[r.request_id]
        assert cache.sequence_length == r.num_prompt_tokens
        assert r.num_generated_tokens == 1 and r.generated_token_ids[-1] == token
        raises(ValueError, lambda: runner.prefill(r))
    assert set(manager.get_block_table('A')).isdisjoint(manager.get_block_table('B'))
    while any(not r.is_finished for r in requests):
        for r in requests:
            if not r.is_finished:
                before = runner.caches[r.request_id].sequence_length
                count = r.num_generated_tokens
                token = runner.decode(r)
                cache = runner.caches[r.request_id]
                assert cache.sequence_length == before + 1 == len(r.token_ids) - 1
                assert r.num_generated_tokens == count + 1
                assert r.generated_token_ids[-1] == token
                assert len(manager.get_block_table(r.request_id)) == (cache.sequence_length + 3) // 4
    for r in requests:
        assert r.token_ids == expected[r.request_id]
        raises(ValueError, lambda: runner.decode(r))
        check_release(runner, r)
    assert manager.num_free_blocks == manager.num_blocks
    print('PASS: prefill/decode, block growth, interleaved generation agreement, release')

    short = Request('reuse', [10, 20, 30], 1, -1)
    short.start()
    runner.prefill(short)
    assert short.is_finished and short.reason == 'length'
    assert runner.caches['reuse'].sequence_length == 3
    assert short.token_ids == generate_greedy(model, torch.tensor([[10, 20, 30]]), 1, -1)[0].tolist()
    check_release(runner, short)
    eos = int(model(torch.tensor([[10, 20, 30]]))[0, -1].argmax().item())
    stopped = Request('eos', [10, 20, 30], 4, eos)
    stopped.start()
    runner.prefill(stopped)
    assert stopped.is_finished and stopped.reason == 'eos'
    check_release(runner, stopped)
    unfinished = Request('unfinished', [10], 4, -1)
    unfinished.start()
    runner.prefill(unfinished)
    assert not unfinished.is_finished
    check_release(runner, unfinished)
    assert manager.num_free_blocks == manager.num_blocks
    print('PASS: block reuse, output-limit/EOS stopping, unfinished cleanup, repeated release')
    print('All CPU runner checks passed.')


if __name__ == '__main__':
    main()
