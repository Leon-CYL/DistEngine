"""Small CPU assertions for ModelRunner; no model download required."""
import argparse
from unittest.mock import patch

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
    check_failures(model)
    print('All CPU runner checks passed.')


def make_runner(model, blocks):
    config = model.config
    parameter = next(model.parameters())
    heads = config.num_key_value_heads or config.num_attention_heads
    manager = BlockManager(blocks, 4)
    storage = KVStorage(config.num_hidden_layers, heads,
                        config.hidden_size // config.num_attention_heads,
                        blocks, 4, parameter.device, parameter.dtype)
    return ModelRunner(model, manager, storage)


def state(runner, request):
    cache = runner.caches[request.request_id]
    return (request.token_ids, request.status, request.reason,
            cache.lengths.copy(), cache.block_table.copy(),
            runner.block_manager.free_blocks.copy(),
            {k: v.copy() for k, v in runner.block_manager.request_blocks.items()})


@torch.inference_mode()
def check_failures(model):
    runner = make_runner(model, 2)
    active = Request('active', [10, 20, 30, 40], 4, -1)
    active.start()
    runner.prefill(active)
    blocker = Request('blocker', [50], 4, -1)
    blocker.start()
    runner.prefill(blocker)
    before = state(runner, active)
    cached = [tuple(t.clone() for t in runner.caches['active'].read(i)) for i in range(2)]
    waiting = Request('waiting', [10], 4, -1)
    waiting.start()
    raises(RuntimeError, lambda: runner.prefill(waiting))
    assert waiting.generated_token_ids == [] and waiting.request_id not in runner.caches
    assert state(runner, active) == before
    raises(RuntimeError, lambda: runner.decode(active))
    assert state(runner, active) == before
    for i, pair in enumerate(cached):
        assert all(torch.equal(a, b) for a, b in zip(pair, runner.caches['active'].read(i)))
    runner.release(blocker)
    runner.decode(active)  # Retry succeeds after capacity becomes available.
    runner.release(active)
    assert runner.block_manager.num_free_blocks == 2
    print('PASS: admission/decode exhaustion preserves state and supports retry')

    runner = make_runner(model, 6)
    survivor = Request('survivor', [50], 4, -1)
    survivor.start()
    runner.prefill(survivor)
    survivor_tokens = survivor.token_ids
    survivor_cache = [tuple(t.clone() for t in runner.caches['survivor'].read(i)) for i in range(2)]
    observed = []
    def fail_after_first_layer(input_ids, positions=None, cache=None):
        k = torch.zeros(1, cache.num_kv_heads, input_ids.shape[1], cache.head_dim,
                        device=cache.device, dtype=cache.dtype)
        cache.append(0, k, k)
        observed.append(cache.lengths.copy())
        raise RuntimeError('Injected failure after partial KV writes')

    for phase in ('prefill', 'decode'):
        request = Request(phase, [10, 20, 30, 40], 4, -1)
        request.start()
        free_before = runner.block_manager.num_free_blocks
        if phase == 'decode':
            runner.prefill(request)
        tokens_before = request.token_ids
        with patch.object(model, 'forward', side_effect=fail_after_first_layer):
            raises(RuntimeError, lambda: getattr(runner, phase)(request))
        assert observed[-1][0] != observed[-1][1]
        assert request.token_ids == tokens_before
        assert request.request_id not in runner.caches
        assert request.request_id not in runner.block_manager.request_blocks
        assert runner.block_manager.num_free_blocks == free_before
        assert survivor.token_ids == survivor_tokens
        for i, pair in enumerate(survivor_cache):
            assert all(torch.equal(a, b) for a, b in zip(pair, runner.caches['survivor'].read(i)))
    runner.decode(survivor)
    runner.release(survivor)
    assert runner.block_manager.num_free_blocks == 6
    print('PASS: prefill/decode partial-write failures free blocks and preserve other requests')


@torch.inference_mode()
def check_checkpoint():
    from transformers import AutoTokenizer
    from .hf_reference import choose_device
    from .load_model import MODEL_ID, REVISION, load_model

    device = choose_device()
    print(f'Runner checkpoint: {REVISION}; device: {device}; dtype: float32', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = load_model(device)
    prompts = [
        'What is the capital of France?',
        'Explain why the sky is blue in two sentences.',
        'Complete this sentence: The best way to learn programming is',
    ]
    for index, prompt in enumerate(prompts):
        formatted = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(formatted, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)
        expected = generate_greedy(model, ids, 32, tokenizer.eos_token_id)[0].tolist()
        runner = make_runner(model, (ids.shape[1] + 32 + 3) // 4)
        request = Request(f'checkpoint-{index}', ids[0].tolist(), 32, tokenizer.eos_token_id)
        request.start()
        try:
            runner.prefill(request)
            while not request.is_finished:
                runner.decode(request)
            assert request.token_ids == expected
            assert runner.caches[request.request_id].sequence_length == len(expected) - 1
            print(f'PASS: runner exact baseline agreement ({request.num_generated_tokens} tokens): {prompt}', flush=True)
        finally:
            if request.request_id in runner.caches:
                runner.release(request)
        assert runner.block_manager.num_free_blocks == runner.block_manager.num_blocks
    print('All real-checkpoint runner checks passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='store_true',
                        help='Also compare the runner with the pinned TinyLlama baseline')
    args = parser.parse_args()
    main()
    if args.checkpoint:
        check_checkpoint()
