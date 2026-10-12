"""Verify engine integration on CPU, optionally with the pinned TinyLlama."""
import argparse
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from distengine.engine import LLMEngine
from generation import generate_greedy
from models.llama import LlamaForCausalLM
from distengine.request import RequestStatus
from verify_runner import raises


def check_empty(engine):
    assert not engine.has_pending_requests
    assert not engine.runner.caches
    assert not engine.block_manager.request_blocks
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks
    assert engine.step() is None
    assert engine.run_until_complete() == []


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32,
    ))
    engine = LLMEngine(model, 2, 4)
    assert engine.runner.model is model
    assert engine.runner.block_manager is engine.block_manager
    assert engine.runner.storage is engine.storage
    assert engine.scheduler.runner is engine.runner
    assert engine.storage.k_cache.shape == (2, 2, 2, 4, 4)
    assert engine.storage.k_cache.dtype == next(model.parameters()).dtype
    assert engine.storage.k_cache.device == next(model.parameters()).device
    assert not model.training
    check_empty(engine)
    for blocks, size in ((0, 4), (2, -1), (True, 4), (2, 1.5)):
        raises(ValueError, lambda: LLMEngine(model, blocks, size))
    print('PASS: component wiring, storage, construction validation, empty engine')

    prompts = [[10, 20, 30], [40], [50, 60]]
    limits = [4, 1, 3]
    requests = [engine.add_request(str(i), prompt, limit, -1)
                for i, (prompt, limit) in enumerate(zip(prompts, limits))]
    assert all(r.status == RequestStatus.WAITING for r in requests)
    expected = [generate_greedy(model, torch.tensor([prompt]), limit, -1)[0].tolist()
                for prompt, limit in zip(prompts, limits)]
    before = (list(engine.scheduler.waiting), engine.scheduler.request_ids.copy())
    for args in [('0', [10], 1, -1), ('huge', [10] * 9, 1, -1),
                 ('invalid', [64], 1, -1), ('empty', [], 1, -1),
                 ('bad-limit', [10], 0, -1)]:
        raises(ValueError, lambda: engine.add_request(*args))
        assert (list(engine.scheduler.waiting), engine.scheduler.request_ids) == before
    assert engine.step() is requests[0]
    assert engine.has_pending_requests
    completed = engine.run_until_complete()
    assert completed == requests
    assert [r.token_ids for r in requests] == expected
    assert all(r.reason == 'length' for r in requests)
    completed.clear()
    assert engine.scheduler.finished == requests
    check_empty(engine)
    print('PASS: FIFO, baseline agreement, manual stepping, rejection atomicity, copied results')

    # Maximum cached history fits exactly: 5 prompt + 4 output - 1 = 8.
    boundary = engine.add_request('boundary', [10, 20, 30, 40, 50], 4, -1)
    assert engine.run_until_complete() == [boundary]
    assert boundary.token_ids == generate_greedy(
        model, torch.tensor([[10, 20, 30, 40, 50]]), 4, -1)[0].tolist()
    eos = int(model(torch.tensor([[10]]))[0, -1].argmax().item())
    stopped = engine.add_request('eos', [10], 4, eos)
    assert engine.run_until_complete() == [stopped]
    assert stopped.reason == 'eos' and stopped.num_generated_tokens == 1
    check_empty(engine)
    print('PASS: repeated use, new-only completion history, exact capacity, EOS, resource reuse')

    # Inject a failure after writing layer zero during both prefill and decode.
    for phase in ('prefill', 'decode'):
        engine = LLMEngine(model, 2, 4)
        failed = engine.add_request('failed', [10, 20, 30, 40], 3, -1)
        later = engine.add_request('later', [50], 1, -1)
        if phase == 'decode':
            engine.step()
        def fail_forward(input_ids, positions=None, cache=None):
            k = torch.zeros(1, cache.num_kv_heads, input_ids.shape[1], cache.head_dim)
            cache.append(0, k, k)
            raise RuntimeError('Injected partial-write failure')
        with patch.object(model, 'forward', side_effect=fail_forward):
            raises(RuntimeError, engine.run_until_complete)
        assert not engine.runner.caches and not engine.block_manager.request_blocks
        assert not engine.scheduler.running
        assert list(engine.scheduler.waiting) == [later]
        assert failed not in engine.scheduler.finished
        assert engine.block_manager.num_free_blocks == 2
        assert engine.run_until_complete() == [later]
        assert later.token_ids == generate_greedy(model, torch.tensor([[50]]), 1, -1)[0].tolist()
        check_empty(engine)
    print('PASS: prefill/decode failure cleanup and recovery of queued work')
    print('All CPU engine checks passed.', flush=True)


@torch.inference_mode()
def check_checkpoint():
    from transformers import AutoTokenizer
    from hf_reference import choose_device
    from load_model import MODEL_ID, REVISION, load_model

    device = choose_device()
    print(f'Engine checkpoint: {REVISION}; device: {device}; dtype: float32', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = load_model(device)
    prompts = [
        'What is the capital of France?',
        'Explain why the sky is blue in two sentences.',
        'Complete this sentence: The best way to learn programming is',
    ]
    inputs = []
    for prompt in prompts:
        formatted = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        inputs.append(tokenizer(formatted, add_special_tokens=False,
                                return_tensors='pt')['input_ids'].to(device))
    expected = [generate_greedy(model, ids, 32, tokenizer.eos_token_id)[0].tolist()
                for ids in inputs]
    num_blocks = max((ids.shape[1] + 32 - 1 + 3) // 4 for ids in inputs)
    engine = LLMEngine(model, num_blocks, 4)
    requests = [engine.add_request(f'checkpoint-{i}', ids[0].tolist(), 32,
                                   tokenizer.eos_token_id)
                for i, ids in enumerate(inputs)]
    assert engine.run_until_complete() == requests
    for prompt, request, tokens in zip(prompts, requests, expected):
        assert request.token_ids == tokens
        print(f'PASS: engine exact baseline agreement ({request.num_generated_tokens} tokens): {prompt}',
              flush=True)
    check_empty(engine)
    print('All real-checkpoint engine checks passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='store_true',
                        help='Also compare engine output with the pinned TinyLlama baseline')
    args = parser.parse_args()
    main()
    if args.checkpoint:
        check_checkpoint()
