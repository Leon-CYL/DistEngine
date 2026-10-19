"""Stage 19 frozen-input integrity, selection, capacity and CPU output checks."""

from collections import Counter
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import torch
from transformers import LlamaConfig

from distengine.holding_pool import PrefillHoldingPool
from distengine.mixed_workload import Prompt, generate_mixed_workload
from distengine.multi_prefill_coordinator import MultiPrefillCoordinator
from distengine.workers import PrefillWorker, DecodeWorker
from distengine.workload import Workload, generate_workload, load_workload, replay_workload, save_workload
from models.llama import LlamaForCausalLM
from generation import generate_greedy
from freeze_workload import BOUNDS, MODEL_ID, REVISION, digest, freeze
from verify_runner import raises
from verify_workers import check_empty
from verify_workload import FakeClock

ROOT = Path(__file__).resolve().parents[1] / "workloads" / "stage19"


def check_frozen(root=ROOT):
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["revision"] == REVISION and manifest["model_id"] == MODEL_ID
    for name in ("workload", "prompt_bank"):
        entry = manifest[name]
        assert digest(root / entry["file"]) == entry["sha256"]
    bank_data = json.loads((root / manifest["prompt_bank"]["file"]).read_text())
    assert bank_data["revision"] == REVISION
    bank = tuple(Prompt(p["prompt_id"], p["length_class"], tuple(p["token_ids"]), p["text"])
                 for p in bank_data["prompts"])
    assert len(bank) == 9 and len({p.token_ids for p in bank}) == 9
    for p in bank:
        minimum, maximum = BOUNDS[p.length_class]
        assert minimum <= len(p.token_ids) <= maximum
    workload = load_workload(root / manifest["workload"]["file"])
    state = random.getstate()
    assert generate_mixed_workload(prompt_bank=bank, **manifest["settings"]) == workload
    assert random.getstate() == state
    assert len(workload.requests) == 100
    by_tokens = {p.token_ids: p for p in bank}
    assignments = [dict(request_id=r.request_id, prompt_id=by_tokens[r.prompt_token_ids].prompt_id,
                        length_class=by_tokens[r.prompt_token_ids].length_class) for r in workload.requests]
    assert assignments == manifest["request_prompts"]
    counts = dict(Counter(a["length_class"] for a in assignments))
    assert counts == manifest["class_counts"] == {"short": 60, "medium": 30, "long": 10}
    totals = [sum(len(r.prompt_token_ids) for r in workload.requests[i::2]) for i in range(2)]
    assert totals == manifest["round_robin"]["prompt_tokens_per_source"]
    assert max(totals) / min(totals) == manifest["round_robin"]["max_to_min_ratio"]
    print("PASS: frozen hashes, nine length-checked prompts, exact quotas, assignments and round-robin totals")
    return bank, manifest


def check_selection(bank, manifest):
    settings = manifest["settings"]
    original = generate_mixed_workload(prompt_bank=bank, **settings)
    changed = generate_mixed_workload(prompt_bank=bank, **(settings | {"prompt_seed": 20}))
    assert [r.arrival_offset_s for r in original.requests] == [r.arrival_offset_s for r in changed.requests]
    assert [r.prompt_token_ids for r in original.requests] != [r.prompt_token_ids for r in changed.requests]
    changed = generate_mixed_workload(prompt_bank=bank, **(settings | {"arrival_seed": 43}))
    assert [r.prompt_token_ids for r in original.requests] == [r.prompt_token_ids for r in changed.requests]
    assert [r.arrival_offset_s for r in original.requests] != [r.arrival_offset_s for r in changed.requests]
    for mode in ("constant", "poisson"):
        for count in (0, 1, 7):
            result = generate_mixed_workload(prompt_bank=bank, **(settings | {"arrival_mode": mode, "num_requests": count}))
            assert len(result.requests) == count
    for changes in (
        {"prompt_seed": True}, {"num_requests": -1}, {"class_weights": {}},
        {"class_weights": {"short": 0}}, {"class_weights": {"short": float("nan")}},
        {"class_weights": {"short": -1}}, {"kv_capacity_tokens": 32},
        {"max_context_tokens": 32}, {"vocab_size": 2}, {"eos_token_id": -2},
    ):
        raises(ValueError, lambda: generate_mixed_workload(prompt_bank=bank, **(settings | changes)))
    raises(ValueError, lambda: generate_mixed_workload(prompt_bank=(), **settings))
    raises(ValueError, lambda: generate_mixed_workload(prompt_bank=bank + (bank[0],), **settings))
    print("PASS: independent arrival/selection seeds, small quotas, both modes and capacity/argument rejection")


def make_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048,
    )).eval()


def check_cpu(bank):
    # Preserve all nine actual prompt lengths while mapping IDs into the small
    # model vocabulary. This is a scheduling fixture, not TinyLlama inference.
    requests = tuple(replace(r, prompt_token_ids=tuple(t % 64 for t in p.token_ids))
                     for r, p in zip(generate_workload(
                         arrival_mode="constant", rate_rps=4, num_requests=len(bank),
                         seed=42, prompt_token_ids=[10], max_new_tokens=4, eos_token_id=-1,
                     ), bank))
    workload = Workload(1, "constant", 4, 42, requests)
    workers = []
    try:
        for _ in range(2):
            workers.append(PrefillWorker(make_model(), 64, 16, PrefillHoldingPool(1)))
        target = DecodeWorker(make_model(), 64, 16)
        workers.append(target)
        coordinator = MultiPrefillCoordinator(workers[:2], target)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.json"
            save_workload(workload, path)
            clock = FakeClock()
            results = replay_workload(load_workload(path), coordinator, clock=clock, sleep=clock.sleep)
        assert len(results) == len(requests)
        by_id = {r["request_id"]: r for r in results}
        assert set(by_id) == {r.request_id for r in requests}
        for r in requests:
            expected = generate_greedy(target.runner.model, torch.tensor([r.prompt_token_ids]),
                                       r.max_new_tokens, r.eos_token_id)[0].tolist()
            actual = by_id[r.request_id]
            assert actual["status"] == "FINISHED"
            assert actual["prompt_token_ids"] + actual["generated_token_ids"] == expected
        assert not coordinator.has_pending() and not any(coordinator.waiting)
        assert not coordinator.decode_ready and not coordinator.drain_completed()
        for worker in workers:
            check_empty(worker)
    finally:
        for worker in workers:
            worker.close()
    print("PASS: all nine prompt lengths replay on small CPU workers, exact outputs and full KV cleanup")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", action="store_true",
                        help="Also retokenize and rebuild with the cached pinned tokenizer")
    args = parser.parse_args()
    torch.set_num_threads(1)
    bank, manifest = check_frozen()
    check_selection(bank, manifest)
    check_cpu(bank)
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
        for prompt in bank:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.text}], tokenize=False,
                add_generation_prompt=True,
            )
            assert tuple(tokenizer(text, add_special_tokens=False)["input_ids"]) == prompt.token_ids
        with TemporaryDirectory() as directory:
            freeze(Path(directory), local_files_only=True)
            for name in ("manifest.json", "prompt-bank.json", "mixed-v1.json"):
                assert (Path(directory) / name).read_bytes() == (ROOT / name).read_bytes()
        print("PASS: pinned retokenization and byte-identical regeneration of frozen files")


if __name__ == "__main__":
    main()
