"""Build the stage 19 TinyLlama prompt bank and freeze benchmark inputs."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from distengine.mixed_workload import Prompt, generate_mixed_workload
from distengine.workload import save_workload

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
BOUNDS = {"short": (32, 64), "medium": (128, 256), "long": (512, 768)}
WEIGHTS = {"short": .6, "medium": .3, "long": .1}
TOPICS = (
    ("garden", "A community garden collects rainwater in tanks. Volunteers water the beds in the morning, record soil moisture, and share ripe vegetables with neighbors.", "Explain how the volunteers could conserve water."),
    ("library", "A town library lends books and hosts evening classes. Staff track popular titles, repair damaged books, and help visitors find reliable information.", "Suggest ways the library could improve its service."),
    ("cycling", "A cycling club plans trips along quiet roads. Riders check their brakes, carry drinking water, and choose rest stops with shelter from bad weather.", "Explain how the club could make its trips safer."),
)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_prompt_bank(tokenizer):
    bank = []
    for length_class, (minimum, maximum) in BOUNDS.items():
        for topic, paragraph, question in TOPICS:
            for repeats in range(1, 100):
                text = " ".join([paragraph] * repeats + [question])
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=False,
                    add_generation_prompt=True,
                )
                ids = tuple(tokenizer(formatted, add_special_tokens=False)["input_ids"])
                if len(ids) >= minimum:
                    if len(ids) > maximum:
                        raise ValueError(f"Cannot fit {topic} into {length_class} bounds")
                    bank.append(Prompt(f"{length_class}-{topic}", length_class, ids, text))
                    break
            else:
                raise ValueError("Prompt construction exceeded repetition limit")
    return tuple(bank)


def freeze(output: Path, *, local_files_only=False, num_requests=100,
           rate_rps=2.0, max_new_tokens=32, kv_capacity_tokens=1024, num_prefill_workers=2):
    if type(num_prefill_workers) is not int or num_prefill_workers <= 0:
        raise ValueError("num_prefill_workers must be a positive integer")
    if type(num_requests) is not int or num_requests <= 0:
        raise ValueError("num_requests must be a positive integer")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; preserve frozen inputs")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, local_files_only=local_files_only,
    )
    bank = build_prompt_bank(tokenizer)
    settings = dict(
        arrival_mode="poisson", rate_rps=rate_rps, num_requests=num_requests,
        arrival_seed=42, prompt_seed=19, class_weights=WEIGHTS,
        max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id,
        vocab_size=32000, max_context_tokens=2048, kv_capacity_tokens=kv_capacity_tokens,
    )
    workload = generate_mixed_workload(prompt_bank=bank, **settings)
    output.mkdir(parents=True, exist_ok=True)
    bank_path = output / "prompt-bank.json"
    workload_path = output / "mixed-v1.json"
    write_json(bank_path, {"schema_version": 1, "model_id": MODEL_ID,
                          "revision": REVISION, "length_bounds": BOUNDS,
                          "prompts": [asdict(p) for p in bank]})
    save_workload(workload, workload_path)
    by_tokens = {p.token_ids: p for p in bank}
    assignments = [{"request_id": r.request_id,
                    "prompt_id": by_tokens[r.prompt_token_ids].prompt_id,
                    "length_class": by_tokens[r.prompt_token_ids].length_class}
                   for r in workload.requests]
    source_totals = [sum(len(r.prompt_token_ids) for r in workload.requests[i::num_prefill_workers]) for i in range(num_prefill_workers)]
    write_json(output / "manifest.json", {
        "schema_version": 1, "model_id": MODEL_ID, "tokenizer_id": MODEL_ID,
        "revision": REVISION, "chat_template": tokenizer.chat_template,
        "settings": settings, "selection": "largest-remainder quotas; seeded shuffle; uniform prompt choice",
        "prompt_bank": {"file": bank_path.name, "sha256": digest(bank_path)},
        "workload": {"file": workload_path.name, "sha256": digest(workload_path)},
        "class_counts": dict(Counter(a["length_class"] for a in assignments)),
        "request_prompts": assignments,
        "round_robin": {"num_prefill_workers": num_prefill_workers, "prompt_tokens_per_source": source_totals,
                        "max_to_min_ratio": max(source_totals) / min(source_totals) if min(source_totals) else None},
        "capacity_note": f"Each worker needs at least {kv_capacity_tokens} KV token slots (e.g. {(kv_capacity_tokens + 15) // 16} blocks of 16); this is not a throughput recommendation.",
    })
    print(f"Frozen {len(workload.requests)} requests in {output}")
    print(f"Prompt lengths: {[(p.prompt_id, len(p.token_ids)) for p in bank]}")
    print(f"Round-robin prompt tokens: {source_totals}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("workloads/stage19"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-prefill-workers", type=int, default=2)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--rate-rps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--kv-capacity-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; preserve frozen inputs or choose a new path")
    freeze(args.output, local_files_only=args.local_files_only,
           num_requests=args.num_requests, rate_rps=args.rate_rps,
           max_new_tokens=args.max_new_tokens, kv_capacity_tokens=args.kv_capacity_tokens,
           num_prefill_workers=args.num_prefill_workers)


if __name__ == "__main__":
    main()
