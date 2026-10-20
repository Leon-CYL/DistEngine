import argparse
import re
from typing import Any, cast

import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import platform
from pathlib import Path


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def resolve_revision(revision: str) -> str:
    """Resolve a branch/tag once; an explicit commit needs no metadata lookup."""
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    commit = HfApi().model_info(MODEL_ID, revision=revision).sha
    if commit is None:
        raise RuntimeError("Hugging Face Hub did not return a model commit.")
    return commit


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("references/tinyllama_hf.json"))
    parser.add_argument(
        "--revision",
        default="main",
        help="Model branch, tag, or commit hash; use a saved commit to reproduce a run.",
    )
    args = parser.parse_args()
    
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    device = choose_device()
    resolved_revision = resolve_revision(args.revision)
    print(f"Model revision: {resolved_revision}")
    print(f"Loading {MODEL_ID} on {device} with float32")

    # Transformers' auto-model annotations misdescribe .to() and .generate()
    # in this version. Keep the typing workaround at the dynamic factory boundary.
    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            attn_implementation="eager",
            revision=resolved_revision,
        ),
    )
    model.to(device)
    model.eval()

    # Use the same immutable commit for both model and tokenizer files.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=resolved_revision,
    )

    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # The chat template already includes the required special tokens.
    inputs = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # generate() returns the prompt followed by newly generated tokens.
    generated_ids = output_ids[0, prompt_length:].cpu().tolist()
    print("Formatted prompt:", repr(formatted_prompt))
    print("Input shape:", tuple(inputs["input_ids"].shape))
    print("Input token IDs:", inputs["input_ids"][0].cpu().tolist())
    print("Generated token IDs:", generated_ids)
    print("Response:", tokenizer.decode(generated_ids, skip_special_tokens=True))

    # Save prompt, token IDs, settings, and environment metadata as JSON.
    reference = {
        "model_id": MODEL_ID,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "user_prompt": args.prompt,
        "formatted_prompt": formatted_prompt,
        "prompt_length": prompt_length,
        "input_token_ids": inputs["input_ids"][0].cpu().tolist(),
        "generated_token_ids": generated_ids,
        "response": tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ),
        "device": str(device),
        "dtype": str(model.dtype),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "generation_settings": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "effective_generation_config": model.generation_config.to_dict(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(f"Saved reference to {args.output}")
    


if __name__ == "__main__":
    main()
