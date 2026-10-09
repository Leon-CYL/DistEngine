import argparse

from transformers import AutoTokenizer

from .generation import generate_greedy
from .hf_reference import choose_device
from .load_model import MODEL_ID, REVISION, load_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text with the custom TinyLlama model."
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    device = choose_device()
    model = load_model(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
    )

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")

    output_ids = generate_greedy(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_id,
    )

    prompt_length = input_ids.shape[1]
    generated_ids = output_ids[0, prompt_length:].cpu().tolist()

    print("Generated token IDs:", generated_ids)
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()