import ray
from ray.actor import ActorClass
import torch
from transformers import LlamaConfig

from distengine.workers import PrefillWorker, DecodeWorker
from models.llama import LlamaForCausalLM


def build_test_model(seed: int) -> LlamaForCausalLM:
    """Build a deterministic CPU model inside the actor process."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )

    return LlamaForCausalLM(config).eval()


class PrefillActorImpl:
    def __init__(
        self,
        num_blocks: int = 8,
        block_size: int = 4,
        seed: int = 0,
    ):
        model = build_test_model(seed)
        self.worker = PrefillWorker(model, num_blocks, block_size)

    @ray.method
    def ready(self) -> dict:
        return self.worker.info()

    @ray.method
    def prefill(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int,
    ) -> dict:
        return self.worker.prefill(
            request_id,
            prompt_token_ids,
            max_new_tokens,
            eos_token_id,
        )

    @ray.method
    def release(self, request_id: str) -> dict:
        return self.worker.release(request_id)

    @ray.method
    def close(self) -> None:
        self.worker.close()


class DecodeActorImpl:
    def __init__(
        self,
        num_blocks: int = 8,
        block_size: int = 4,
        seed: int = 0,
    ):
        model = build_test_model(seed)
        self.worker = DecodeWorker(model, num_blocks, block_size)

    @ray.method
    def ready(self) -> dict:
        return self.worker.info()

    @ray.method
    def prepare_local_fixture(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int,
    ) -> dict:
        return self.worker.prepare_local_fixture(
            request_id,
            prompt_token_ids,
            max_new_tokens,
            eos_token_id,
        )

    @ray.method
    def step(self, request_id: str) -> dict:
        return self.worker.step(request_id)

    @ray.method
    def release(self, request_id: str) -> dict:
        return self.worker.release(request_id)

    @ray.method
    def close(self) -> None:
        self.worker.close()


# Preserve the implementation types so editors understand actor methods.
PrefillActor: ActorClass[PrefillActorImpl] = ray.remote(PrefillActorImpl).options(
    num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0,
)
DecodeActor: ActorClass[DecodeActorImpl] = ray.remote(DecodeActorImpl).options(
    num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0,
)
