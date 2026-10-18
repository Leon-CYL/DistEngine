import os
import ray
from ray.actor import ActorClass
import torch
from transformers import LlamaConfig

from distengine.workers import PrefillWorker, DecodeWorker
from distengine.holding_pool import PrefillHoldingPool
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


def build_worker_model(seed: int, device: str, checkpoint: bool):
    """Construct a model in its owning actor, after Ray assigns devices."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda":
        ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
        if len(ids) != 1:
            raise RuntimeError("CUDA workers require exactly one Ray GPU reservation")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA workers must see exactly one NVIDIA GPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    target = torch.device("cuda:0" if device == "cuda" else "cpu")
    if checkpoint:
        from load_model import load_model

        return load_model(target)
    return build_test_model(seed).to(target)


def actor_info(worker) -> dict:
    info = worker.info()
    info.update({
        "actor_id": ray.get_runtime_context().get_actor_id(),
        "node_id": ray.get_runtime_context().get_node_id(),
        "ray_gpu_ids": ray.get_runtime_context().get_accelerator_ids().get("GPU", []),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kv_device": str(worker.runner.storage.k_cache.device),
    })
    device = worker.runner.device
    if any(p.device != device for p in worker.runner.model.parameters()):
        raise RuntimeError("Model parameters have inconsistent placement")
    if (worker.runner.storage.k_cache.device != device
            or worker.runner.storage.v_cache.device != device):
        raise RuntimeError("KV storage must be on the model device")
    if device.type == "cuda":
        info["visible_gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


class PrefillActorImpl:
    def __init__(
        self,
        num_blocks: int = 8,
        block_size: int = 4,
        seed: int = 0,
        device: str = "cpu",
        checkpoint: bool = False,
        max_held_requests: int = 4,
    ):
        model = build_worker_model(seed, device, checkpoint)
        self.worker = PrefillWorker(model, num_blocks, block_size, holding_pool=PrefillHoldingPool(max_held_requests))

    @ray.method
    def ready(self) -> dict:
        return actor_info(self.worker)

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
    def can_prefill(
        self,
        num_prompt_tokens: int,
        max_new_tokens: int,
    ) -> bool:
        return self.worker.can_prefill(num_prompt_tokens, max_new_tokens)

    @ray.method
    def export_request(self, request_id: str) -> dict:
        return self.worker.export_request(request_id)
    
    @ray.method
    def peek_ready(self) -> dict | None:
        return self.worker.peek_ready()

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
        device: str = "cpu",
        checkpoint: bool = False,
    ):
        model = build_worker_model(seed, device, checkpoint)
        self.worker = DecodeWorker(model, num_blocks, block_size)

    @ray.method
    def ready(self) -> dict:
        return actor_info(self.worker)

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
    def import_request(
        self,
        payload: dict,
        *,
        reserve_full: bool = False,
    ) -> dict:
        return self.worker.import_request(
            payload,
            reserve_full=reserve_full,
        )
        
    @ray.method
    def can_admit(
        self,
        num_prompt_tokens: int,
        max_new_tokens: int,
    ) -> bool:
        return self.worker.can_admit(
            num_prompt_tokens,
            max_new_tokens,
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
PrefillActor: ActorClass[PrefillActorImpl] = ray.remote(
    num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0,
)(PrefillActorImpl)

DecodeActor: ActorClass[DecodeActorImpl] = ray.remote(
    num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0,
)(DecodeActorImpl)
