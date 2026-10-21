"""Prefill/decode replicas and their Ray actor interfaces for multi-worker serving."""

import os
import sys
import time

import ray
from ray.actor import ActorClass
import torch
from transformers import LlamaConfig

from .telemetry import WorkerActivity, measured

from .block_manager import BlockManager
from .model_runner import ModelRunner
from .paged_kv_cache import KVStorage
from distengine.request import Request
from models.llama import LlamaForCausalLM
from distengine.holding_pool import PrefillHoldingPool


class _WorkerBase:
    def __init__(
        self,
        model: LlamaForCausalLM,
        num_blocks: int,
        block_size: int,
    ):
        self.kv_transport = "nccl" if next(model.parameters()).device.type == "cuda" else "cpu"
        self.kv_export_bytes = self.kv_import_bytes = 0
        manager = BlockManager(num_blocks, block_size)
        parameter = next(model.parameters())
        if not parameter.dtype.is_floating_point:
            raise ValueError("Model parameters must use a floating-point dtype")
        if any(
            p.device != parameter.device or p.dtype != parameter.dtype for p in model.parameters()
        ):
            raise ValueError("Model parameters must share one device and dtype")

        config = model.config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by attention heads")
        num_kv_heads = config.num_key_value_heads
        if num_kv_heads is None:
            num_kv_heads = config.num_attention_heads

        storage = KVStorage(
            num_layers=config.num_hidden_layers,
            num_kv_heads=num_kv_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            num_physical_blocks=manager.num_blocks,
            block_size=manager.block_size,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.runner = ModelRunner(model, manager, storage)

        self.requests: dict[str, Request] = {}
        self.closed = False
        self.activity = None

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError("Worker is closed")

    def _snapshot(self, request: Request) -> dict:
        return {
            "request_id": request.request_id,
            "prompt_token_ids": request.prompt_token_ids.copy(),
            "generated_token_ids": request.generated_token_ids.copy(),
            "status": request.status.name,
            "reason": request.reason,
        }

    def info(self) -> dict:
        return {
            "pid": os.getpid(),
            "python_executable": sys.executable,
            "role": type(self).__name__,
            "device": str(self.runner.device),
            "dtype": str(self.runner.dtype),
            "kv_transport": self.kv_transport,
            "kv_export_bytes": self.kv_export_bytes,
            "kv_import_bytes": self.kv_import_bytes,
            "active_request_ids": list(self.requests),
            "num_free_blocks": self.runner.block_manager.num_free_blocks,
            "num_blocks": self.runner.block_manager.num_blocks,
            "block_size": self.runner.block_manager.block_size,
            "vocab_size": self.runner.model.config.vocab_size,
            "closed": self.closed,
        }

    def _sync_measurement(self):
        if self.runner.device.type == "cuda":
            torch.cuda.synchronize(self.runner.device)

    def start_measurement(self):
        self._ensure_open()
        self._sync_measurement()
        self.activity = WorkerActivity()
        self.decode_batch_sizes = {}
        self.kv_export_bytes = self.kv_import_bytes = 0

    def telemetry(self, stop=False):
        if stop and self.activity is not None and self.activity.stopped is None:
            self._sync_measurement()
            self.activity.stopped = time.perf_counter()
        info = self.info()
        info["held_requests"] = len(self.prefill_pool) if hasattr(self, "prefill_pool") else 0
        info["kv_blocks_used"] = info["num_blocks"] - info["num_free_blocks"]
        info["kv_occupancy_fraction"] = info["kv_blocks_used"] / info["num_blocks"]
        info["activity"] = self.activity.report() if self.activity is not None else None
        info["decode_batch_sizes"] = dict(getattr(self, "decode_batch_sizes", {}))
        return info

    @measured("release")
    def release(self, request_id: str) -> dict:
        self._ensure_open()

        # Unknown IDs raise KeyError.
        request = self.requests[request_id]
        result = self._snapshot(request)

        # Release KV ownership before forgetting the request.
        self.runner.release(request)
        del self.requests[request_id]

        return result

    def close(self):
        if self.closed:
            return

        for request_id in list(self.requests):
            self.release(request_id)

        self.closed = True

    def _start_local(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int,
    ) -> dict:
        self._ensure_open()

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")

        if request_id in self.requests:
            raise ValueError("Request ID is already active")

        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")

        request = Request(
            request_id,
            prompt_token_ids,
            max_new_tokens,
            eos_token_id,
        )

        # Reserve enough total pool capacity for this request to finish
        # when run alone. The final generated token remains uncached.
        maximum_cached_tokens = request.num_prompt_tokens + max_new_tokens - 1
        pool_capacity = self.runner.block_manager.num_blocks * self.runner.block_manager.block_size

        if maximum_cached_tokens > pool_capacity:
            raise ValueError("Request exceeds the worker's KV capacity")

        request.start()

        # ModelRunner validates tokens, allocates KV, runs the prompt,
        # and appends the first generated token.
        # It releases its allocation if model execution fails.
        self.runner.prefill(request)

        self.requests[request_id] = request
        result = self._snapshot(request)

        # max_new_tokens=1 or an immediate EOS can finish in prefill.
        if request.is_finished:
            self.release(request_id)

        return result


class PrefillWorker(_WorkerBase):
    def __init__(
        self,
        model: LlamaForCausalLM,
        num_blocks: int,
        block_size: int,
        holding_pool: PrefillHoldingPool,
    ):
        super().__init__(model, num_blocks, block_size)
        self.prefill_pool = holding_pool

    @measured("export_request")
    def export_request(self, request_id: str) -> dict:
        """Keep source ownership until the destination acknowledges import."""
        from .kv_transfer import export_prefill

        self._ensure_open()
        payload = export_prefill(self.runner, self.requests[request_id])
        self.kv_export_bytes += sum(
            payload[k].numel() * payload[k].element_size() for k in ("keys", "values")
        )
        return payload

    @measured("prefill")
    def prefill(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int,
    ) -> dict:
        self._ensure_open()

        if self.prefill_pool.is_full():
            raise RuntimeError("Prefill holding pool is full")

        result = self._start_local(
            request_id,
            prompt_token_ids,
            max_new_tokens,
            eos_token_id,
        )

        if result["status"] != "FINISHED":
            self.prefill_pool.add(request_id)

        return result

    @measured("release")
    def release(self, request_id: str) -> dict:
        result = super().release(request_id)

        if request_id in self.prefill_pool.request_ids():
            self.prefill_pool.remove(request_id)

        return result

    def peek_ready(self) -> dict | None:
        self._ensure_open()

        request_id = self.prefill_pool.peek()
        if request_id is None:
            return None

        request = self.requests[request_id]

        return {
            "request_id": request.request_id,
            "num_prompt_tokens": request.num_prompt_tokens,
            "max_new_tokens": request.max_new_tokens,
        }

    def can_prefill(
        self,
        num_prompt_tokens: int,
        max_new_tokens: int,
    ) -> bool:
        self._ensure_open()

        if type(num_prompt_tokens) is not int or num_prompt_tokens <= 0:
            raise ValueError("num_prompt_tokens must be a positive integer")

        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")

        manager = self.runner.block_manager
        maximum = num_prompt_tokens + max_new_tokens - 1

        if maximum > manager.num_blocks * manager.block_size:
            raise ValueError("Request exceeds worker KV capacity")

        if self.prefill_pool.is_full():
            return False

        return manager.can_allocate(num_prompt_tokens)


class DecodeWorker(_WorkerBase):
    @measured("import_request")
    def import_request(
        self,
        payload: dict,
        *,
        reserve_full: bool = False,
    ) -> dict:
        from .kv_transfer import import_prefill

        self._ensure_open()

        if (
            isinstance(payload, dict)
            and isinstance(payload.get("request_id"), str)
            and payload["request_id"] in self.requests
        ):
            raise ValueError("Request ID is already active")

        request = import_prefill(
            self.runner,
            payload,
            reserve_full=reserve_full,
        )

        self.requests[request.request_id] = request
        self.kv_import_bytes += sum(
            payload[k].numel() * payload[k].element_size() for k in ("keys", "values")
        )
        return self._snapshot(request)

    @measured("step_batch")
    def step_batch(self, request_ids: list[str]) -> list[dict]:
        self._ensure_open()
        requests = [self.requests[request_id] for request_id in request_ids]
        try:
            self.runner.decode_batch(requests)
        except Exception:
            for request in requests:
                if request.request_id not in self.runner.caches:
                    self.requests.pop(request.request_id, None)
            raise
        if self.activity is not None:
            size = len(requests)
            self.decode_batch_sizes[size] = self.decode_batch_sizes.get(size, 0) + 1
        results = [self._snapshot(request) for request in requests]
        for request in requests:
            if request.is_finished:
                self.release(request.request_id)
        return results

    @measured("step")
    def step(self, request_id: str) -> dict:
        self._ensure_open()
        request = self.requests[request_id]

        try:
            self.runner.decode(request)
        except Exception:
            # A model execution failure causes ModelRunner to retire
            # the damaged cache. Forget its worker-level state too.
            #
            # Capacity errors preserve the cache, so retain the request
            # in that case: it can be retried after capacity is released.
            if request_id not in self.runner.caches:
                del self.requests[request_id]
            raise

        if self.activity is not None:
            self.decode_batch_sizes[1] = self.decode_batch_sizes.get(1, 0) + 1
        result = self._snapshot(request)

        if request.is_finished:
            self.release(request_id)

        return result

    def prepare_local_fixture(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int,
    ) -> dict:
        return self._start_local(
            request_id,
            prompt_token_ids,
            max_new_tokens,
            eos_token_id,
        )

    def can_admit(
        self,
        num_prompt_tokens: int,
        max_new_tokens: int,
    ) -> bool:
        self._ensure_open()

        if type(num_prompt_tokens) is not int or num_prompt_tokens <= 0:
            raise ValueError("num_prompt_tokens must be a positive integer")

        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")

        maximum = num_prompt_tokens + max_new_tokens - 1
        manager = self.runner.block_manager

        if maximum > manager.num_blocks * manager.block_size:
            raise ValueError("Request exceeds destination pool capacity")

        return manager.can_allocate(maximum)


def build_cpu_model(seed: int) -> LlamaForCausalLM:
    """Build the deterministic small model used by CPU benchmark runs."""
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

    return LlamaForCausalLM(config).float().eval()


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
        from utils.load_model import load_model

        return load_model(target)
    return build_cpu_model(seed).to(device=target, dtype=torch.float32)


def actor_info(worker) -> dict:
    info = worker.info()
    info.update(
        {
            "actor_id": ray.get_runtime_context().get_actor_id(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "ray_gpu_ids": ray.get_runtime_context().get_accelerator_ids().get("GPU", []),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "kv_device": str(worker.runner.storage.k_cache.device),
        }
    )
    device = worker.runner.device
    if any(p.device != device for p in worker.runner.model.parameters()):
        raise RuntimeError("Model parameters have inconsistent placement")
    if (
        worker.runner.storage.k_cache.device != device
        or worker.runner.storage.v_cache.device != device
    ):
        raise RuntimeError("KV storage must be on the model device")
    if device.type == "cuda":
        info["visible_gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return info


def scheduled_state(worker):
    """State returned with an operation, avoiding RPCs behind a busy actor."""
    state = worker.telemetry()
    if isinstance(worker, PrefillWorker):
        state["max_held_requests"] = worker.prefill_pool._max_held_requests
        state["held_descriptors"] = [
            {
                "request_id": rid,
                "num_prompt_tokens": worker.requests[rid].num_prompt_tokens,
                "max_new_tokens": worker.requests[rid].max_new_tokens,
            }
            for rid in worker.prefill_pool.request_ids()
        ]
    return state


def run_scheduled(worker, operation, *args, **kwargs):
    allowed = {"prefill", "step", "step_batch", "export_request", "import_request", "release"}
    if operation not in allowed:
        raise ValueError("Unknown scheduled worker operation")
    try:
        result = getattr(worker, operation)(*args, **kwargs)
    except Exception as exc:
        return {"result": None, "error": exc, "state": scheduled_state(worker)}
    return {"result": result, "error": None, "state": scheduled_state(worker)}


def import_scheduled(worker, exported):
    if exported["error"] is not None:
        return {"result": None, "error": exported["error"], "state": scheduled_state(worker)}
    return run_scheduled(worker, "import_request", exported["result"], reserve_full=True)


class ScheduledActorMethods:
    @ray.method
    def transfer_probe(self):
        device = self.worker.runner.device
        return torch.arange(32, device=device, dtype=self.worker.runner.dtype)

    @ray.method
    def validate_transfer_probe(self, tensor):
        device = self.worker.runner.device
        if tensor.device != device or tensor.dtype != self.worker.runner.dtype:
            raise RuntimeError("Tensor transport changed the expected KV device/dtype")
        expected = torch.arange(32, device=device, dtype=tensor.dtype)
        if not torch.equal(tensor, expected):
            raise RuntimeError("Tensor transport probe corrupted data")
        return {
            "received_device": str(tensor.device),
            "dtype": str(tensor.dtype),
            "bytes": tensor.numel() * tensor.element_size(),
        }

    @ray.method
    def scheduled_state(self):
        return scheduled_state(self.worker)

    @ray.method
    def scheduled(self, operation, *args, **kwargs):
        return run_scheduled(self.worker, operation, *args, **kwargs)

    @ray.method
    def scheduled_import(self, exported):
        # Ray resolves the export reference in this actor, never in the driver.
        return import_scheduled(self.worker, exported)


class PrefillActorImpl(ScheduledActorMethods):
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
        self.worker = PrefillWorker(
            model,
            num_blocks,
            block_size,
            holding_pool=PrefillHoldingPool(max_held_requests),
        )

    @ray.method
    def start_measurement(self):
        self.worker.start_measurement()

    @ray.method
    def telemetry(self, stop=False):
        return self.worker.telemetry(stop=stop)

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


class DecodeActorImpl(ScheduledActorMethods):
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
    def start_measurement(self):
        self.worker.start_measurement()

    @ray.method
    def telemetry(self, stop=False):
        return self.worker.telemetry(stop=stop)

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
    def step_batch(self, request_ids):
        return self.worker.step_batch(request_ids)

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
    num_cpus=1,
    num_gpus=0,
    max_restarts=0,
    max_task_retries=0,
    enable_tensor_transport=True,
)(PrefillActorImpl)

DecodeActor: ActorClass[DecodeActorImpl] = ray.remote(
    num_cpus=1,
    num_gpus=0,
    max_restarts=0,
    max_task_retries=0,
    enable_tensor_transport=True,
)(DecodeActorImpl)
