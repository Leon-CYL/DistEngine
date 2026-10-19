import os
import sys
import time
import torch

from .telemetry import WorkerActivity, measured

from distengine.engine import LLMEngine
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
        # Reuse the existing component wiring.
        # We use the runner directly, rather than the engine scheduler.
        engine = LLMEngine(model, num_blocks, block_size)

        self.runner = engine.runner
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

    def telemetry(self, stop=False):
        if stop and self.activity is not None and self.activity.stopped is None:
            self._sync_measurement()
            self.activity.stopped = time.perf_counter()
        info = self.info()
        info["held_requests"] = len(self.prefill_pool) if hasattr(self, "prefill_pool") else 0
        info["kv_blocks_used"] = info["num_blocks"] - info["num_free_blocks"]
        info["kv_occupancy_fraction"] = info["kv_blocks_used"] / info["num_blocks"]
        info["activity"] = self.activity.report() if self.activity is not None else None
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
        pool_capacity = (
            self.runner.block_manager.num_blocks
            * self.runner.block_manager.block_size
        )

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
    def __init__(self, model: LlamaForCausalLM, num_blocks: int, block_size: int, holding_pool: PrefillHoldingPool):
        super().__init__(model, num_blocks, block_size)
        self.prefill_pool = holding_pool
    
    @measured("export_request")
    def export_request(self, request_id: str) -> dict:
        """Keep source ownership until the destination acknowledges import."""
        from .kv_transfer import export_prefill

        self._ensure_open()
        return export_prefill(self.runner, self.requests[request_id])

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
        return self._snapshot(request)

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
