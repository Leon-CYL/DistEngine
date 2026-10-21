import torch

from layers.sampler import GreedySampler
from .paged_kv_cache import KVStorage, PagedKVCache, BatchedPagedKVCache
from models.llama import LlamaForCausalLM
from .block_manager import BlockManager
from .request import Request, RequestStatus


class ModelRunner:
    def __init__(
        self,
        model: LlamaForCausalLM,
        block_manager: BlockManager,
        storage: KVStorage,
    ) -> None:
        parameter = next(model.parameters())
        device = parameter.device
        dtype = parameter.dtype
        if not dtype.is_floating_point:
            raise ValueError("Model parameters must use a floating-point dtype")
        if any(p.device != device or p.dtype != dtype for p in model.parameters()):
            raise ValueError("Model parameters must share one device and dtype")

        config = model.config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by attention heads")
        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        if num_kv_heads is None:
            num_kv_heads = config.num_attention_heads

        expected_shape = (
            config.num_hidden_layers,
            block_manager.num_blocks,
            num_kv_heads,
            block_manager.block_size,
            head_dim,
        )
        for name, tensor in (("K", storage.k_cache), ("V", storage.v_cache)):
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} storage shape must be {expected_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.device != device:
                raise ValueError(f"{name} storage device must be {device}, got {tensor.device}")
            if tensor.dtype != dtype:
                raise ValueError(f"{name} storage dtype must be {dtype}, got {tensor.dtype}")

        self.model = model
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.block_manager = block_manager
        self.storage = storage
        self.sampler = GreedySampler()
        self.caches: dict[str, PagedKVCache] = {}

    def _validate_request(self, request: Request) -> None:
        if not isinstance(request, Request):
            raise ValueError("Expected a Request")

        if not isinstance(request.request_id, str) or not request.request_id:
            raise ValueError("request_id must be a nonempty string")

        if not request.prompt_token_ids:
            raise ValueError("The prompt must contain at least one token")

        if type(request.max_new_tokens) is not int or request.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")

        for token_id in request.token_ids:
            if type(token_id) is not int:
                raise ValueError("Token IDs must be integers")

            if not 0 <= token_id < self.model.config.vocab_size:
                raise ValueError("Token ID is outside the model vocabulary")

    def prepare_inputs(
        self,
        token_ids: list[int],
        past_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not token_ids:
            raise ValueError("Must provide at least one token")

        if type(past_length) is not int or past_length < 0:
            raise ValueError("past_length must be a nonnegative integer")

        for token_id in token_ids:
            if type(token_id) is not int:
                raise ValueError("Token IDs must be integers")

            if not 0 <= token_id < self.model.config.vocab_size:
                raise ValueError("Token ID is outside the model vocabulary")

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        positions = torch.arange(
            past_length,
            past_length + len(token_ids),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        return input_ids, positions

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        self._validate_request(request)

        if request.status != RequestStatus.RUNNING:
            raise ValueError("Prefill requires a RUNNING request")

        if request.num_generated_tokens != 0:
            raise ValueError("Prefill requires no generated tokens")

        if request.request_id in self.caches:
            raise ValueError("Request already has a runner cache")

        input_ids, positions = self.prepare_inputs(
            request.prompt_token_ids,
            past_length=0,
        )

        # Keep allocation outside try: a duplicate allocation failure must
        # never cause us to free another request's existing blocks.
        table = self.block_manager.allocate(
            request.request_id,
            request.num_prompt_tokens,
        )

        cache = None

        try:
            config = self.model.config
            num_kv_heads = config.num_key_value_heads
            if num_kv_heads is None:
                num_kv_heads = config.num_attention_heads

            cache = PagedKVCache(
                num_layers=config.num_hidden_layers,
                num_kv_heads=num_kv_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                num_physical_blocks=self.block_manager.num_blocks,
                block_size=self.block_manager.block_size,
                block_table=table,
                device=self.device,
                dtype=self.dtype,
                storage=self.storage,
            )

            logits = self.model(
                input_ids,
                positions=positions,
                cache=cache,
            )

            token = int(self.sampler(logits[:, -1, :]).item())
            request.append_token(token)
            self.caches[request.request_id] = cache

            return token

        except Exception:
            # Retire the cache view before returning its blocks.
            cache = None
            self.block_manager.free(request.request_id)
            raise

    @torch.inference_mode()
    def decode(self, request: Request) -> int:
        self._validate_request(request)

        if request.status != RequestStatus.RUNNING:
            raise ValueError("Decode requires a RUNNING request")

        if request.request_id not in self.caches:
            raise KeyError("Request has no runner cache")

        if request.num_generated_tokens == 0:
            raise ValueError("Decode requires a generated token")

        cache = self.caches[request.request_id]

        if cache.sequence_length != len(request.token_ids) - 1:
            raise ValueError("Request tokens and cache length are inconsistent")

        input_ids, positions = self.prepare_inputs(
            [request.generated_token_ids[-1]],
            past_length=cache.sequence_length,
        )

        # Capacity failure preserves the cache and request for a later retry.
        table = self.block_manager.ensure_capacity(
            request.request_id,
            cache.sequence_length + 1,
        )

        try:
            cache.update_block_table(table)

            logits = self.model(
                input_ids,
                positions=positions,
                cache=cache,
            )

            token = int(self.sampler(logits[:, -1, :]).item())
            request.append_token(token)

            return token

        except Exception:
            # A failed forward may have written only some cache layers.
            del self.caches[request.request_id]
            cache = None
            self.block_manager.free(request.request_id)
            raise

    @torch.inference_mode()
    def decode_batch(self, requests: list[Request]) -> list[int]:
        requests = list(requests)
        if not requests:
            raise ValueError("A decode batch must not be empty")
        for request in requests:
            self._validate_request(request)
        ids = [r.request_id for r in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate request in decode batch")
        caches = []
        additional = 0
        manager = self.block_manager
        for request in requests:
            cache = self.caches[request.request_id]
            if request.status != RequestStatus.RUNNING or not request.num_generated_tokens:
                raise ValueError("Decode requires an active generated token")
            if cache.sequence_length != len(request.token_ids) - 1:
                raise ValueError("Request tokens and cache length are inconsistent")
            additional += max(
                0,
                manager._required_blocks(cache.sequence_length + 1)
                - len(manager.get_block_table(request.request_id)),
            )
            caches.append(cache)
        # Preflight the whole batch before changing block ownership or KV contents.
        if additional > manager.num_free_blocks:
            raise RuntimeError("Not enough free blocks for decode batch")
        for request, cache in zip(requests, caches):
            cache.update_block_table(
                manager.ensure_capacity(request.request_id, cache.sequence_length + 1)
            )
        batch = BatchedPagedKVCache(caches)
        inputs = torch.tensor([[r.generated_token_ids[-1]] for r in requests], device=self.device)
        try:
            logits = self.model(inputs, cache=batch)
            tokens = self.sampler(logits[:, -1, :]).tolist()
            for request, token in zip(requests, tokens):
                request.append_token(int(token))
            return tokens
        except Exception:
            # Any layer may have written KV for every selected row.
            for request in requests:
                self.release(request)
            raise

    def release(self, request: Request) -> None:
        if not isinstance(request, Request):
            raise ValueError("Expected a Request")

        request_id = request.request_id

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")

        if request_id not in self.caches:
            raise KeyError("Request has no runner cache")

        # Retire the runner's cache view before releasing physical blocks.
        del self.caches[request_id]
        self.block_manager.free(request_id)
