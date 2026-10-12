from .request import Request
from models.llama import LlamaForCausalLM
from .block_manager import BlockManager
from .paged_kv_cache import KVStorage
from .model_runner import ModelRunner
from .scheduler import Scheduler

class LLMEngine:
    def __init__(
        self,
        model: LlamaForCausalLM,
        num_blocks: int,
        block_size: int,
    ) -> None:
        manager = BlockManager(num_blocks, block_size)
        parameter = next(model.parameters())
        if not parameter.dtype.is_floating_point:
            raise ValueError("Model parameters must use a floating-point dtype")
        if any(p.device != parameter.device or p.dtype != parameter.dtype
               for p in model.parameters()):
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
        runner = ModelRunner(model, manager, storage)
        scheduler = Scheduler(runner)

        self.block_manager = manager
        self.storage = storage
        self.runner = runner
        self.scheduler = scheduler
        

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int,
    ) -> Request: 
        
        request = Request(request_id, prompt_token_ids, max_new_tokens, eos_token_id)
        self.scheduler.add_request(request)
        return request
        

    def step(self) -> Request | None: 
        return self.scheduler.step()
        

    @property
    def has_pending_requests(self) -> bool: 
        return self.scheduler.has_pending_requests
        

    def run_until_complete(self) -> list[Request]: 
        start = len(self.scheduler.finished)

        while self.has_pending_requests:
            self.step()

        return self.scheduler.finished[start:].copy()
        
        