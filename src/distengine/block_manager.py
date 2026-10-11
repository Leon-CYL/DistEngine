class BlockManager:
    """Own physical block assignments; cache views must be retired before free."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        for name, value in (("num_blocks", num_blocks), ("block_size", block_size)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.request_blocks: dict[str, list[int]] = {}

    @staticmethod
    def _validate_request_id(request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")

    def _required_blocks(self, num_tokens: int) -> int:
        if type(num_tokens) is not int or num_tokens < 0:
            raise ValueError("num_tokens must be a nonnegative integer")
        
        return (num_tokens + self.block_size - 1) // self.block_size

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, num_tokens: int) -> bool:
        """Check capacity for a new request without reserving blocks."""
        return self._required_blocks(num_tokens) <= self.num_free_blocks

    def allocate(self, request_id: str, num_tokens: int) -> list[int]:
        self._validate_request_id(request_id)
        required = self._required_blocks(num_tokens)
        
        if request_id in self.request_blocks:
            raise ValueError("Request already exists")
        if required > self.num_free_blocks:
            raise RuntimeError("Not enough free blocks for this request")
        
        blocks = [self.free_blocks.pop() for _ in range(required)]
        self.request_blocks[request_id] = blocks
        return blocks.copy()

    def get_block_table(self, request_id: str) -> list[int]:
        self._validate_request_id(request_id)
        if request_id not in self.request_blocks:
            raise KeyError("Request does not exist")
        
        return self.request_blocks[request_id].copy()

    def ensure_capacity(self, request_id: str, num_tokens: int) -> list[int]:
        """Grow to total token capacity, preserving existing blocks; never shrink."""
        self._validate_request_id(request_id)
        required = self._required_blocks(num_tokens)
        
        if request_id not in self.request_blocks:
            raise KeyError("Request does not exist")
        
        blocks = self.request_blocks[request_id]
        
        additional = max(0, required - len(blocks))
        if additional > self.num_free_blocks:
            raise RuntimeError("Not enough free blocks for this request")
        
        blocks.extend(self.free_blocks.pop() for _ in range(additional))
        return blocks.copy()

    def free(self, request_id: str) -> None:
        """Release ownership. Unknown IDs and repeated frees raise KeyError."""
        self._validate_request_id(request_id)
        if request_id not in self.request_blocks:
            raise KeyError("Request does not exist")
        
        self.free_blocks.extend(self.request_blocks.pop(request_id))
