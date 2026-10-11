import torch


class KVStorage:
    """Physical K/V tensors shared by request-local cache views."""

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 num_physical_blocks: int, block_size: int,
                 device: torch.device, dtype: torch.dtype) -> None:
        dimensions = (num_layers, num_physical_blocks, num_kv_heads, block_size, head_dim)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Storage dimensions must be positive integers")
        if not dtype.is_floating_point:
            raise ValueError("KV cache dtype must be floating point")
        self.k_cache = torch.empty(dimensions, device=device, dtype=dtype)
        self.v_cache = torch.empty_like(self.k_cache)


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_physical_blocks: int,
        block_size: int,
        block_table: list[int],
        device: torch.device,
        dtype: torch.dtype,
        storage: KVStorage | None = None,
    ) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
            ("num_physical_blocks", num_physical_blocks),
            ("block_size", block_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
            
        if not dtype.is_floating_point:
            raise ValueError("KV cache dtype must be floating point")

        if any(type(block_id) is not int for block_id in block_table):
            raise ValueError("Block IDs must be integers")

        if any(
            block_id < 0 or block_id >= num_physical_blocks
            for block_id in block_table
        ):
            raise ValueError("Block ID is out of range")

        if len(set(block_table)) != len(block_table):
            raise ValueError("Block IDs must be unique")
                
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_physical_blocks = num_physical_blocks
        self.block_size = block_size
        self.block_table = block_table.copy()
        self.device = torch.device(device)
        self.dtype = dtype

        self.capacity = len(self.block_table) * self.block_size
        self.lengths = [0] * self.num_layers
        
        shape = (
            self.num_layers,
            self.num_physical_blocks,
            self.num_kv_heads,
            self.block_size,
            self.head_dim,
        )

        if storage is None:
            storage = KVStorage(num_layers, num_kv_heads, head_dim,
                                num_physical_blocks, block_size, self.device, dtype)
        resolved_device = torch.empty(0, device=self.device).device
        for tensor in (storage.k_cache, storage.v_cache):
            if tuple(tensor.shape) != shape:
                raise ValueError("Shared storage shape does not match cache dimensions")
            if tensor.dtype != dtype or tensor.device != resolved_device:
                raise ValueError("Shared storage device or dtype does not match cache")
        self.storage = storage
        self.k_cache = storage.k_cache
        self.v_cache = storage.v_cache
        

    def update_block_table(self, block_table: list[int]) -> None:
        """Install a manager-provided expansion without moving existing tokens."""
        table = block_table.copy()
        if any(type(block_id) is not int or not 0 <= block_id < self.num_physical_blocks
               for block_id in table):
            raise ValueError("Block IDs must be integers in the physical storage range")
        if len(set(table)) != len(table):
            raise ValueError("Block IDs must be unique")
        if len(table) < len(self.block_table) or table[:len(self.block_table)] != self.block_table:
            raise ValueError("An updated block table must preserve the existing prefix")
        self.block_table = table
        self.capacity = len(table) * self.block_size

    @property
    def sequence_length(self) -> int:
        length = self.lengths[0]

        if any(layer_length != length for layer_length in self.lengths):
            raise RuntimeError("KV cache layer lengths are inconsistent")

        return length


    @torch.no_grad()
    def append(
        self,
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if not 0 <= layer_index < self.num_layers:
            raise ValueError("layer_index is out of range")

        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("K and V must be four-dimensional")

        if k.shape != v.shape:
            raise ValueError("K and V must have identical shapes")

        if (k.shape[0] != 1 or k.shape[1] != self.num_kv_heads or k.shape[3] != self.head_dim):
            raise ValueError("Expected shape [1, num_kv_heads, new_tokens, head_dim]")

        new_tokens = k.shape[2]

        if new_tokens == 0:
            raise ValueError("Must append at least one token")

        if k.device != self.k_cache.device or v.device != self.v_cache.device:
            raise ValueError("K and V must be on the cache device")

        if k.dtype != self.dtype or v.dtype != self.dtype:
            raise ValueError("K and V must match the cache dtype")

        start = self.lengths[layer_index]
        end = start + new_tokens

        if end > self.capacity:
            raise ValueError("KV cache capacity exceeded")
        
        for token_offset in range(new_tokens):
            position = start + token_offset

            logical_block = position // self.block_size
            block_offset = position % self.block_size
            physical_block = self.block_table[logical_block]

            self.k_cache[layer_index, physical_block, :, block_offset, :].copy_(k[0, :, token_offset, :])

            self.v_cache[layer_index, physical_block, :, block_offset, :].copy_(v[0, :, token_offset, :])

        self.lengths[layer_index] = end

        return self.read(layer_index)


    def read(
        self,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_index < self.num_layers:
            raise ValueError("layer_index is out of range")

        length = self.lengths[layer_index]

        if length == 0:
            shape = (1, self.num_kv_heads, 0, self.head_dim)
            return (
                self.k_cache.new_empty(shape),
                self.v_cache.new_empty(shape),
            )

        key_parts = []
        value_parts = []
        remaining = length

        for physical_block in self.block_table:
            if remaining == 0:
                break

            count = min(self.block_size, remaining)

            key_parts.append(self.k_cache[layer_index, physical_block, :, :count, :])
            value_parts.append(self.v_cache[layer_index, physical_block, :, :count, :])
            remaining -= count

        return (
            torch.cat(key_parts, dim=1).unsqueeze(0),
            torch.cat(value_parts, dim=1).unsqueeze(0),
        )
        

    def reset(self) -> None:
        self.lengths = [0] * self.num_layers
        
        
def main():
    table = [3, 0, 5]

    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        num_physical_blocks=6,
        block_size=4,
        block_table=table,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert cache.k_cache.shape == (2, 6, 2, 4, 4)
    assert cache.v_cache.shape == (2, 6, 2, 4, 4)
    assert cache.capacity == 12
    assert cache.lengths == [0, 0]
    assert cache.k_cache.dtype == torch.float32
    assert cache.k_cache.device == torch.device("cpu")

    table[0] = 1
    assert cache.block_table == [3, 0, 5]

    print("PASS: paged KV cache constructor")
    
    # An empty read must expose zero tokens.
    keys, values = cache.read(0)
    assert keys.shape == (1, 2, 0, 4)
    assert values.shape == (1, 2, 0, 4)

    # First write: three tokens into physical block 3.
    k1 = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    v1 = k1 + 100

    keys, values = cache.append(0, k1, v1)

    assert cache.lengths == [3, 0]
    assert torch.equal(keys, k1)
    assert torch.equal(values, v1)

    # Append six tokens, crossing two block boundaries.
    k2 = torch.arange(48, dtype=torch.float32).reshape(1, 2, 6, 4) + 200
    v2 = k2 + 100

    keys, values = cache.append(0, k2, v2)

    assert cache.lengths == [9, 0]
    assert keys.shape == (1, 2, 9, 4)
    assert torch.equal(keys, torch.cat((k1, k2), dim=2))
    assert torch.equal(values, torch.cat((v1, v2), dim=2))

    # Check physical placement of positions 3, 4, and 8.
    assert torch.equal(cache.k_cache[0, 3, :, 3, :], k2[0, :, 0, :])
    assert torch.equal(cache.k_cache[0, 0, :, 0, :], k2[0, :, 1, :])
    assert torch.equal(cache.k_cache[0, 5, :, 0, :], k2[0, :, 5, :])

    # Another layer has independent contents and length.
    before = keys.clone()
    cache.append(1, k1, v1)

    assert cache.lengths == [9, 3]
    assert torch.equal(cache.read(0)[0], before)

    # Overflow must preserve existing data and lengths.
    too_large = torch.zeros((1, 2, 4, 4))
    before_k, before_v = cache.read(0)

    try:
        cache.append(0, too_large, too_large)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected capacity overflow")

    assert cache.lengths == [9, 3]
    after_k, after_v = cache.read(0)
    assert torch.equal(after_k, before_k)
    assert torch.equal(after_v, before_v)

    print("PASS: paged writes, logical reads, boundaries, and overflow")
    
    try:
        _ = cache.sequence_length
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected inconsistent layer lengths")

    k_pointer = cache.k_cache.data_ptr()
    v_pointer = cache.v_cache.data_ptr()
    table_before = cache.block_table.copy()

    cache.reset()

    assert cache.sequence_length == 0
    assert cache.lengths == [0, 0]
    assert cache.block_table == table_before
    assert cache.k_cache.data_ptr() == k_pointer
    assert cache.v_cache.data_ptr() == v_pointer

    keys, values = cache.read(0)
    assert keys.shape == (1, 2, 0, 4)
    assert values.shape == (1, 2, 0, 4)
    
    expected_keys = [torch.empty(1, 2, 0, 4) for _ in range(cache.num_layers)]
    expected_values = [torch.empty(1, 2, 0, 4) for _ in range(cache.num_layers)]
    torch.manual_seed(0)

    # Cumulative lengths: 3, 4, 5, 8, 9.
    for chunk_size in (3, 1, 1, 3, 1):
        for layer_index in range(cache.num_layers):
            k = torch.randn(1, 2, chunk_size, 4)
            v = torch.randn(1, 2, chunk_size, 4)

            paged_k, paged_v = cache.append(layer_index, k, v)
            expected_keys[layer_index] = torch.cat((expected_keys[layer_index], k), dim=2)
            expected_values[layer_index] = torch.cat((expected_values[layer_index], v), dim=2)

            assert torch.equal(paged_k, expected_keys[layer_index])
            assert torch.equal(paged_v, expected_values[layer_index])

        assert cache.sequence_length == expected_keys[0].shape[2]

    assert cache.sequence_length == 9
    print("PASS: paged storage matches expected token history")
    
    cache.reset()

    for layer_index in range(cache.num_layers):
        k = torch.full((1, 2, 1, 4), 7.0)
        v = torch.full((1, 2, 1, 4), 8.0)

        paged_k, paged_v = cache.append(layer_index, k, v)

        assert paged_k.shape == (1, 2, 1, 4)
        assert torch.equal(paged_k, k)
        assert torch.equal(paged_v, v)

    assert cache.sequence_length == 1
    print("PASS: paged reset and reuse")


if __name__ == "__main__":
    main()