import torch


class KVStorage:
    """Physical K/V tensors shared by request-local cache views."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_physical_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
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

        if any(block_id < 0 or block_id >= num_physical_blocks for block_id in block_table):
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
            storage = KVStorage(
                num_layers,
                num_kv_heads,
                head_dim,
                num_physical_blocks,
                block_size,
                self.device,
                dtype,
            )
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
        if any(
            type(block_id) is not int or not 0 <= block_id < self.num_physical_blocks
            for block_id in table
        ):
            raise ValueError("Block IDs must be integers in the physical storage range")
        if len(set(table)) != len(table):
            raise ValueError("Block IDs must be unique")
        if len(table) < len(self.block_table) or table[: len(self.block_table)] != self.block_table:
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

        if k.shape[0] != 1 or k.shape[1] != self.num_kv_heads or k.shape[3] != self.head_dim:
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

            self.k_cache[layer_index, physical_block, :, block_offset, :].copy_(
                k[0, :, token_offset, :]
            )

            self.v_cache[layer_index, physical_block, :, block_offset, :].copy_(
                v[0, :, token_offset, :]
            )

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

    @torch.no_grad()
    def load_prefix(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Import all layers by physical block, avoiding per-token CUDA copies."""
        if any(self.lengths):
            raise ValueError("Prefix import requires an empty cache")
        if keys.ndim != 5 or keys.shape != values.shape:
            raise ValueError("Expected matching [layers, 1, KV heads, tokens, head dim] tensors")
        length = keys.shape[3]
        if (
            tuple(keys.shape[:3]) != (self.num_layers, 1, self.num_kv_heads)
            or keys.shape[4] != self.head_dim
            or not 0 < length <= self.capacity
        ):
            raise ValueError("KV prefix shape or capacity mismatch")
        for tensor in (keys, values):
            if tensor.dtype != self.dtype or tensor.device != self.k_cache.device:
                raise ValueError("KV prefix device or dtype mismatch")
        for logical, start in enumerate(range(0, length, self.block_size)):
            count = min(self.block_size, length - start)
            physical = self.block_table[logical]
            self.k_cache[:, physical, :, :count, :].copy_(keys[:, 0, :, start : start + count, :])
            self.v_cache[:, physical, :, :count, :].copy_(values[:, 0, :, start : start + count, :])
        self.lengths = [length] * self.num_layers

    def reset(self) -> None:
        self.lengths = [0] * self.num_layers


class BatchedPagedKVCache:
    """One-token decode view over independent paged caches, padded on the right."""

    def __init__(self, caches: list[PagedKVCache]):
        if not caches:
            raise ValueError("A decode batch must not be empty")
        self.caches = list(caches)
        self.num_layers = caches[0].num_layers
        self.lengths = [cache.sequence_length for cache in caches]
        if any(
            cache.num_layers != self.num_layers or length + 1 > cache.capacity
            for cache, length in zip(caches, self.lengths)
        ):
            raise ValueError("Incompatible or full batch cache")
        first = caches[0]
        self.key_lengths = torch.tensor([n + 1 for n in self.lengths], device=first.device)
        if any(cache.storage is not first.storage for cache in caches):
            raise ValueError("Batched decode requires a shared KV storage pool")

        # Build mapping once per decode step, then reuse it across model layers.
        # The manager gives active requests disjoint physical block ownership.
        blocks = [block for cache in caches for block in cache.block_table]
        if len(set(blocks)) != len(blocks):
            raise ValueError("Batched requests must own disjoint physical blocks")
        device = first.k_cache.device
        self.write_blocks = torch.tensor(
            [
                cache.block_table[length // first.block_size]
                for cache, length in zip(caches, self.lengths)
            ],
            device=device,
        )
        self.write_offsets = torch.tensor(
            [length % first.block_size for length in self.lengths],
            device=device,
        )
        longest = max(self.lengths) + 1
        table_width = (longest + first.block_size - 1) // first.block_size
        tables = torch.tensor(
            [
                cache.block_table[:table_width] + [0] * max(0, table_width - len(cache.block_table))
                for cache in caches
            ],
            device=device,
        )
        positions = torch.arange(longest, device=device)
        self.read_blocks = tables[:, positions // first.block_size]
        self.read_offsets = positions % first.block_size
        self.padding = positions[None, :] >= self.key_lengths[:, None]
        self._appended_layers = set()

    @torch.no_grad()
    def append(self, layer_index, k, v):
        if not 0 <= layer_index < self.num_layers:
            raise ValueError("layer_index is out of range")
        first = self.caches[0]
        expected = (len(self.caches), first.num_kv_heads, 1, first.head_dim)
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError("Batched cache requires [batch, KV heads, 1, head dim]")
        if any(t.device != first.k_cache.device or t.dtype != first.dtype for t in (k, v)):
            raise ValueError("Batched K/V device or dtype mismatch")
        if layer_index in self._appended_layers or any(
            cache.lengths[layer_index] != length for cache, length in zip(self.caches, self.lengths)
        ):
            raise ValueError("Batch cache is a one-step view; layer already advanced")

        # Two indexed writes replace two copies per request per layer.
        first.k_cache[layer_index, self.write_blocks, :, self.write_offsets, :] = k[:, :, 0, :]
        first.v_cache[layer_index, self.write_blocks, :, self.write_offsets, :] = v[:, :, 0, :]

        # Gather straight into batched tensors; do not reconstruct, pad and
        # concatenate each request separately. Mask padding with zero even if
        # unused physical storage contains NaN or data from another request.
        result = []
        for storage in (first.k_cache, first.v_cache):
            gathered = storage[layer_index, self.read_blocks, :, self.read_offsets, :]
            gathered.masked_fill_(self.padding[:, :, None, None], 0)
            result.append(gathered.permute(0, 2, 1, 3).contiguous())
        for cache, length in zip(self.caches, self.lengths):
            cache.lengths[layer_index] = length + 1
        self._appended_layers.add(layer_index)
        return tuple(result)
