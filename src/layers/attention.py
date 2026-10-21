import torch
from torch import nn

from .linear import Linear
from .rotary_embedding import RotaryEmbedding
from distengine.paged_kv_cache import PagedKVCache, BatchedPagedKVCache


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()

        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q: [B, num_heads,    T, head_dim]
        # k: [B, num_kv_heads, T, head_dim]
        # v: [B, num_kv_heads, T, head_dim]

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        # scores: [B, num_heads, T, T]

        query_length = q.shape[-2]
        key_length = k.shape[-2]

        if key_length < query_length:
            raise ValueError("Key length must be at least query length")

        past_length = key_length - query_length

        query_positions = past_length + torch.arange(
            query_length,
            device=q.device,
        )
        key_positions = torch.arange(
            key_length,
            device=q.device,
        )

        future_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)

        scores = scores.masked_fill(future_mask, float("-inf"))

        if key_lengths is not None:
            padding = key_positions[None, :] >= key_lengths[:, None]
            scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))

        probabilities = torch.softmax(
            scores.float(),
            dim=-1,
        ).to(q.dtype)

        return probabilities @ v
        # output: [B, num_heads, T, head_dim]


class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        q_size = num_heads * self.head_dim
        kv_size = num_kv_heads * self.head_dim

        self.q_proj = Linear(hidden_size, q_size)
        self.k_proj = Linear(hidden_size, kv_size)
        self.v_proj = Linear(hidden_size, kv_size)
        self.o_proj = Linear(q_size, hidden_size)

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=rope_theta,
        )

        self.attn = Attention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=self.head_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: PagedKVCache | BatchedPagedKVCache | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        q = (
            self.q_proj(x)
            .view(batch_size, sequence_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        k = (
            self.k_proj(x)
            .view(batch_size, sequence_length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        v = (
            self.v_proj(x)
            .view(batch_size, sequence_length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = self.rotary_emb(q, k, positions)

        if cache is not None:
            k, v = cache.append(layer_index, k, v)

        context = self.attn(q, k, v, key_lengths=getattr(cache, "key_lengths", None))

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.num_heads * self.head_dim,
            )
        )

        return self.o_proj(context)
