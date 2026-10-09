import torch
from torch import nn

from .linear import Linear
from .rotary_embedding import RotaryEmbedding


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
        self.scale = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # q: [B, num_heads,    T, head_dim]
        # k: [B, num_kv_heads, T, head_dim]
        # v: [B, num_kv_heads, T, head_dim]

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        # scores: [B, num_heads, T, T]

        sequence_length = q.shape[-2]
        future_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=q.device,
        ).triu(diagonal=1)

        scores = scores.masked_fill(future_mask, float("-inf"))

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
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        q = self.q_proj(x).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        q, k = self.rotary_emb(q, k, positions)

        context = self.attn(q, k, v)

        context = context.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.num_heads * self.head_dim,
        )

        return self.o_proj(context)