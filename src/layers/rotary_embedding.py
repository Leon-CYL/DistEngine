import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (indices / head_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q:         [B, query_heads, T, head_dim]
        # k:         [B, kv_heads,    T, head_dim]
        # positions: [B, T]

        angles = (positions.float().unsqueeze(-1) * self.inv_freq.float())

        cos = angles.cos().unsqueeze(1)
        sin = angles.sin().unsqueeze(1)

        return (
            self.apply_rotation(q, cos, sin),
            self.apply_rotation(k, cos, sin),
        )

    @staticmethod
    def apply_rotation(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        first, second = x.float().chunk(2, dim=-1)

        rotated_first = first * cos - second * sin
        rotated_second = second * cos + first * sin

        return torch.cat(
            (rotated_first, rotated_second),
            dim=-1,
        ).to(original_dtype)
