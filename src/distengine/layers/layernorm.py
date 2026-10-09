import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_float = x.float()

        variance = x_float.square().mean(
            dim=-1,
            keepdim=True,
        )

        normalized = x_float * torch.rsqrt(variance + self.eps)

        return normalized.to(original_dtype) * self.weight