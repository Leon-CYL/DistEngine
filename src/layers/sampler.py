import torch
from torch import nn


class GreedySampler(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, vocab_size]
        return logits.argmax(dim=-1)
        # token IDs: [B]