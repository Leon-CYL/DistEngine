import torch
from torch import nn
from torch.nn import functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size)
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)
    
    
class LMHead(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size)
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)