import torch
from torch import nn
from torch.nn import functional as f

class Linear(nn.Module):
    def __init__(self, input_size, output_size, bias=False):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_parameter("bias", None)
            
    def forward(self, x):
        return f.linear(x, self.weight, self.bias)