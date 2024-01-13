import torch
import torch.nn as nn


# LayerNorm layer
class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeroes(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dims=-1, keepdims=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# Feed forward block
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, droupout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.droupout = nn.Dropout(droupout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.droupout(torch.relu(self.linear_1(x))))


# Residual block
class ResidualConnection(nn.Module):

    def __init__(self,droupout: float) -> None:
        super().__init__()
        self.droupout = nn.Dropout(droupout)
        self.norm = LayerNormalization()

    def forward(self, x, sub_layer):
        return x + self.droupout(sub_layer(self.norm(x)))