import torch.nn as nn
import torch
import math


# Embedding layer to create embedding from text
class InputEmbedding(nn.Module):
    """
    Embedding Layer to get text embedding

    Args:
        d_model (int): Size of embedding (output size of embedding layer)
        vocab_size (int): Size of vocabulary
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Droupout(dropout)

        # Create matrix of shape (seq_len, d_model)
        pe = torch.zeroes(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))

        # Apply sine to even position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        # add batch dim to pe
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # register buffer is used to save the rensor in the model file when saving
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
