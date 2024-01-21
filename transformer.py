import torch.nn as nn
import torch
import math
from utils import FeedForwardBlock, ResidualConnection, LayerNormalization


# Multihead attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, droupout: float):
        """MultiHeadAttention layer

        Args:
            d_model (int): Embedding size
            h (int): number of attention heads needed
            droupout (float): dropout value
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert self.d_model % self.h == 0, "d_model should be divisible by h"

        self.d_k = self.d_model // self.h # divide emb by no of heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(self.d_k * self.h, d_model)

        self.droupout = nn.Dropout(droupout)
    
    @staticmethod
    def attention(query, key, value, mask, droupout):
        d_k = query.shape[-1]
        print(f'Shape of query:- {query.shape}')
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        ## Using torch.bmm (bmm only works in 3d so reshape to 3 dim and after bmm again revert to og shape)
        # b, h, seq_len, feat_dim = query.shape
        # attention_score = torch.bmm(query.reshape(b*h, seq_len,feat_dim), key.reshape(b*h, seq_len,feat_dim).transpose(-2, -1))
        # attention_score = attention_score.contiguous().view(b, h, seq_len, seq_len) / math.sqrt(d_k)

        print(attention_score.shape)
        # apply mask if provided
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
    
        attention_score = attention_score.softmax(dim=-1) # (batch, no_head, seq_len, seq_len)
        if droupout is not None:
            attention_score = droupout(attention_score)
        
        return (attention_score @ value), attention_score
        # return torch.bmm(attention_score, value), attention_score
    

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq-len, d-model) --> (batch, seq-len, d-model)
        key = self.w_k(k) # (batch, seq-len, d-model) --> (batch, seq-len, d-model)
        value = self.w_v(v) # (batch, seq-len, d-model) --> (batch, seq-len, d-model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)  # (batch, seq-len, d-model) --> (batch, seq-len, no_head, d-k)
        query = query.transpose(1,2)  # (batch, seq-len,no_head, d-k) --> (batch, no_h, seq-len, d-k)

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)  # (batch, seq-len, d-model) --> (batch, seq-len, no_head, d-k)
        key = key.transpose(1,2)  # (batch, seq-len,no_head, d-k) --> (batch, no_h, seq-len, d-k)

        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k)  # (batch, seq-len, d-model) --> (batch, seq-len, no_head, d-k)
        value = value.transpose(1,2)  # (batch, seq-len,no_head, d-k) --> (batch, no_h, seq-len, d-k)

        x, self.attention_score = self.attention(query, key, value, mask, self.droupout)
        x = x.transpose(1,2) # (batch, no_h, seq-len, d_k) --> (batch, seq-len, no_h, d_k)
        x = x.contiguous().view(x.shape[0], x.shape[1], -1) # (batch, seq-len, no_h, d_k) --> (batch, seq-len, d-model)

        return self.w_o(x), self.attention_score


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, droupout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # use modulelist for residual blocks
        self.residual_connections = nn.ModuleList([ResidualConnection(droupout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

