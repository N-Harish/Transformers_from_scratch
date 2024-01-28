import torch.nn as nn
import torch
import math
from utils import FeedForwardBlock, ResidualConnection, LayerNormalization
from positional_embedding import InputEmbedding, PositionalEmbedding


# Multihead attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        """MultiHeadAttention layer

        Args:
            d_model (int): Embedding size
            h (int): number of attention heads needed
            dropout (float): dropout value
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

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout):
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
        if dropout is not None:
            attention_score = dropout(attention_score)
        
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

        x, self.attention_score = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2) # (batch, no_h, seq-len, d_k) --> (batch, seq-len, no_h, d_k)
        x = x.contiguous().view(x.shape[0], x.shape[1], -1) # (batch, seq-len, no_h, d_k) --> (batch, seq-len, d-model)

        return self.w_o(x), self.attention_score


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # use modulelist for residual blocks
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

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


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEmbedding, tgt_pos: PositionalEmbedding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create embed layer
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # positional encoding layer
    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)

    # create encoder blocks
    encoder_blocks = []

    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

