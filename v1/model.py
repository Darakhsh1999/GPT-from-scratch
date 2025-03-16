import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """ Single self-attention head"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.Q = nn.Linear(config["d_embed"], config["d_embed"]//config["n_heads"], bias=False)
        self.K = nn.Linear(config["d_embed"], config["d_embed"]//config["n_heads"], bias=False)
        self.V = nn.Linear(config["d_embed"], config["d_embed"]//config["n_heads"], bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config["block_size"],config["block_size"])))
    

    def forward(self, x:torch.Tensor):
        B,T,head_size = x.shape()
        q = self.Q(x) # [B,T,head_size]
        k = self.K(x) # [B,T,head_size]
        v = self.V(x) # [B,T,head_size]
        _tril = self.get_buffer("tril")

        attention_pattern = q @ k.transpose(-2,-1) / math.sqrt(self.config["block_size"])
        attention_pattern = attention_pattern.masked_fill(_tril[:T,:T] == 0, float('-inf'))
        attention_pattern = F.softmax(attention_pattern, dim=-1)

        attention = attention_pattern @ v

        return attention


class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config["n_heads"])])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention_heads = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config["d_embed"])
        self.layer_norm2 = nn.LayerNorm(config["d_embed"])
        self.linear_up = nn.Linear(config["d_embed"], 4*config["d_embed"])
        self.linear_down = nn.Linear(4*config["d_embed"], config["d_embed"])
        
    def forward(self, x):
        x = self.layer_norm1(x + self.attention_heads(x))
        x = self.layer_norm2(x + self.linear_down(F.relu(self.linear_up(x))))
        return x


class Transformer(nn.Module):
    
    def __init__(self, vocab_size, config):
        super().__init__()

        self.config = config
        self.embedding = nn.Embedding(vocab_size, config["d_embed"])
        self.position_embedding = nn.Embedding(config["block_size"], config["d_embed"])
        self.transformer_layers = nn.Sequential(*[Block(config) for _ in range(config["n_layers"])])
        self.output_mapping = nn.Linear(config["d_embed"], vocab_size)

    def forward(self, x):
        B,T = x.shape # [B,T]
        
        # Input embedding
        token_embedding = self.embedding(x) # [B,T,d_embed]
        position_embedding = self.position_embedding(torch.arange(T, device=self.config["device"])) # [T,d_embed]
        embedding = token_embedding + position_embedding # [B,T,d_embed]

        # Transformer blocks
        x = self.transformer_layers(x)

        # Output mapping
        x = self.output_mapping(x)

        return x







if __name__ == "__main__":

    model = Transformer(vocab_size=32, )

    x = torch.randn(size=())