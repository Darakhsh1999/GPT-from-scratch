import torch
from dataclasses import dataclass

@dataclass
class GPTConfig():

    ### Architecture
    context_size: int = 1024 # Max sequence length
    vocab_size: int = 50257 # vocabulary size
    n_layers: int = 12 # number of layers, l
    n_heads: int = 12 # number of heads, h
    d_embed: int = 768 # embedding dimension, d_e
    use_flash_attention: bool = True

    ### Training params
    n_epochs: int = 30
    batch_size: int = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"