import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig
from torchsummary import summary


class MultiLayerPerceptron(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.linear_up = nn.Linear(config.d_embed, 4*config.d_embed)
        self.linear_down = nn.Linear(4*config.d_embed, config.d_embed)
        self.linear_down.SCALE_INIT = 1
        self.gelu = nn.GELU(approximate="tanh")
    
    def forward(self, x):
        x = self.linear_up(x)
        x = self.gelu(x)
        x = self.linear_down(x)
        return x

class MaskedMultiHeadAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.d_embed % config.n_heads == 0
        self.config = config

        # Size vars
        self.d_embed = config.d_embed
        self.n_heads = config.n_heads
        self.head_size = config.d_embed // config.n_heads # d_h (d_e = d_h*h)
        
        self.QKV = nn.Linear(config.d_embed, 3*config.d_embed) # Compact QKV 
        self.output_projection = nn.Linear(config.d_embed, config.d_embed)
        self.output_projection.SCALE_INIT = 1

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_size,config.context_size))
        )

    def forward(self, x):
        B,T,d_emb = x.shape

        qkv = self.QKV(x) # [B,T,3*d_e]
        q,k,v = qkv.split(self.d_embed, dim=2) # 3*[B,T,d_e]

        # Manipulate view so we can do compact multihead-attention in one single matrix multiplication
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1,2) # [B,T,d_e] -> [B,T,h,d_h] -> [B,h,T,d_h]
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1,2) # [B,T,d_e] -> [B,T,h,d_h] -> [B,h,T,d_h]
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1,2) # [B,T,d_e] -> [B,T,h,d_h] -> [B,h,T,d_h]

        # Attention
        if self.config.use_flash_attention:
            context_vector = F.scaled_dot_product_attention(q, k, v, is_causal=True) # MUCH FASTER!!!
            context_vector = context_vector.transpose(1,2).contiguous().view(B,T,d_emb) # "Concatenation" [B,T,h,d_h] -> # [B,T,d_e]
        else:
            assert k.shape[-1] == self.head_size
            attention_scores = (q @ k.transpose(2,3)) / self.head_size**0.5 # [B,h,T,T]
            masked_attention_scores = attention_scores.masked_fill(self.mask[:T,:T] == 0, -torch.inf) # [B,h,T,T]
            attention_weights = F.softmax(masked_attention_scores, dim=-1) # [B,h,T,T]

            # Context vector
            context_vector = attention_weights @ v # [B,h,T,d_h]
            context_vector = context_vector.transpose(1,2).contiguous().view(B,T,d_emb) # "Concatenation" [B,T,h,d_h] -> # [B,T,d_e]

        # Output projection
        return self.output_projection(context_vector) # [B,T,d_e]


class TransformerDecoderBlock(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_embed)
        self.attention = MaskedMultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_embed)
        self.mlp = MultiLayerPerceptron(config)
    
    def forward(self, x):
        """ LayerNorms are applied before SubLayers, unliked in original Transformer """
        x = x + self.attention(self.ln_1(x)) # x <-- x + LN(ATN(X))
        x = x + self.mlp(self.ln_2(x)) # x <-- x + LN(ATN(X))
        return x

class GPT(nn.Module):


    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "token_embedding": nn.Embedding(config.vocab_size, config.d_embed),
            "position_embedding": nn.Embedding(config.context_size, config.d_embed),
            "transformer_blocks": nn.ModuleList([TransformerDecoderBlock(config) for _ in range(config.n_layers)]), # N layers of decoder blocks
            "layer_norm": nn.LayerNorm(config.d_embed) # Final LayerNorm before linear_head
        })

        # Language model head
        self.linear_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)

        # Weight sharing
        self.transformer["token_embedding"].weight = self.linear_head.weight


        # Weight initialization
        self.apply(self._init_weights)
    

    def _init_weights(self, module):
        """ Weight initialization for specific layers in accordance with GPT2 """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2*self.config.n_layers)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        # x Batched token ID sequence [B,T]
        _,T = x.shape
        assert T <= self.config.context_size, f"Input sequence with length {T}, exceeded maximum context size {self.config.context_size}"

        # Embeddings
        positions = torch.arange(T, dtype=torch.long, device=x.device)
        position_embedding = self.transformer["position_embedding"](positions) # [T,d_e]
        token_embedding = self.transformer["token_embedding"](x) # [B,T,d_e]
        x = position_embedding + token_embedding # Combine embeddings [B,T,d_e]

        # Propogate through Transformer decoder blocks 
        for transformer_block in self.transformer["transformer_blocks"]:
            x = transformer_block(x)
        
        # Map to vocabulary logits
        logits = self.linear_head(self.transformer["layer_norm"](x)) # [B,T,vocab_size]

        # Calculate loss
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1)) # [B*T,vocab_size]
        else:
            loss = None

        return logits, loss
    

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        
        model.eval()
        for _ in range(max_new_tokens):

            # If the sequence context is growing too long we must crop it at context_size
            idx_cond = idx if idx.size(1) <= self.config.context_size else idx[:, -self.config.context_size:]

            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
        



if __name__ == "__main__":

    config = GPTConfig()
    model = GPT(config)

    example_input = torch.randint(0,config.vocab_size,(3,100), dtype=torch.long)
    targets = torch.randint(0,config.vocab_size,(3,100), dtype=torch.long)
    output, loss = model(example_input, targets)
    print(output.shape)
    if loss:
        print(f"Loss = {loss:.3f}")
    
    example_input = torch.randint(0,config.vocab_size,(1,100), dtype=torch.long)
    generated_text = model.generate(example_input, max_new_tokens=50)
    #print(generated_text.shape)
    

    summary(model, input_data=example_input)