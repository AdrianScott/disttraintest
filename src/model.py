import torch
import torch.nn as nn
from .layers import TransformerBlock

class TinyGPT(nn.Module):
    """ A standard GPT model with KV Caching. """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])

        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, use_cache=False, kv_caches=None):
        B, T = idx.size()
        assert T <= self.max_len, f"Cannot forward sequence of length {T}, block size is only {self.max_len}"

        # If we are in generation mode (using KV cache), the position is the length of the cache
        pos_start = 0
        if use_cache and kv_caches is not None:
             # Get cached sequence length from the first layer's key cache
             pos_start = kv_caches[0][0].size(2)
        pos = torch.arange(pos_start, pos_start + T, dtype=torch.long, device=idx.device).unsqueeze(0)

        # Forward the embeddings
        tok_emb = self.token_embedding(idx) # (B, T, d_model)
        pos_emb = self.position_embedding(pos) # (1, T, d_model)
        x = self.dropout(tok_emb + pos_emb)

        # Forward through the transformer blocks
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_caches[i] if use_cache and kv_caches is not None else None
            x, new_kv_cache = layer(x, use_cache=use_cache, kv_cache=layer_kv_cache)
            if use_cache:
                new_kv_caches.append(new_kv_cache)

        # Final normalization and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Always return a tuple for a consistent API, which is important for DDP and torch.compile.
        # new_kv_caches will be an empty list if use_cache is False.
        return logits, new_kv_caches