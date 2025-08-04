import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """ A standard Causal Self-Attention module with KV Cache support and Flash Attention. """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)
        # Regularization
        self.resid_dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout

    def forward(self, x, use_cache=False, kv_cache=None):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (d_model)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        # KV Caching logic
        new_kv_cache = None
        if use_cache:
            if kv_cache is not None:
                # Append new k, v to cache
                past_k, past_v = kv_cache
                k = torch.cat((past_k, k), dim=2)
                v = torch.cat((past_v, v), dim=2)
            new_kv_cache = (k, v)
        
        # When kv_cache is used, we don't want to apply a causal mask in SDPA,
        # because the query (new token) should attend to all keys (past tokens).
        # is_causal=True is for training on a full sequence without a cache.
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=(kv_cache is None)
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv_cache

class TransformerBlock(nn.Module):
    """ A standard Transformer block. """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, use_cache=False, kv_cache=None):
        attn_output, new_kv_cache = self.attn(self.ln_1(x), use_cache=use_cache, kv_cache=kv_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache