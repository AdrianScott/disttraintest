import torch
import pytest
from src.model import TinyGPT

# Model parameters consistent with DDP script
VOCAB_SIZE = 16000
D_MODEL = 512
N_LAYERS = 4
N_HEADS = 8
BATCH_SIZE = 8
SEQ_LEN = 64
MAX_LEN = 512

@pytest.fixture
def model() -> TinyGPT:
    """Instantiates the TinyGPT model for testing."""
    return TinyGPT(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        n_layers=N_LAYERS, 
        n_heads=N_HEADS,
        max_len=MAX_LEN
    )

def test_model_shape_flow(model: TinyGPT):
    """ 
    Tests that a tensor of token IDs flows through the model and produces
    logits of the correct shape.
    """
    # Input is a batch of token ID sequences
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # Forward pass
    logits, kv_cache = model(x)

    # Output should be logits with shape (batch_size, seq_len, vocab_size)
    expected_shape = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert logits.shape == expected_shape, \
        f"Expected logits shape {expected_shape}, but got {logits.shape}"
    
    # Check KV cache structure
    assert isinstance(kv_cache, list), "KV cache should be a list"
    assert len(kv_cache) == N_LAYERS, f"Expected {N_LAYERS} cache entries, one for each layer"
    assert isinstance(kv_cache[0], tuple) and len(kv_cache[0]) == 2, "Each cache entry should be a (k, v) tuple"
    
    # Check shape of a key tensor from the first layer
    k_cache_shape = kv_cache[0][0].shape
    expected_k_shape = (BATCH_SIZE, N_HEADS, SEQ_LEN, D_MODEL // N_HEADS)
    assert k_cache_shape == expected_k_shape, f"Expected K-cache shape {expected_k_shape}, but got {k_cache_shape}"


def test_model_creation(model: TinyGPT):
    """Tests that the model and its components are created correctly."""
    assert isinstance(model, TinyGPT)
    assert len(model.layers) == N_LAYERS, f"Expected {N_LAYERS} layers, but got {len(model.layers)}"
    assert model.lm_head.out_features == VOCAB_SIZE, f"Expected output layer with {VOCAB_SIZE} units."

def test_causal_mask(model: TinyGPT):
    """ 
    Tests that the causal mask prevents attention to future tokens.
    It does this by checking that the output at a given timestep `t` is 
    the same whether the input sequence is of length `t+1` or `t+2`.
    """
    # Input sequences of two different lengths
    input1 = torch.randint(0, VOCAB_SIZE, (1, 5))
    input2 = input1[:, :-1].clone() # A shorter sequence (length 4)

    # Get outputs
    logits1, _ = model(input1)
    logits2, _ = model(input2)

    # The logits for the first 4 tokens of the longer sequence...
    logits1_prefix = logits1[:, :-1, :]

    # ...should be nearly identical to the logits of the shorter sequence.
    assert torch.allclose(logits1_prefix, logits2, atol=1e-6), \
        "Model output should not depend on future tokens due to causal mask."