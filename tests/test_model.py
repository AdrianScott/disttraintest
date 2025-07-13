import torch
import pytest
from src.model import TinyGPT

# Model parameters from draft-plan.md
VOCAB_SIZE = 16000
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
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
    output = model(x)
    
    # Output should be logits with shape (batch_size, seq_len, vocab_size)
    expected_shape = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"

def test_model_creation(model: TinyGPT):
    """Tests that the model and its components are created correctly."""
    assert isinstance(model, TinyGPT)
    assert len(model.layers) == N_LAYERS, f"Expected {N_LAYERS} layers, but got {len(model.layers)}"
    assert model.fc_out.out_features == VOCAB_SIZE, f"Expected output layer with {VOCAB_SIZE} units."

def test_causal_mask(model: TinyGPT):
    """ 
    Tests that the causal mask prevents attention to future tokens.
    It does this by checking that the output at a given timestep `t` is 
    the same whether the input sequence is of length `t+1` or `t+2`.
    """
    # Input sequences of two different lengths
    input1 = torch.randint(0, VOCAB_SIZE, (1, 5))
    input2 = input1.clone()

    # Get outputs
    output1 = model(input1)
    output2 = model(input2[:, :-1]) # Pass a shorter sequence

    # The logits for the first 4 tokens should be nearly identical
    assert torch.allclose(output1[:, :-1, :], output2, atol=1e-6), \
        "Model output should not depend on future tokens due to causal mask."
