import torch
from tokenizers import Tokenizer
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F

from src.model import TinyGPT

# --- Configuration ---
VOCAB_SIZE = 16_000
D_MODEL = 512
N_LAYERS = 4
N_HEADS = 8
MAX_LEN = 512

# Paths
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.json"
DEFAULT_CHECKPOINT = ARTIFACTS_DIR.parent / "runs" / "checkpoint.pt"

def generate(checkpoint_path: Path, prompt: str, max_new_tokens: int, method: str, temperature: float, top_k: int, top_p: float):
    """ Loads a trained model and generates text with advanced sampling. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load tokenizer and model
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        print("DDP checkpoint detected. Cleaning state dict...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Prepare the prompt
    token_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 3. Generate tokens with KV Caching
    print(f"Generating from prompt: '{prompt}' with method '{method}'")
    generated_ids = list(token_ids)
    kv_caches = None

    with torch.no_grad():
        # Process the initial prompt and get the first set of logits and the initial KV cache
        logits, kv_caches = model(input_tensor, kv_caches=None)
        next_token_logits = logits[:, -1, :]

        # Autoregressive generation loop
        for _ in range(max_new_tokens):
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply Top-K
            if method == 'top-k':
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)

            # Apply Top-P (Nucleus Sampling)
            if method == 'top-p':
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()

            # Sample the next token
            if method == 'greedy':
                next_token_id = torch.argmax(probs, dim=-1).unsqueeze(0)
            else:
                next_token_id = torch.multinomial(probs, num_samples=1)

            # Check for end-of-sequence token
            if next_token_id.item() == tokenizer.token_to_id('</s>'):
                break

            generated_ids.append(next_token_id.item())

            # Feed the new token back into the model
            logits, kv_caches = model(next_token_id, kv_caches=kv_caches)
            next_token_logits = logits[:, -1, :]

    # 4. Decode and print
    generated_text = tokenizer.decode(generated_ids)
    print(f"\n--- Generated Text ---\n{generated_text}\n----------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a TinyGPT model with advanced sampling.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Path to the model checkpoint file.")
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="The prompt to start generation from.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate.")
    parser.add_argument("--method", type=str, default="top-k", choices=['greedy', 'top-k', 'top-p'], help="Generation method.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling. Higher is more random.")
    parser.add_argument("--top_k", type=int, default=50, help="K for top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.92, help="P for top-p (nucleus) sampling.")
    args = parser.parse_args()

    generate(args.checkpoint, args.prompt, args.max_new_tokens, args.method, args.temperature, args.top_k, args.top_p)
