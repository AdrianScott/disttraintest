import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from datasets import load_dataset
from pathlib import Path
import wandb

from src.model import TinyGPT
from src.utils.seed import set_seed

# --- Configuration ---
# Model params
VOCAB_SIZE = 16_000 # Should match tokenizer
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
MAX_LEN = 512

# Training params
BATCH_SIZE = 16 # Reduced for local GPU memory
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
SEQ_LEN = 256
WANDB_PROJECT = "tiny-transformer-from-scratch"

# Paths
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.json"
CHECKPOINT_DIR = Path(__file__).parent.parent / "runs"

# --- Dataset ---
class TextDataset(Dataset):
    """ A simple dataset to serve tokenized text chunks. """
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        # Calculate the number of full sequences we can create
        return (len(self.token_ids) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        # Input is the sequence, target is the sequence shifted by one
        inputs = torch.tensor(self.token_ids[start:end], dtype=torch.long)
        targets = torch.tensor(self.token_ids[start+1:end+1], dtype=torch.long)
        return inputs, targets

# --- Main Training Logic ---
def train():
    """ Main function to run the training loop. """
    # Reproducibility
    used_seed = set_seed(deterministic=False)
    print(f"Seeding with {used_seed}")
    # 1. Initialization
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    wandb.init(project=WANDB_PROJECT, config={
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
    })

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Load Tokenizer
    if not TOKENIZER_PATH.exists():
        print(f"Tokenizer not found at {TOKENIZER_PATH}.")
        print("Please run `python data/prepare_data.py` first.")
        return
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # 3. Load and Prepare Data (train/val split)
    print("Loading and tokenizing WikiText-2 dataset...")
    full = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    all_text = " ".join([text for text in full["text"] if text.strip()])
    token_ids = tokenizer.encode(all_text).ids

    split_idx = int(0.95 * len(token_ids))
    train_ids, val_ids = token_ids[:split_idx], token_ids[split_idx:]

    train_dataset = TextDataset(train_ids, SEQ_LEN)
    val_dataset = TextDataset(val_ids, SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Datasets created: train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    # 4. Model, Optimizer, Loss, Scheduler
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * NUM_EPOCHS)

    # 5. Training Loop
    print("Starting training...")
    step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            logits, _ = model(inputs)
            # Reshape for loss calculation
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{step}], Loss: {loss.item():.4f}")
            step += 1

        # Validation: compute loss and perplexity
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                batch_loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
                val_loss += batch_loss.item() * inputs.size(0)
                val_tokens += inputs.size(0)
        val_loss /= max(val_tokens, 1)
        ppl = torch.exp(torch.tensor(val_loss)).item()
        print(f"Validation — Epoch {epoch+1}: loss={val_loss:.4f}, ppl={ppl:.2f}")
        wandb.log({"val/loss": val_loss, "val/ppl": ppl, "epoch": epoch + 1})

    # 6. Save final model
    final_path = CHECKPOINT_DIR / "checkpoint.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")
    wandb.finish()

if __name__ == "__main__":
    train()
