import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tokenizers import Tokenizer
from datasets import load_dataset
from pathlib import Path
import wandb
import os
import time

from src.model import TinyGPT

# --- Configuration ---
# Model params
VOCAB_SIZE = 16_000
D_MODEL = 512  # Increased model dimension for more capacity
N_LAYERS = 4   # Keeping layers the same for now
N_HEADS = 8    # Increased heads to keep head dimension the same (512/8=64)
MAX_LEN = 512

# Training params
BATCH_SIZE = 16 # This will be per-GPU
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
SEQ_LEN = 256
WANDB_PROJECT = "tiny-transformer-from-scratch-ddp"

# Paths
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.json"
CHECKPOINT_DIR = Path(__file__).parent.parent / "runs_ddp"

# --- Dataset ---
class TextDataset(Dataset):
    """ A simple dataset to serve tokenized text chunks. """
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.token_ids) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        inputs = torch.tensor(self.token_ids[start:end], dtype=torch.long)
        targets = torch.tensor(self.token_ids[start+1:end+1], dtype=torch.long)
        return inputs, targets

# --- DDP Setup ---
def setup_ddp():
    """ Initializes the distributed process group. """
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """ Cleans up the distributed process group. """
    dist.destroy_process_group()

# --- Main Training Logic ---
def train():
    """ Main function to run the DDP training loop. """
    setup_ddp()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 1. Initialization (only on main process)
    if rank == 0:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        wandb.init(project=WANDB_PROJECT, config={
            "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "learning_rate": LEARNING_RATE,
            "batch_size_total": BATCH_SIZE * world_size,
            "num_epochs": NUM_EPOCHS,
            "seq_len": SEQ_LEN,
            "world_size": world_size
        })
        print(f"DDP training started on {world_size} GPUs.")

    # 2. Load Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # 3. Load and Prepare Data
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    all_text = " ".join([text for text in wikitext["text"] if text.strip()])
    token_ids = tokenizer.encode(all_text).ids

    train_dataset = TextDataset(token_ids, SEQ_LEN)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)

    # 4. Model, Optimizer, Loss, Scheduler
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, MAX_LEN).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * NUM_EPOCHS)

    # 5. Training Loop
    print(f"[{rank}] Starting training...")
    total_steps = 0
    for epoch in range(NUM_EPOCHS):
        train_loader.sampler.set_epoch(epoch) # Important for shuffling
        for step, (inputs, targets) in enumerate(train_loader):
            step_start_time = time.time()
            inputs, targets = inputs.to(local_rank), targets.to(local_rank)

            optimizer.zero_grad()
            # Forward pass
            # The model returns logits and the kv_cache, we only need logits for training
            logits, _ = model(inputs)
            # Calculate loss
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_steps += 1

            if rank == 0 and total_steps % 100 == 0:
                step_time = time.time() - step_start_time
                throughput = (BATCH_SIZE * world_size) / step_time
                wandb.log({
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "throughput_samples_per_sec": throughput
                })
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{total_steps}], Loss: {loss.item():.4f}, Throughput: {throughput:.2f} samples/sec")

        # Save a checkpoint periodically (only on main process)
        if rank == 0 and (epoch + 1) % 5 == 0:
            epoch_checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.module.state_dict(), epoch_checkpoint_path)
            print(f"Saved epoch checkpoint to {epoch_checkpoint_path}")

    # 6. Save final model (only on main process)
    if rank == 0:
        final_path = CHECKPOINT_DIR / "checkpoint_ddp.pt"
        torch.save(model.module.state_dict(), final_path)
        print(f"Training complete. Final model saved to {final_path}")
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    train()
