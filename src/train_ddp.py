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

import argparse
from src.model import TinyGPT

# --- Constants ---
VOCAB_SIZE = 16_000 # Set by tokenizer

# --- Paths ---
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.json"
CHECKPOINT_DIR = Path(__file__).parent.parent / "runs_ddp"

# --- Argument Parser ---
def get_args():
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a TinyGPT model with DDP.")
    # Model params
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length for positional embeddings")
    # Training params
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length for training")
    # W&B
    parser.add_argument("--wandb_project", type=str, default="tiny-transformer-from-scratch-ddp", help="WandB project name")
    return parser.parse_args()

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

def prepare_data(tokenizer, rank, world_size):
    """Downloads, tokenizes, and caches the dataset to prevent redundant processing."""
    CACHE_PATH = ARTIFACTS_DIR / "wikitext-2-raw-v1_train_token_ids.pt"

    if rank == 0:
        if CACHE_PATH.exists():
            print(f"Tokenized data found at {CACHE_PATH}")
        else:
            print("Tokenized data not found. Downloading and tokenizing wikitext-2...")
            wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            all_text = " ".join([text for text in wikitext["text"] if text.strip()])
            token_ids = tokenizer.encode(all_text).ids
            torch.save(token_ids, CACHE_PATH)
            print(f"Tokenized data saved to {CACHE_PATH}")

    if world_size > 1:
        dist.barrier()

    token_ids = torch.load(CACHE_PATH, map_location='cpu')
    return token_ids

# --- DDP Setup ---
def setup_ddp():
    """ Initializes the distributed process group. """
    # Set the backend. NCCL is recommended for NVIDIA GPUs.
    dist.init_process_group(backend="nccl")
    # Set the device for the current process.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """ Cleans up the distributed process group. """
    dist.destroy_process_group()

# --- Main Training Logic ---
def train():
    """ Main function to run the DDP training loop. """
    args = get_args()
    setup_ddp()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 1. Initialization (only on main process)
    if rank == 0:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        # Combine args with other relevant info for wandb config
        config = vars(args)
        config["vocab_size"] = VOCAB_SIZE
        config["batch_size_total"] = args.batch_size * world_size
        config["world_size"] = world_size
        wandb.init(project=args.wandb_project, config=config)
        print(f"DDP training started on {world_size} GPUs.")
        print(f"Run configuration: {config}")

    # 2. Load Tokenizer and Prepare Data
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    token_ids = prepare_data(tokenizer, rank, world_size)
    
    train_dataset = TextDataset(token_ids, args.seq_len)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
    
    if rank == 0:
        print(f"Dataset prepared with {len(train_loader)} batches per GPU.")

    # 4. Model, Optimizer, Loss, Scheduler
    model = TinyGPT(VOCAB_SIZE, args.d_model, args.n_layers, args.n_heads, args.max_len).to(local_rank)
    # We set find_unused_parameters=True because our model's forward pass
    # has logic for a KV cache that is not used during training. This prevents
    # DDP from hanging when it can't find gradients for those unused parameters.
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.num_epochs)

    # 5. Training Loop
    if rank == 0:
        last_log_time = time.time()
        
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(local_rank, non_blocking=True)
            targets = targets.to(local_rank, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(inputs) # model returns (logits, kv_cache)
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if i % 100 == 0 and rank == 0:
                current_time = time.time()
                elapsed_time = current_time - last_log_time
                steps_since_last_log = 100 if i > 0 else 1
                tokens_processed = steps_since_last_log * args.batch_size * args.seq_len * world_size
                throughput = tokens_processed / elapsed_time if elapsed_time > 0 else 0
                last_log_time = current_time

                wandb.log({
                    "train_loss": loss.item(), 
                    "throughput_tokens_per_sec": throughput,
                    "lr": scheduler.get_last_lr()[0]
                })
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step {i}, Loss: {loss.item():.4f}, Throughput: {throughput:.2f} tokens/sec")

        if rank == 0:
            avg_loss = epoch_loss / len(train_loader)
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch})
            print(f"Epoch [{epoch+1}/{args.num_epochs}] finished. Average Loss: {avg_loss:.4f}")

            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pt"
                torch.save(model.module.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

    # 6. Final Cleanup
    if rank == 0:
        final_model_path = CHECKPOINT_DIR / "model_final.pt"
        torch.save(model.module.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    train()