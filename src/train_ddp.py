import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.amp import GradScaler, autocast
from tokenizers import Tokenizer
from datasets import load_dataset
from pathlib import Path
import wandb
import os
import sys
import time
import argparse

from src.model import TinyGPT
from src.utils.perf import AverageMeter

# --- Constants ---
VOCAB_SIZE = 16_000  # Set by tokenizer

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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients before optimizer step")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers per process")
    parser.add_argument("--log_every", type=int, default=20, help="Steps between logs on rank 0")
    parser.add_argument("--profile_first_steps", type=int, default=0, help="Steps to run with torch.profiler on rank0")
    # W&B
    parser.add_argument("--wandb_project", type=str, default="tiny-transformer-from-scratch-ddp", help="WandB project name")
    # DDP / perf
    parser.add_argument("--bucket_cap_mb", type=int, default=25, help="DDP gradient bucket size (MB)")
    parser.add_argument("--disable_broadcast_buffers", action="store_true", help="Disable DDP broadcast buffers")
    parser.add_argument("--find_unused_parameters", action="store_true", help="Set True only if needed")
    parser.add_argument("--n_processes_per_node", type=int, default=1, help="Number of processes per node")
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

# --- DDP Helper ---
def log_with_rank(rank, msg):
    """Prepends a rank-specific prefix to a log message."""
    print(f"[Rank {rank}] {msg}")

def cleanup_ddp():
    """ Cleans up the distributed process group. """
    dist.destroy_process_group()

# --- Main Training Logic ---
def train(args):
    """ Main function to run the DDP training loop. """
    # Info from torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    device_type = device.type

    # Speed-related flags
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Init process group
    log_with_rank(rank, "Initializing process group...")
    dist.init_process_group(backend="nccl", init_method="env://")
    log_with_rank(rank, f"Process group initialized. World size: {world_size}")

    # 1. Initialization (only on main process)
    if rank == 0:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        config = vars(args)
        config["vocab_size"] = VOCAB_SIZE
        config["batch_size_total"] = args.batch_size * world_size * args.gradient_accumulation_steps
        config["world_size"] = world_size
        wandb.init(project=args.wandb_project, config=config)
        print(f"Starting DDP training on {world_size} GPUs.")
        print(f"Effective batch size: {config['batch_size_total']} tokens sequences")

    # 2. Load Tokenizer
    # All ranks load the tokenizer, but only rank 0 prints messages
    if rank == 0: print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    log_with_rank(rank, "Tokenizer loaded.")

    # 3. Load and Prepare Data
    if rank == 0: print("Loading and tokenizing dataset...")
    # Only rank 0 should download. Other ranks will use the cache.
    if rank != 0: dist.barrier()
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    if rank == 0: dist.barrier()

    all_text = " ".join([text for text in wikitext["text"] if text.strip()])
    token_ids = tokenizer.encode(all_text).ids
    log_with_rank(rank, f"Tokenized {len(token_ids)} tokens.")

    train_dataset = TextDataset(token_ids, args.seq_len)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    log_with_rank(rank, f"DataLoader created with {len(train_loader)} batches.")


    # 4. Model, Optimizer, Loss, Scheduler
    log_with_rank(rank, "Creating model...")
    model = TinyGPT(VOCAB_SIZE, args.d_model, args.n_layers, args.n_heads, args.max_len).to(device)

    # For PyTorch 2.0, compile the model for a significant speedup
    log_with_rank(rank, "Compiling model with torch.compile()...")
    model = torch.compile(model)
    log_with_rank(rank, "Model compiled.")

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=args.find_unused_parameters,
        broadcast_buffers=not args.disable_broadcast_buffers,
        bucket_cap_mb=args.bucket_cap_mb,
    )
    log_with_rank(rank, "Model wrapped in DDP.")


    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.num_epochs)

    # Mixed precision scaler
    scaler = GradScaler()

    # Meters
    loss_meter = AverageMeter()
    step_time_meter = AverageMeter()
    tokens_meter = AverageMeter()

    # Optional profiler on rank 0
    profiler = None
    if args.profile_first_steps > 0 and rank == 0:
        from torch.profiler import profile, record_function, ProfilerActivity
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=args.profile_first_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_prof"),
            record_shapes=True,
            with_stack=True,
        )
        profiler.__enter__()

    # Barrier to ensure all processes have a correctly setup model before starting
    log_with_rank(rank, "Waiting at barrier before training loop...")
    dist.barrier()
    log_with_rank(rank, "Barrier passed. Starting training.")

    # 5. Training Loop
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        loss_meter.reset()
        tokens_meter.reset()

        # Iterate
        for i, (inputs, targets) in enumerate(train_loader):
            batch_start = time.time()

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Use no_sync during gradient accumulation microsteps
            sync_needed = ((i + 1) % args.gradient_accumulation_steps == 0) or ((i + 1) == len(train_loader))
            context = model.no_sync() if not sync_needed else torch.enable_grad()
            with context:
                # Use bfloat16 for mixed precision, which is generally better for transformers
                with autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = model(inputs)  # model returns (logits, caches)
                    loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if sync_needed:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Metrics
            tokens_processed = inputs.numel()
            tokens_meter.update(tokens_processed)
            loss_meter.update(loss.item() * args.gradient_accumulation_steps)
            step_time_meter.update(time.time() - batch_start)
            global_step += 1

            # Logging (rank 0 only)
            if rank == 0 and (global_step % args.log_every == 0):
                per_gpu_tps = tokens_meter.avg / step_time_meter.avg
                total_tps = per_gpu_tps * world_size
                wandb.log({
                    "train/loss": loss_meter.avg,
                    "train/per_gpu_tokens_per_s": per_gpu_tps,
                    "train/total_tokens_per_s": total_tps,
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                    "epoch": epoch
                })
                print(f"[E{epoch+1} S{global_step}] loss={loss_meter.avg:.4f} "
                      f"perGPU={per_gpu_tps:.0f} tok/s total={total_tps:.0f} tok/s")

            if profiler is not None:
                profiler.step()

        # Epoch end
        epoch_dur = time.time() - epoch_start
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs} finished in {epoch_dur:.1f}s. Avg loss {loss_meter.avg:.4f}")
            wandb.log({"epoch/loss": loss_meter.avg, "epoch/time_s": epoch_dur, "epoch": epoch})

            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pt"
                torch.save(model.module.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        # Sync at epoch boundary once
        dist.barrier()

    # Cleanup
    if rank == 0:
        final_model_path = CHECKPOINT_DIR / "model_final.pt"
        torch.save(model.module.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        wandb.finish()

    if profiler is not None:
        profiler.__exit__(None, None, None)

    cleanup_ddp()
    log_with_rank(rank, "Training complete. Process finished.")

if __name__ == "__main__":
    args = get_args()
    train(args)