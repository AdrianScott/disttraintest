import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
# Handle different PyTorch versions
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    # For PyTorch 2.2+
    from torch.cuda.amp import GradScaler, autocast
from tokenizers import Tokenizer
from datasets import load_dataset
from pathlib import Path
import wandb
import os
import sys
import time
import argparse
import socket
import datetime
import traceback
import signal
import atexit

# Force immediate log flushing to see all messages in real-time
import functools
print = functools.partial(print, flush=True)

# Helper function to format elapsed time in a more readable way
def format_elapsed_time(seconds):
    """Format seconds into hours, minutes, seconds format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    else:
        return f"{seconds}s"

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

# --- DDP Setup ---
def setup_ddp(rank, world_size):
    """Setup already done by torchrun; just sanity info."""
    # Optional: override default timeout
    print(f"Rank {rank}: DDP setup complete with world_size={world_size}")
    return rank, world_size

def test_communication(local_rank, world_size):
    """Quick all-reduce sanity check."""
    rank = dist.get_rank()
    tensor = torch.ones(1, device=f"cuda:{local_rank}") * (rank + 1)
    print(f"Rank {rank}: Starting communication test with tensor={tensor.item()}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    result = abs(tensor.item() - expected) < 1e-3
    print(f"Rank {rank}: Communication test {'PASSED' if result else 'FAILED'}. Got {tensor.item()}, expected {expected}")
    return result

# Flag to track if cleanup has been done
_cleanup_done = False

def cleanup_ddp():
    """Cleans up the distributed process group."""
    global _cleanup_done
    if _cleanup_done:
        return
    
    try:
        if dist.is_initialized():
            rank = dist.get_rank()
            print(f"Rank {rank}: Cleaning up distributed process group", flush=True)
            dist.destroy_process_group()
            print(f"Rank {rank}: Process group destroyed successfully", flush=True)
        else:
            print("DDP not initialized, no cleanup needed", flush=True)
    except Exception as e:
        print(f"Error during cleanup: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
    
    _cleanup_done = True

# Register cleanup handlers
atexit.register(cleanup_ddp)

def signal_handler(sig, frame):
    print(f"Received signal {sig}, cleaning up...", flush=True)
    cleanup_ddp()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def print_network_info():
    """Print network information to help diagnose connection issues."""
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "Unable to determine IP"
    
    print(f"Network info: Hostname={hostname}, Local IP={local_ip}")
    print(f"Environment: MASTER_ADDR={os.environ.get('MASTER_ADDR', 'not set')}, MASTER_PORT={os.environ.get('MASTER_PORT', 'not set')}")
    print(f"RANK={os.environ.get('RANK', 'not set')}, WORLD_SIZE={os.environ.get('WORLD_SIZE', 'not set')}, LOCAL_RANK={os.environ.get('LOCAL_RANK', 'not set')}")
    
    # Try to ping master node if this is not the master
    if os.environ.get('RANK', '0') != '0' and os.environ.get('MASTER_ADDR'):
        master_addr = os.environ.get('MASTER_ADDR')
        print(f"Attempting to reach master node at {master_addr}...")
        try:
            socket.create_connection((master_addr, int(os.environ.get('MASTER_PORT', '29500'))), timeout=5)
            print(f"Successfully connected to master at {master_addr}:{os.environ.get('MASTER_PORT', '29500')}")
        except Exception as e:
            print(f"Failed to connect to master: {str(e)}")

# --- DDP Helper ---
def log_with_rank(rank, msg):
    """Prepends a rank-specific prefix to a log message."""
    print(f"[Rank {rank}] {msg}")

# --- Main Training Logic ---
def train(args):
    """ Main function to run the DDP training loop. """
    # Info from torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Start overall timer
    training_start_time = time.time()
    
    print(f"Rank {rank}: Starting train() function at {time.strftime('%Y-%m-%d %H:%M:%S')} with local_rank={local_rank}, world_size={world_size}")
    
    # Print network information
    print_network_info()
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    device_type = device.type
    
    print(f"Rank {rank}: Using device {device}")

    # Speed-related flags
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Init process group
    print(f"Rank {rank}: Initializing process group with NCCL backend (timeout: 60s)")
    init_start_time = time.time()
    try:
        # Increase timeout for better debugging
        dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=60))
        init_time = time.time() - init_start_time
        print(f"Rank {rank}: Process group initialized successfully in {init_time:.2f}s")
    except Exception as e:
        print(f"Rank {rank}: Failed to initialize process group after {time.time() - init_start_time:.2f}s: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

    # Quick comm test
    print(f"Rank {rank}: Running communication test")
    comm_start_time = time.time()
    if not test_communication(local_rank, world_size):
        if rank == 0:
            print(f"Communication test FAILED after {time.time() - comm_start_time:.2f}s. Aborting.")
        cleanup_ddp()
        sys.exit(1)
    if rank == 0:
        print(f"DDP initialized. World size: {world_size}. Initialization took {time.time() - init_start_time:.2f}s total")

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
        
        if rank == 0:
            elapsed_time = time.time() - training_start_time
            elapsed_str = format_elapsed_time(elapsed_time)
            print(f"[{elapsed_str} elapsed] Starting epoch {epoch+1}/{args.num_epochs}")

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
                elapsed_time = time.time() - training_start_time
                per_gpu_tps = tokens_meter.avg / step_time_meter.avg
                total_tps = per_gpu_tps * world_size
                elapsed_str = format_elapsed_time(elapsed_time)
                print(f"[{elapsed_str} elapsed] Step {global_step}, Epoch {epoch+1}/{args.num_epochs}, Loss: {loss_meter.avg:.4f}, Speed: {total_tps:.1f} tokens/sec")
                wandb.log({
                    "train/loss": loss_meter.avg,
                    "train/per_gpu_tokens_per_s": per_gpu_tps,
                    "train/total_tokens_per_s": total_tps,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/elapsed_hours": elapsed_time / 3600,  # Convert to hours for consistency in wandb
                    "step": global_step,
                    "epoch": epoch
                })

            if profiler is not None:
                profiler.step()

        # End of epoch logging
        epoch_time = time.time() - epoch_start
        if rank == 0:
            elapsed_time = time.time() - training_start_time
            elapsed_str = format_elapsed_time(elapsed_time)
            epoch_time_str = format_elapsed_time(epoch_time)
            print(f"Epoch {epoch+1}/{args.num_epochs} finished in {epoch_time_str}. Total elapsed time: {elapsed_str}, Avg loss: {loss_meter.avg:.4f}")
            wandb.log({"epoch/loss": loss_meter.avg, "epoch/time_s": epoch_time, "epoch": epoch})

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

    # End of training
    if rank == 0:
        total_training_time = time.time() - training_start_time
        elapsed_str = format_elapsed_time(total_training_time)
        print(f"Training completed in {elapsed_str}")
    
    cleanup_ddp()


def main():
    args = get_args()
    print("Starting training process with args:", vars(args), flush=True)
    
    try:
        print("Entering train() function...", flush=True)
        train(args)
        print("Completed train() function normally", flush=True)
    except Exception as e:
        print(f"Error in training: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        # Make sure to cleanup even if there's an error
        cleanup_ddp()
        sys.exit(1)
    
    # Normal exit path should also call cleanup
    cleanup_ddp()
    print("Training process completed successfully", flush=True)

if __name__ == "__main__":
    main()