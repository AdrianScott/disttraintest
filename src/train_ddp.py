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
import sys
import time
import datetime
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of steps to accumulate gradients before optimizer step")
    # W&B
    parser.add_argument("--wandb_project", type=str, default="tiny-transformer-from-scratch-ddp", help="WandB project name")
    # DDP specific
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
def setup_ddp(rank, world_size, args):
    """Setup DDP process group with extended timeout for multi-node."""
    try:
        local_rank = rank % args.n_processes_per_node
        node_rank = rank // args.n_processes_per_node

        print(f"Setting up DDP: Global rank {rank}, Node {node_rank}, Local rank {local_rank}, World size {world_size}")

        # Log NCCL configuration from environment
        nccl_socket_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "not set")
        nccl_debug = os.environ.get("NCCL_DEBUG", "not set")
        print(f"Rank {rank}: NCCL config: SOCKET_IFNAME={nccl_socket_ifname}, DEBUG={nccl_debug}")

        # Set device before initializing process group
        torch.cuda.set_device(local_rank)
        print(f"Rank {rank}: Set CUDA device to: {local_rank} (GPU {torch.cuda.current_device()})")

        # Initialize with extended timeout for multi-node
        print(f"Rank {rank}: Initializing process group with timeout={torch.distributed.constants.default_pg_timeout * 3}s")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            # Extend default timeout for multi-node
            timeout=torch.distributed.constants.default_pg_timeout * 3
        )

        # Get master address from environment
        master_addr = os.environ.get("MASTER_ADDR", "unknown")
        master_port = os.environ.get("MASTER_PORT", "unknown")
        print(f"Rank {rank}: Connected to process group. Master: {master_addr}:{master_port}")

        return rank, world_size
    except Exception as e:
        print(f"Failed to initialize process group: {str(e)}")
        sys.exit(1)

def test_communication(local_rank, world_size):
    """Perform a simple all-reduce operation to test multi-node communication."""
    rank = dist.get_rank()
    local_world_size = torch.cuda.device_count()
    node_rank = rank // local_world_size

    print(f"Rank {rank}: Starting communication test (Node {node_rank}, Local rank {local_rank})")

    # First test: basic all-reduce with small tensor
    print(f"Rank {rank}: Testing basic all-reduce...")
    tensor = torch.ones(1, device=f"cuda:{local_rank}") * (rank + 1)  # Different value per rank
    original_value = tensor.item()

    # Time the all-reduce operation
    start_time = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    duration = time.time() - start_time

    # Expected sum is sum of 1+2+3+...+world_size = world_size*(world_size+1)/2
    expected = world_size * (world_size + 1) / 2
    passed = abs(tensor.item() - expected) < 1e-3

    print(f"Rank {rank}: Communication test {'passed' if passed else 'FAILED'} - "
          f"sent {original_value}, got {tensor.item():.1f}, expected {expected:.1f}, "
          f"took {duration*1000:.2f}ms")

    # Use barrier to ensure all processes complete tests
    dist.barrier()

    if rank == 0:
        print(f"All {world_size} processes completed communication tests")

    return passed

def cleanup_ddp():
    """ Cleans up the distributed process group. """
    dist.destroy_process_group()

# --- Main Training Logic ---
def train():
    """ Main function to run the DDP training loop. """
    args = get_args()

    # Get basic process information
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Set up distributed process group with enhanced logging
    rank, world_size = setup_ddp(rank, world_size, args)

    # Run communication test to verify multi-node setup
    print(f"Rank {rank}: Beginning communication test...")
    if test_communication(local_rank, world_size):
        print(f"Rank {rank}: Communication test passed!")
    else:
        print(f"Rank {rank}: Communication test failed! Exiting.")
        cleanup_ddp()
        sys.exit(1)

    # 1. Initialization (only on main process)
    if rank == 0:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        # Combine args with other relevant info for wandb config
        config = vars(args)
        config["vocab_size"] = VOCAB_SIZE
        config["batch_size_total"] = args.batch_size * world_size
        config["world_size"] = world_size
        # Determine number of nodes based on world_size and processes per node
        n_nodes = world_size // int(os.environ.get("N_PROCS_PER_NODE", "1"))
        wandb.init(project=args.wandb_project, config=config)
        print(f"DDP training started on {world_size} GPUs across {n_nodes} node(s).")
        print(f"Run configuration: {config}")

    # 2. Load Tokenizer
    print(f"Rank {rank}: Loading tokenizer...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    # Remove barrier here - not needed for tokenizer loading

    # 3. Load and Prepare Data
    print(f"Rank {rank}: Loading and tokenizing dataset...")
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    all_text = " ".join([text for text in wikitext["text"] if text.strip()])
    token_ids = tokenizer.encode(all_text).ids

    if rank == 0:
        print(f"Tokenized dataset with {len(token_ids)} tokens")
    # Remove barrier here - not needed for data loading

    train_dataset = TextDataset(token_ids, args.seq_len)
    if rank % 8 == 0:  # Print from a subset of ranks
        print(f"Rank {rank}: Dataset has {len(train_dataset)} samples")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,          # Use 0 for multi-node DDP to avoid multiprocessing issues
        pin_memory=True,
        persistent_workers=False # Disable since num_workers=0
    )
    print(f"Rank {rank}: DataLoader initialized with {len(train_loader)} batches")

    # 4. Model, Optimizer, Loss, Scheduler
    print(f"Rank {rank}: Initializing model...")
    model = TinyGPT(VOCAB_SIZE, args.d_model, args.n_layers, args.n_heads, args.max_len).to(local_rank)
    # Remove find_unused_parameters as it's causing overhead and warnings
    # The model doesn't actually have unused parameters in the forward pass
    model = DDP(model, device_ids=[local_rank])

    # Wait for all processes to finish initialization with timeout
    try:
        dist.barrier()
        if rank == 0:
            print("Initialization barrier completed successfully")
    except Exception as e:
        print(f"Rank {rank}: Initialization barrier timed out: {str(e)}")
        print(f"Rank {rank}: Continuing despite barrier timeout...")
        # Continue even if barrier fails - this is more robust
    print(f"Rank {rank}: Model initialized and ready for training")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.num_epochs)

    # 5. Training Loop
    # Calculate node information once for consistent logging
    local_world_size = torch.cuda.device_count()
    node_rank = rank // local_world_size
    node_local_rank = rank % local_world_size
    
    try:
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0.0
            tokens_processed = 0
            start_time = time.time()

            print(f"Rank {rank}: Starting epoch {epoch+1}/{args.num_epochs}")
            train_sampler.set_epoch(epoch)  # Important for proper shuffling in multi-node

            # Zero gradients at the beginning of epoch
            optimizer.zero_grad()
            
            for i, (inputs, targets) in enumerate(train_loader):
                print(f"Rank {rank}: Fetched batch {i}")
                inputs = inputs.to(local_rank)
                targets = targets.to(local_rank)

                # Log for debugging
                if i == 0 and (rank == 0 or rank % 8 == 0):
                    print(f"Rank {rank}: Starting batch with shape {inputs.shape}")

                # Forward pass
                try:
                    logits, _ = model(inputs)  # model returns (logits, kv_cache)
                except Exception as e:
                    print(f"Rank {rank}: Forward pass failed with error: {str(e)}")
                    cleanup_ddp()
                    sys.exit(1)

                # Calculate loss
                loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
                
                # Scale loss by accumulation steps to maintain correct gradients
                loss = loss / gradient_accumulation_steps

                # Backward pass
                try:
                    loss.backward()
                except Exception as e:
                    print(f"Rank {rank}: Backward pass failed with error: {str(e)}")
                    cleanup_ddp()
                    sys.exit(1)

                # Track loss (unscaled for logging)
                epoch_loss += loss.item() * gradient_accumulation_steps

                # Count tokens processed
                tokens_processed += inputs.numel()
                
                # Only step optimizer every N accumulation steps
                if (i + 1) % gradient_accumulation_steps == 0:
                    # Log when we're doing communication
                    if rank == 0:
                        print(f"Rank {rank}: Performing optimizer step after {gradient_accumulation_steps} batches of gradient accumulation")
                    
                    # Optimize
                    optimizer.step()
                    scheduler.step()
                    
                    # Clear gradients after stepping
                    optimizer.zero_grad()

                # Step timing
                elapsed_time = time.time() - start_time
                step_time = elapsed_time / (i + 1)

                # Print batch stats
                if i % 10 == 0 and rank == 0:
                    print(f"Rank {rank} (Node {node_rank}, Local {local_rank}): Epoch {epoch+1}/{args.num_epochs}, Step {i}, Loss: {loss.item():.4f}, Step time: {step_time:.2f}s")
                    gpu_mem_alloc = torch.cuda.max_memory_allocated(device=local_rank) / 1024**3
                    gpu_mem_res = torch.cuda.max_memory_reserved(device=local_rank) / 1024**3
                    print(f"GPU Memory: {gpu_mem_alloc:.2f}GB allocated, {gpu_mem_res:.2f}GB reserved")
                    if os.environ.get("NCCL_DEBUG", "") == "INFO":
                        print(f"Check NCCL INFO logs for communication details")
                    
                    # Calculate per-GPU throughput (this GPU only)
                    per_gpu_throughput = tokens_processed / elapsed_time
                    
                    # Calculate total system throughput (all GPUs across all nodes)
                    total_system_throughput = per_gpu_throughput * world_size
                    
                    wandb.log({
                        "train_loss": loss.item(), 
                        "per_gpu_throughput": per_gpu_throughput,
                        "total_system_throughput": total_system_throughput
                    })
                    print(f"Epoch [{epoch+1}/{args.num_epochs}], Step {i}, Loss: {loss.item():.4f}")
                    print(f"Node 0 GPU 0 Throughput: {per_gpu_throughput:.2f} tokens/sec")
                    
                    # Calculate number of nodes from environment or args
                    try:
                        num_nodes = int(os.environ.get("NNODES", "1"))  # Default to 1 if not set
                    except ValueError:
                        num_nodes = 1
                    
                    # Add gradient accumulation info to logging
                    effective_batch_size = args.batch_size * gradient_accumulation_steps * world_size
                    print(f"Total System Throughput: {total_system_throughput:.2f} tokens/sec ({world_size} GPUs across {num_nodes} node{'s' if num_nodes > 1 else ''})")
                    print(f"Gradient Accumulation: {gradient_accumulation_steps} steps (effective batch size: {effective_batch_size})")



            # Collect timing stats at end of epoch
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - start_time

            # Log per-node statistics
            # Aggregate stats from different nodes - log from first process on each node
            if node_local_rank == 0:
                print(f"Node {node_rank}: Completed epoch {epoch+1} in {epoch_duration:.2f}s")

            # Wait for all processes to finish epoch with timeout protection
            barrier_start = time.time()
            try:
                dist.barrier()
                barrier_time = time.time() - barrier_start
                if rank == 0:
                    print(f"Synchronization barrier completed in {barrier_time:.3f}s")
            except Exception as e:
                barrier_time = time.time() - barrier_start
                print(f"Rank {rank}: Warning: Barrier timed out after {barrier_time:.3f}s: {str(e)}")
                print(f"Rank {rank}: Continuing training despite barrier timeout...")

            # Log barrier time from rank 0 (useful to detect stragglers)
            if rank == 0:
                print(f"Synchronization barrier took {barrier_time:.3f}s")
                avg_loss = epoch_loss / len(train_loader)
                wandb.log({"epoch_loss": avg_loss, "epoch": epoch})
                print(f"Epoch [{epoch+1}/{args.num_epochs}] finished. Average Loss: {avg_loss:.4f}")

                # Save checkpoint periodically
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pt"
                    torch.save(model.module.state_dict(), checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        print(f"Rank {rank}: Exception in training loop: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_ddp()
        sys.exit(1)

    # 6. Final Cleanup
    # Specify device ID to avoid warnings and add timeout
    try:
        dist.barrier()  # Ensure all processes reach this point
        if rank == 0:
            print("Final synchronization barrier completed successfully")
    except Exception as e:
        print(f"Rank {rank}: Final barrier timed out or failed: {str(e)}")
        # Continue cleanup anyway

    if rank == 0:
        final_model_path = CHECKPOINT_DIR / "model_final.pt"
        torch.save(model.module.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    train()
