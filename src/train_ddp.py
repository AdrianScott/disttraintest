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
    
    # Second test: larger tensor to test bandwidth if first test passes
    if passed and rank % 8 == 0:  # Only test from a subset of ranks
        print(f"Rank {rank}: Testing bandwidth with larger tensor...")
        large_tensor = torch.ones(1024*1024, device=f"cuda:{local_rank}")  # 4MB tensor
        
        # Perform a timed all-reduce
        torch.cuda.synchronize()
        start = time.time()
        dist.all_reduce(large_tensor)
        torch.cuda.synchronize()
        bw_duration = time.time() - start
        
        size_mb = large_tensor.nelement() * large_tensor.element_size() / (1024*1024)
        bandwidth = size_mb / bw_duration
        print(f"Rank {rank}: Bandwidth test: {bandwidth:.2f} MB/s")
    
    # Use barrier to ensure all processes complete tests
    # Specify device ID to avoid warnings
    dist.barrier(device_ids=[local_rank])
    
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
    # Specify device ID to avoid warnings
    dist.barrier(device_ids=[local_rank])  # Ensure all processes have loaded the tokenizer

    # 3. Load and Prepare Data
    print(f"Rank {rank}: Loading and tokenizing dataset...")
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    all_text = " ".join([text for text in wikitext["text"] if text.strip()])
    token_ids = tokenizer.encode(all_text).ids
    
    if rank == 0:
        print(f"Tokenized dataset with {len(token_ids)} tokens")
    # Specify device ID to avoid warnings
    dist.barrier(device_ids=[local_rank])  # Synchronize after data loading

    train_dataset = TextDataset(token_ids, args.seq_len)
    if rank % 8 == 0:  # Print from a subset of ranks
        print(f"Rank {rank}: Dataset has {len(train_dataset)} samples")
        
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,          # Speed up host-to-GPU pipeline
        pin_memory=True,
        persistent_workers=True # Avoid worker respawn each epoch
    )
    print(f"Rank {rank}: DataLoader initialized with {len(train_loader)} batches")

    # 4. Model, Optimizer, Loss, Scheduler
    print(f"Rank {rank}: Initializing model...")
    model = TinyGPT(VOCAB_SIZE, args.d_model, args.n_layers, args.n_heads, args.max_len).to(local_rank)
    # We set find_unused_parameters=True because our model's forward pass
    # has logic for a KV cache that is not used during training. This prevents
    # DDP from hanging when it can't find gradients for those unused parameters.
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Wait for all processes to finish initialization
    dist.barrier()
    print(f"Rank {rank}: Model initialized and ready for training")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.num_epochs)

    # 5. Training Loop
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling in multi-node
        epoch_loss = 0
        start_time = time.time()
        
        if rank == 0:
            print(f"\n===== Starting epoch {epoch+1}/{args.num_epochs} =====")

        for i, (inputs, targets) in enumerate(train_loader):
            # Add step timing for debugging
            step_start = time.time()
            
            inputs = inputs.to(local_rank)
            targets = targets.to(local_rank)

            # Log for debugging
            if i == 0 and (rank == 0 or rank % 8 == 0):
                print(f"Rank {rank}: Starting batch with shape {inputs.shape}")
                
            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            try:
                logits, _ = model(inputs)  # model returns (logits, kv_cache)
            except Exception as e:
                print(f"Rank {rank}: Forward pass failed with error: {str(e)}")
                cleanup_ddp()
                sys.exit(1)
                
            # Calculate loss
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            
            # Backward pass
            try:
                loss.backward()
            except Exception as e:
                print(f"Rank {rank}: Backward pass failed with error: {str(e)}")
                cleanup_ddp()
                sys.exit(1)
                
            # Optimize
            optimizer.step()
            scheduler.step()

            # Track loss
            epoch_loss += loss.item()
            
            # Periodically log progress with detailed timing breakdown
            if i % 100 == 0 and (rank == 0 or rank % 8 == 0):
                step_time = time.time() - step_start
                local_world_size = torch.cuda.device_count()
                node_rank = rank // local_world_size
                
                print(f"Rank {rank} (Node {node_rank}, Local {rank % local_world_size}): "
                      f"Epoch {epoch+1}/{args.num_epochs}, Step {i}, Loss: {loss.item():.4f}, "
                      f"Step time: {step_time:.3f}s")
                
                # Every 100 steps, perform a quick communication check from a subset of ranks
                if i % 100 == 0 and (rank == 0 or rank == world_size // 2):
                    print(f"Rank {rank}: Performing quick comm check during training...")
                    comm_start = time.time()
                    test_tensor = torch.ones(1, device=f"cuda:{local_rank}") * rank
                    dist.all_reduce(test_tensor)
                    torch.cuda.synchronize()
                    comm_time = time.time() - comm_start
                    print(f"Rank {rank}: Comm check completed in {comm_time*1000:.2f}ms, value: {test_tensor.item()}")

            # More comprehensive logging from rank 0
            if i % 100 == 0 and rank == 0:
                # Calculate throughput
                end_time = time.time()
                elapsed_time = end_time - start_time
                tokens_processed = (i + 1) * args.batch_size * args.seq_len * world_size
                
                # Print GPU memory stats
                gpu_mem_alloc = torch.cuda.max_memory_allocated(device=local_rank) / 1024**3
                gpu_mem_res = torch.cuda.max_memory_reserved(device=local_rank) / 1024**3
                print(f"GPU Memory: {gpu_mem_alloc:.2f}GB allocated, {gpu_mem_res:.2f}GB reserved")
                
                # Print NCCL stats if available
                if os.environ.get("NCCL_DEBUG", "") == "INFO":
                    print(f"Check NCCL INFO logs for communication details")
                throughput = tokens_processed / elapsed_time
                wandb.log({"train_loss": loss.item(), "throughput_tokens_per_sec": throughput})
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step {i}, Loss: {loss.item():.4f}, Throughput: {throughput:.2f} tokens/sec")

        # Collect timing stats at end of epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - start_time
        
        # Log per-node statistics
        local_world_size = torch.cuda.device_count()
        node_rank = rank // local_world_size
        node_local_rank = rank % local_world_size
        
        # Aggregate stats from different nodes - log from first process on each node
        if node_local_rank == 0:
            print(f"Node {node_rank}: Completed epoch {epoch+1} in {epoch_duration:.2f}s")
        
        # Wait for all processes to finish epoch
        barrier_start = time.time()
        # Specify device ID to avoid warnings
        dist.barrier(device_ids=[local_rank])
        barrier_time = time.time() - barrier_start
        
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

    # 6. Final Cleanup
    # Specify device ID to avoid warnings
    dist.barrier(device_ids=[local_rank])  # Ensure all processes reach this point
    
    if rank == 0:
        final_model_path = CHECKPOINT_DIR / "model_final.pt"
        torch.save(model.module.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    train()
