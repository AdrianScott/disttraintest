#!/usr/bin/env python
# Simple DDP communication test script
import os
import sys
import torch
import torch.distributed as dist
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DDP Communication Test")
    parser.add_argument("--timeout", type=int, default=60, 
                        help="Timeout for initialization in seconds")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Print basic environment info
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    n_procs_per_node = int(os.environ.get("N_PROCS_PER_NODE", "1"))
    nnodes = int(os.environ.get("NNODES", "1"))
    global_rank = node_rank * n_procs_per_node + local_rank
    
    # Get important network information
    hostname = os.popen('hostname').read().strip()
    ip_info = os.popen('hostname -I').read().strip()
    
    # Important environment variables that affect NCCL
    nccl_socket_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "not_set")
    nccl_debug = os.environ.get("NCCL_DEBUG", "not_set")
    nccl_async_error = os.environ.get("NCCL_ASYNC_ERROR_HANDLING", "not_set")
    nccl_ib_disable = os.environ.get("NCCL_IB_DISABLE", "not_set")
    nccl_p2p_disable = os.environ.get("NCCL_P2P_DISABLE", "not_set")
    
    # Print comprehensive environment info
    print(f"\n{'='*50}")
    print(f"DIAGNOSTIC INFO - Node {node_rank}/{nnodes}, Rank {global_rank}/{world_size}")
    print(f"{'='*50}")
    print(f"Hostname: {hostname}")
    print(f"IP addresses: {ip_info}")
    print(f"Process info: Node {node_rank}, Local rank {local_rank}, Global rank {global_rank}")
    print(f"CUDA device: {local_rank}, Available devices: {torch.cuda.device_count()}")
    
    # Print NCCL configuration
    print(f"\nNCCL Configuration:")
    print(f"  NCCL_SOCKET_IFNAME       = {nccl_socket_ifname}")
    print(f"  NCCL_DEBUG              = {nccl_debug}")
    print(f"  NCCL_ASYNC_ERROR_HANDLING = {nccl_async_error}")
    print(f"  NCCL_IB_DISABLE         = {nccl_ib_disable}")
    print(f"  NCCL_P2P_DISABLE        = {nccl_p2p_disable}")
    
    # Get master address info
    master_addr = os.environ.get("MASTER_ADDR", "not_set")
    master_port = os.environ.get("MASTER_PORT", "not_set")
    print(f"  MASTER_ADDR            = {master_addr}")
    print(f"  MASTER_PORT            = {master_port}")
    print(f"{'='*50}\n")
    
    # Check network connectivity to master (if not on master)
    if node_rank > 0 and master_addr != "not_set":
        print(f"Testing connectivity to master node ({master_addr})...")
        ping_result = os.system(f"ping -c 1 -W 2 {master_addr} > /dev/null 2>&1")
        if ping_result == 0:
            print(f"✅ Successfully pinged master node")
        else:
            print(f"❌ Failed to ping master node! This may cause DDP initialization to fail.")
    
    # Print interface information if running on rank 0 of each node
    if local_rank == 0:
        print(f"\nNetwork interfaces on node {node_rank} ({hostname}):")
        os.system("ip -br addr | grep -v 'lo'")
    
    try:
        # Set device before initialization
        torch.cuda.set_device(local_rank)
        
        print(f"[{node_rank}:{local_rank}] Initializing process group...")
        # Initialize with extended timeout for multi-node
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=torch.distributed.constants.default_pg_timeout * (args.timeout // 30)
        )
        
        # Get actual rank after initialization
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[{node_rank}:{local_rank}] Connected as rank {rank}/{world_size}")
        
        # Wait for all processes to reach this point
        if rank == 0:
            print("All processes have connected. Starting communication test...")
        
        # Simple tensor all-reduce test
        device = torch.device(f"cuda:{local_rank}")
        tensor = torch.ones(1, device=device) * (rank + 1)  # Different value for each rank
        print(f"[{node_rank}:{local_rank}] Before all-reduce: tensor = {tensor.item()}")
        
        # Perform all-reduce (sum operation)
        print(f"[{node_rank}:{local_rank}] Starting all-reduce...")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # Expected sum is 1+2+3+...+world_size = world_size*(world_size+1)/2
        expected = world_size * (world_size + 1) / 2
        print(f"[{node_rank}:{local_rank}] After all-reduce: tensor = {tensor.item()}")
        
        if abs(tensor.item() - expected) < 1e-3:
            print(f"[{node_rank}:{local_rank}] TEST PASSED! Got {tensor.item()}, expected {expected}")
        else:
            print(f"[{node_rank}:{local_rank}] TEST FAILED! Got {tensor.item()}, expected {expected}")
        
        # Node-to-node communication test
        print(f"\n[{node_rank}:{local_rank}] Running node-to-node communication tests...")
        
        # Create a tensor with unique value based on node_rank (not just rank)
        # This helps identify communication between specific nodes
        node_tensor = torch.ones(1, device=device) * (node_rank + 1)
        node_original = node_tensor.item()
        
        # All-reduce across nodes
        print(f"[{node_rank}:{local_rank}] Starting node-specific all-reduce...")
        node_start = time.time()
        dist.all_reduce(node_tensor)
        torch.cuda.synchronize()
        node_duration = time.time() - node_start
        
        # Expected result if all nodes participated
        expected_nodes = sum(range(1, nnodes + 1)) * n_procs_per_node
        print(f"[{node_rank}:{local_rank}] Node test result: {node_tensor.item():.1f} (expected ~{expected_nodes}), took {node_duration*1000:.2f}ms")
        
        # More complex test with larger tensor for bandwidth measurement
        print(f"\n[{node_rank}:{local_rank}] Testing bandwidth with larger tensor...")
        large_tensor = torch.ones(1024*1024, device=device) * (rank + 1)  # 4MB tensor
        
        # Warm-up
        for _ in range(5):
            dist.all_reduce(large_tensor)
            torch.cuda.synchronize()
        
        # Time multiple all-reduces
        print(f"[{node_rank}:{local_rank}] Starting bandwidth test across {nnodes} nodes...")
        torch.cuda.synchronize()
        start = time.time()
        iterations = 10
        for i in range(iterations):
            dist.all_reduce(large_tensor)
            torch.cuda.synchronize()
            if i == 0 and local_rank == 0:
                # Print first result to confirm operation is working
                print(f"[{node_rank}:{local_rank}] First bandwidth test iteration completed")
        end = time.time()
        
        elapsed = end - start
        size_mb = large_tensor.nelement() * large_tensor.element_size() * iterations / (1024*1024)
        bandwidth = size_mb / elapsed
        
        print(f"[{node_rank}:{local_rank}] Bandwidth test: {bandwidth:.2f} MB/s")
        
        # Test point-to-point communication if multiple nodes
        if nnodes > 1 and rank == 0:
            target_rank = n_procs_per_node  # First process on second node
            p2p_tensor = torch.tensor([42.0], device=device)
            print(f"[{node_rank}:{local_rank}] Testing direct P2P send/recv with rank {target_rank}...")
            
            try:
                if rank == 0:
                    dist.send(p2p_tensor, dst=target_rank)
                    print(f"[{node_rank}:{local_rank}] P2P send to rank {target_rank} completed")
                elif rank == target_rank:
                    dist.recv(p2p_tensor, src=0)
                    print(f"[{node_rank}:{local_rank}] P2P recv from rank 0 completed, got {p2p_tensor.item()}")
            except Exception as e:
                print(f"[{node_rank}:{local_rank}] P2P test failed: {str(e)}")
        
        # Final barrier to ensure all processes complete
        print(f"[{node_rank}:{local_rank}] Waiting at final barrier...")
        barrier_start = time.time()
        dist.barrier()
        barrier_time = time.time() - barrier_start
        
        if rank == 0:
            print(f"\nFinal barrier completed in {barrier_time:.3f}s")
            print("\nSUCCESS: All tests completed. Multi-node communication is working correctly!\n")
        elif rank % n_procs_per_node == 0:
            # Also print from first process of each node
            print(f"\n[Node {node_rank}] All tests completed successfully. Barrier time: {barrier_time:.3f}s\n")
            
        # Final node-specific synchronization check
        if local_rank == 0:
            # First process on each node reports readiness
            print(f"Node {node_rank} ({hostname}) communication tests PASSED")
        
    except Exception as e:
        print(f"[{node_rank}:{local_rank}] ERROR: {str(e)}")
        if "NCCL" in str(e) and "eth1" in nccl_socket_ifname:
            print("Recommendation: There might be an issue with the eth1 interface.")
            print("Try changing NCCL_SOCKET_IFNAME to another interface or check network connectivity.")
        sys.exit(1)
        
    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
