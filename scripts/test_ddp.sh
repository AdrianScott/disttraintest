#!/bin/bash
# This script runs a quick DDP communication test to verify multi-node setup

# --- Configuration ---
# Set the number of GPUs to use per node
N_PROCS_PER_NODE=${N_PROCS_PER_NODE:-1}

# Multi-Node Config
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# --- Network Configuration ---
# Essential debugging info
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Create log directory for NCCL logs
mkdir -p ./logs

# Get primary network interface (try to detect automatically)
get_network_interface() {
    # Try to detect a suitable network interface
    # First check for common interfaces
    for iface in eth0 eno1 enp0s3 enp39s0 ens3; do
        if ip link show $iface &>/dev/null; then
            echo $iface
            return 0
        fi
    done
    # If not found, use the first non-loopback interface
    primary_iface=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)
    if [[ ! -z "$primary_iface" ]]; then
        echo $primary_iface
        return 0
    fi
    # Fallback
    echo "lo"
    return 1
}

# Network interface selection based on environment
if [[ "$NNODES" -gt 1 ]]; then
    echo "Multi-node test: $NNODES nodes"
    # Use existing env var if set, otherwise default to eth1 for multi-node
    if [[ -z "$NCCL_SOCKET_IFNAME" ]]; then
        export NCCL_SOCKET_IFNAME=eth1
    fi
    echo "Using network interface: NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME for multi-node"
    export NCCL_DEBUG_FILE="./logs/nccl_test_node${NODE_RANK}.log"
else
    echo "Single-node test with $N_PROCS_PER_NODE processes"
    # Use existing env var if set, otherwise auto-detect
    if [[ -z "$NCCL_SOCKET_IFNAME" ]]; then
        interface=$(get_network_interface)
        export NCCL_SOCKET_IFNAME=$interface
    fi
    echo "Using network interface: NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME for single-node"
    export NCCL_DEBUG_FILE="./logs/nccl_test_local.log"
fi

# Set NCCL debug settings (same for both single and multi-node)
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

echo "Starting DDP communication test..."
echo "Master node: $MASTER_ADDR:$MASTER_PORT, Current node rank: $NODE_RANK"

torchrun --nproc_per_node=$N_PROCS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m src.test_ddp \
    --timeout=60

echo "DDP communication test completed."
