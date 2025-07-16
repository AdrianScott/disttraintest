#!/bin/bash
# This script launches a distributed training job using torchrun.
# It supports both single-node and multi-node training.

# --- Configuration ---

# -- Single-Node Config --
# Set the number of GPUs to use per node.
# For a single-machine run, this is the total number of GPUs you want to use.
N_PROCS_PER_NODE=${N_PROCS_PER_NODE:-1}

# -- Multi-Node Config --
# These variables are read from the environment, with defaults for single-node.
# Total number of nodes (machines) in the cluster.
# For a single-node run, this will be 1.
NNODES=${NNODES:-1}
# The rank of the current node (0, 1, 2, ...). Default is 0 for single-node.
NODE_RANK=${NODE_RANK:-0}
# The IP address of the master node (node with rank 0). Default is localhost.
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# A free port on the master node for communication.
MASTER_PORT=${MASTER_PORT:-29500}

# The Python module to execute
SCRIPT_MODULE="src.train_ddp"

# --- DDP/NCCL Debugging ---
# For hangs or other network-related issues, uncommenting these can be helpful.
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Set the network interface for NCCL. This is crucial for multi-node training
# in containerized environments like Runpod or AWS.
#
# How to find the right interface name:
# 1. Run `ip addr` or `ifconfig` on your pod/machine.
# 2. Look for the interface with your main private IP address (e.g., starts with 10.x.x.x).
#    Common names are `eth0`, `ens5`, `eno1`, or `ib0` for InfiniBand.
#
# The following command attempts to find it automatically, but may not work in all environments.
# If training hangs, manually set this to the correct interface name.
IFNAME=$(ip -o -4 route show to default | awk '{print $5}')
export NCCL_SOCKET_IFNAME=${IFNAME:-eth0}
echo "Using network interface: $NCCL_SOCKET_IFNAME for NCCL."

# In some cloud environments, direct GPU-to-GPU communication (P2P) over the network
# is not well-supported and can cause hangs. Disabling it can sometimes resolve issues.
# export NCCL_P2P_DISABLE=1

# --- Validation ---
if [ ! -f "${SCRIPT_MODULE//.//}.py" ]; then
    echo "Error: Training script ${SCRIPT_MODULE//.//}.py not found."
    exit 1
fi

# --- Execution ---
echo "Starting DDP training..."
echo "NNODES: $NNODES, NODE_RANK: $NODE_RANK, MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT, N_PROCS_PER_NODE: $N_PROCS_PER_NODE"


torchrun --nproc_per_node=$N_PROCS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m ${SCRIPT_MODULE} \
    --d_model=512 \
    --n_layers=4 \
    --n_heads=8 \
    --batch_size=16 \
    --learning_rate=3e-4 \
    --num_epochs=150 \
    --seq_len=256

echo "Training script finished."