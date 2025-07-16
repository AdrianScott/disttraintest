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

# The Python script to execute
# We use the module (-m) flag to ensure correct imports
SCRIPT="src.train_ddp"

# --- Validation ---
if [ ! -f "${SCRIPT//.//}.py" ]; then
    echo "Error: Training script ${SCRIPT//.//}.py not found."
    exit 1
fi

# --- Execution ---
echo "Starting DDP training..."

# Set the network interface for NCCL. This is crucial for Docker/container environments.
# We are choosing 'podnet1' based on the output of 'ip addr'.
export NCCL_SOCKET_IFNAME=podnet1



torchrun --nproc_per_node=$N_PROCS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m src.train_ddp \
    --d_model=512 \
    --n_layers=4 \
    --n_heads=8 \
    --batch_size=16 \
    --learning_rate=3e-4 \
    --num_epochs=150 \
    --seq_len=256

echo "Training script finished."
