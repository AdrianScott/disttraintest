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
echo "Starting DDP training with $N_PROCS_PER_NODE processes per node on $NNODES nodes..."
echo "Master node: $MASTER_ADDR:$MASTER_PORT, Current node rank: $NODE_RANK"

# Pass N_PROCS_PER_NODE to the training script for node count calculation
export N_PROCS_PER_NODE

# --- Network Configuration for Multi-node ---
# Create log directory for NCCL logs
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# Generate a unique run ID based on timestamp
RUN_ID=$(date +"%Y%m%d_%H%M%S")

# Essential debugging info with log redirection
export NCCL_DEBUG=INFO              # More verbose logging if something hangs
export NCCL_DEBUG_FILE="${LOG_DIR}/nccl_${RUN_ID}_node${NODE_RANK}.log"  # Save NCCL logs to file
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Better error reporting (new variable name)

# Additional network reliability settings
export NCCL_IB_TIMEOUT=30          # Longer timeout (default is 14)
export NCCL_IB_RETRY_CNT=10        # More retry attempts
export NCCL_SOCKET_NTHREADS=4      # Use more threads for socket comm
export NCCL_BUFFSIZE=4194304       # Smaller buffer size (4MB)

# Disable P2P operations which are causing hangs
export NCCL_P2P_DISABLE=1          # Disable direct P2P operations
export NCCL_SHM_DISABLE=0          # Keep shared memory enabled
export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME  # Ensure interface is consistent

# Display network interfaces - useful for debugging
echo "Available network interfaces:"
ip -br addr | grep -v 'lo'

# Network interface selection based on environment
if [[ "$NNODES" -gt 1 ]]; then
    echo "Multi-node training detected: $NNODES nodes"
    echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"
    
    # Use existing env var if set, otherwise default to eth1 for multi-node
    if [[ -z "$NCCL_SOCKET_IFNAME" ]]; then
        export NCCL_SOCKET_IFNAME=eth1
    fi
    echo "NCCL using network interface: $NCCL_SOCKET_IFNAME"
    
    # Test network connectivity between nodes if this isn't the master node
    if [[ "$NODE_RANK" -gt 0 ]]; then
        echo "Testing connectivity to master node ($MASTER_ADDR)..."
        if ping -c 1 -W 2 $MASTER_ADDR > /dev/null; then
            echo "✓ Successfully connected to master node"
        else
            echo "✗ WARNING: Cannot ping master node! This may cause DDP initialization to fail."
        fi
    fi
else
    echo "Single-node training with $N_PROCS_PER_NODE processes"
    
    # Use existing env var if set, otherwise default to a reasonable interface for local training
    if [[ -z "$NCCL_SOCKET_IFNAME" ]]; then
        # Try to find a suitable interface
        PRIMARY_IFACE=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)
        if [[ ! -z "$PRIMARY_IFACE" ]]; then
            export NCCL_SOCKET_IFNAME=$PRIMARY_IFACE
        fi
    fi
    echo "NCCL using network interface: $NCCL_SOCKET_IFNAME"
fi


torchrun --nproc_per_node=$N_PROCS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m src.train_ddp \
    --d_model=512 \
    --n_layers=4 \
    --n_heads=8 \
    --batch_size=16 \
    --learning_rate=3e-4 \
    --num_epochs=150 \
    --n_processes_per_node=$N_PROCS_PER_NODE \
    --seq_len=256

echo "Training script finished."
