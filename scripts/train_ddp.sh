#!/bin/bash
# This script launches a distributed training job using torchrun.
# It supports both single-node and multi-node training.

# --- Configuration ---

N_PROCS_PER_NODE=${N_PROCS_PER_NODE:-1}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# Optional perf knobs
NUM_WORKERS=${NUM_WORKERS:-2}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
LOG_EVERY=${LOG_EVERY:-20}
BUCKET_CAP_MB=${BUCKET_CAP_MB:-25}
NUM_EPOCHS=${NUM_EPOCHS:-150}

SCRIPT="src.train_ddp"

if [ ! -f "${SCRIPT//.//}.py" ]; then
    echo "Error: Training script ${SCRIPT//.//}.py not found."
    exit 1
fi

echo "Starting DDP training with $N_PROCS_PER_NODE processes per node on $NNODES nodes..."
echo "Master node: $MASTER_ADDR:$MASTER_PORT, Current node rank: $NODE_RANK"

export N_PROCS_PER_NODE

# Logs
LOG_DIR="./logs"
mkdir -p $LOG_DIR
RUN_ID=$(date +"%Y%m%d_%H%M%S")

# NCCL / Torch env (minimize overhead, only set debug when troubleshooting)
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_FILE="${LOG_DIR}/nccl_${RUN_ID}_node${NODE_RANK}.log"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Socket tuning
export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-2}
export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-2}

# IB / P2P flags (disable only if hangs)
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}

# Buffer size
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-8388608}

# Pick interface if not set
if [[ -z "$NCCL_SOCKET_IFNAME" ]]; then
    PRIMARY_IFACE=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)
    export NCCL_SOCKET_IFNAME=${PRIMARY_IFACE:-eth0}
fi
echo "NCCL interface: $NCCL_SOCKET_IFNAME"

echo "Available network interfaces:"
ip -br addr | grep -v 'lo'

torchrun --nproc_per_node=$N_PROCS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m src.train_ddp \
    --d_model=512 \
    --n_layers=4 \
    --n_heads=8 \
    --batch_size=32 \
    --learning_rate=3e-4 \
    --num_epochs=$NUM_EPOCHS \
    --n_processes_per_node=$N_PROCS_PER_NODE \
    --seq_len=256 \
    --num_workers=$NUM_WORKERS \
    --gradient_accumulation_steps=$GRAD_ACC_STEPS \
    --log_every=$LOG_EVERY \
    --bucket_cap_mb=$BUCKET_CAP_MB

echo "Training script finished."