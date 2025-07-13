# TinyGPT: A Distributed Transformer from Scratch

This project is a complete, end-to-end implementation of a Transformer-based language model, similar to GPT, built from scratch using PyTorch. It includes a full pipeline for data preparation, single-GPU training, multi-GPU distributed training with DDP, and advanced text generation with sampling.

## Core Features

- **Custom Transformer Model:** A from-scratch implementation of the decoder-only Transformer architecture, including multi-head self-attention and sinusoidal positional encodings.
- **Data Preparation:** A script to download the WikiText-2 dataset and train a custom Byte-Pair Encoding (BPE) tokenizer.
- **Distributed Training (DDP):** A robust distributed training script (`src/train_ddp.py`) using PyTorch's `DistributedDataParallel` (DDP) and `torchrun` for both single-node and multi-node scaling.
- **Advanced Text Generation:** A flexible generation script (`src/generate.py`) that supports both greedy decoding and advanced sampling methods like **Top-K** and **Nucleus (Top-P)** sampling.

---

## Local Setup and Usage

### 1. Prerequisites

- Python 3.9+
- Conda (recommended for environment management)
- An NVIDIA GPU with CUDA installed

### 2. Installation

Clone the repository and set up the Conda environment.

```bash
git clone <your-repo-url>
cd disttraintest

conda create -n disttrain python=3.10 -y
conda activate disttrain

pip install -r requirements.txt
```

### 3. Prepare Data

Download the dataset and train the tokenizer. This only needs to be done once.

```bash
python -m data.prepare_data
```

---

## Showcase 1: Single-Node, Multi-GPU Training

This project scales seamlessly to use all available GPUs on a single machine. The following instructions detail how to run this on a 4-GPU instance on a cloud platform like [RunPod](https://runpod.io) or [Vast.ai](https://vast.ai).

1.  **Launch a cloud instance** with 4 GPUs and a standard PyTorch image.
2.  **Follow the local installation steps** to clone the repo and set up the Conda environment.
3.  **Prepare the data** by running `python -m data.prepare_data`.
4.  **Edit `scripts/train_ddp.sh`** and ensure `N_PROCS_PER_NODE` is set to `4`.
5.  **Launch the training job:**

```bash
bash scripts/train_ddp.sh
```

`torchrun` will automatically use the default settings for a single-node run, launching and managing 4 processes on the machine.

---

## Showcase 2: Multi-Node, Multi-GPU Distributed Training

The launch script is also pre-configured to scale across multiple machines (nodes) for large-scale training. The following example demonstrates how to launch a job on **2 nodes, each with 4 GPUs** (for a total of 8 GPUs).

### 1. Environment Setup

- Launch two separate 4-GPU instances. These are your **nodes**.
- Designate one as the **master node** (this will be `NODE_RANK=0`). Note its private IP address (e.g., `10.0.0.5`).
- On **both** machines, perform the same setup: clone the repo, install dependencies, and prepare the data.

### 2. Launch Command

The `scripts/train_ddp.sh` script reads environment variables to configure the multi-node environment. You will run the script on **both** machines, but with different environment variables set for each.

#### On the Master Node (Node 0):

Open a terminal and run the following. We are telling it that there are 2 nodes in total (`NNODES=2`), this is node 0 (`NODE_RANK=0`), and its IP is the master address.

```bash
export NNODES=2
export NODE_RANK=0
export MASTER_ADDR=10.0.0.5 # Use the master node's actual private IP
bash scripts/train_ddp.sh
```

#### On the Second Node (Node 1):

Open a terminal on the second machine and run the following. We tell it that it is node 1 (`NODE_RANK=1`) and that it needs to communicate with the master node at its IP address.

```bash
export NNODES=2
export NODE_RANK=1
export MASTER_ADDR=10.0.0.5 # Use the master node's actual private IP
bash scripts/train_ddp.sh
```

### 3. What Happens Next

- `torchrun` on both machines will coordinate via the master address and port.
- A total of 8 processes (4 per node) will be launched.
- The dataset will be sharded across all 8 processes, and training will begin.
- The final checkpoint will be saved only on the master node (rank 0) to prevent conflicts.

This setup demonstrates a true, production-grade distributed training workflow.
