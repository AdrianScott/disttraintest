# TinyGPT: A Distributed Transformer from Scratch

This project is a complete, end-to-end implementation of a Transformer-based language model, similar to GPT, built from scratch using PyTorch. It includes a full pipeline for data preparation, single-GPU training, multi-GPU distributed training with DDP, and advanced text generation with sampling.

This repository is prepared to work with MLOps best practices, including containerization with Docker and a clear path towards production deployment with Kubernetes.

## Core Features

- **Custom Transformer Model:** A from-scratch implementation of the decoder-only Transformer architecture with KV Caching for efficient generation.
- **Containerized Environment:** A `Dockerfile` is provided for a fully reproducible and portable development and execution environment.
- **Distributed Training (DDP):** A robust distributed training script (`src/train_ddp.py`) using PyTorch's `DistributedDataParallel` (DDP) and `torchrun` for both single-node and multi-node scaling.
- **Advanced Text Generation:** A flexible generation script (`src/generate.py`) that supports greedy decoding and advanced sampling methods like Top-K and Nucleus (Top-P) sampling.

---

## Developer Quick Start

For local development without Docker, use the Makefile shortcuts:

```bash
# 1) Install dependencies (prod + dev)
make install

# 2) Prepare data and tokenizer
make prepare-data

# 3) Run tests and lint
make test
make lint   # or: make fmt

# 4) Validate distributed setup and train (adjust envs)
make ddp-test N_PROCS_PER_NODE=2
make ddp-train N_PROCS_PER_NODE=2 NNODES=1
```


## Project Architecture

### Model Layers

The model (`src/model.py`) is built from two primary custom layers defined in `src/layers.py`:

- **`CausalSelfAttention`**: This is the brain of the Transformer. For each token in the input sequence, it performs the following:

  - **Projects to Q, K, V**: It first creates three vectors from the token's embedding: a **Query** (what I'm looking for), a **Key** (what I contain), and a **Value** (what I will provide).
  - **Calculates Attention Scores**: It compares the Query of the current token to the Keys of all previous tokens in the sequence. This comparison produces a score indicating how relevant each previous token is to the current one.
  - **Applies Causal Mask**: To prevent the model from looking into the future (a requirement for language generation), a mask is applied to hide all subsequent tokens.
  - **Computes Weighted Sum**: The scores are converted into probabilities (weights) using a softmax function. These weights are then used to create a weighted sum of the Value vectors of all previous tokens. The result is a new representation for the current token, enriched with context from its past.
  - **KV Caching**: For efficient generation, the Keys and Values for each token are cached after being computed. This means that when generating the next token, the model only needs to compute the Q, K, V for the single new token and can reuse the cached K and V from all previous tokens, dramatically speeding up inference.

- **`TransformerBlock`**: This is the fundamental building block that is stacked multiple times to create the full model. Each block is responsible for refining the token representations. It consists of two main sub-layers:
  1.  **Communication (`CausalSelfAttention`)**: The first layer allows tokens to communicate with each other and gather context, as described above.
  2.  **Computation (`FeedForward Network`)**: The second layer is a standard Multi-Layer Perceptron (MLP) that processes each token's representation independently. This can be thought of as the model "thinking" about the context it just gathered.
  - **Residual Connections & Layer Normalization**: Each of these two sub-layers is wrapped with a residual ("skip") connection and a Layer Normalization step. This is a crucial design pattern that allows gradients to flow more easily through the deep network, preventing issues like vanishing gradients and making the training process much more stable.

### Text Generation & Sampling

The `src/generate.py` script is used to produce text from a trained model checkpoint. It supports several decoding strategies to control the trade-off between coherence and creativity:

- **Greedy Search (`--method greedy`)**: The simplest approach. At each step, it selects the single token with the highest probability. It is fast and deterministic but can lead to repetitive and uninteresting text.

- **Top-K Sampling (`--method top-k`)**: Filters the vocabulary to only the `k` most likely next tokens (controlled by `--top_k`). It then redistributes the probability mass among these `k` tokens and samples from the result. This introduces randomness while preventing the model from picking very unlikely tokens.

- **Top-P / Nucleus Sampling (`--method top-p`)**: A more dynamic approach. It selects the smallest set of tokens whose cumulative probability is greater than `p` (controlled by `--top_p`). This allows the set of candidate tokens to grow or shrink depending on the model's certainty, often producing more diverse and interesting results than Top-K.

- **Temperature (`--temperature`)**: This parameter controls the "creativity" of the sampling by scaling the logits before the softmax function. Values > 1.0 make the output more random, while values < 1.0 make it more deterministic and focused.

---

## Manual Installation (Alternative)

This method is for users who prefer not to use Docker.

### Prerequisites

- Python 3.9+
- Conda (recommended for environment management)
- An NVIDIA GPU with CUDA installed

### Installation

Clone the repository and set up the Conda environment.

```bash
git clone <your-repo-url>
cd disttraintest

conda create -n disttrain python=3.10 -y
conda activate disttrain

pip install -r requirements.txt
```

### Prepare Data

Download the dataset and train the tokenizer. This only needs to be done once.

```bash
python -m data.prepare_data
```

---

## Setup and Usage (Docker Recommended)

### Prerequisites

- An NVIDIA GPU with CUDA drivers installed.
- [Docker](https://docs.docker.com/get-docker/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your system.

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd disttraintest
```

### 2. Build the Docker Image

Build the Docker image which contains all dependencies and code. This only needs to be done once.

```bash
docker build -t tinygpt-trainer .
```

### 3. Run the Container

Run the container to get an interactive shell. This command mounts the current directory into the container's `/app` directory, so any changes you make locally will be reflected inside the container.

```bash
docker run --gpus all -it --rm -v "$(pwd)":/app tinygpt-trainer
```

_You are now inside the container's shell and can run the following commands._

### 4. Prepare Data

Inside the container, run the data preparation script. This will download the dataset and train the tokenizer.

```bash
python -m data.prepare_data
```

---

## Training the Model

### Production Environment 1: Single-Node, Multi-GPU Training

To train on a single machine with multiple GPUs (e.g., 4 GPUs), first edit `scripts/train_ddp.sh` and set `N_PROCS_PER_NODE=4`. Then, run the script from within the Docker container:

```bash
bash scripts/train_ddp.sh
```

`torchrun` will automatically manage the processes and utilize all 4 GPUs on the machine.

### Production Environment 2: Multi-Node, Multi-GPU Distributed Training

This demonstrates scaling across multiple machines. For this example, we'll use **2 nodes, each with 4 GPUs**.

1.  **Setup:**

    - Launch two cloud instances (e.g., on [RunPod](https://runpod.io) or [Vast.ai](https://vast.ai)).
    - On both nodes, ensure Docker is installed and clone the repository.
    - Designate one node as the **master node** and find its private IP address (e.g., `10.0.0.5`).

2.  **Launch Command:**
    You will run the Docker container on **both** machines, but pass in the required environment variables for DDP.

    **On the Master Node (Node 0):**

    ```bash
    # First, build the image on this node
    docker build -t tinygpt-trainer .

    # Now, run the container with the correct environment variables
    docker run --gpus all --network=host -it --rm -v "$(pwd)":/app \
      -e NNODES=2 \
      -e NODE_RANK=0 \
      -e MASTER_ADDR=10.0.0.5 \
      -e MASTER_PORT=29500 \
      tinygpt-trainer
    ```

    _Inside the container, run `bash scripts/train_ddp.sh`._

    **On the Second Node (Node 1):**

    ```bash
    # First, build the image on this node
    docker build -t tinygpt-trainer .

    # Run the container, pointing to the master node
    docker run --gpus all --network=host -it --rm -v "$(pwd)":/app \
      -e NNODES=2 \
      -e NODE_RANK=1 \
      -e MASTER_ADDR=10.0.0.5 \
      -e MASTER_PORT=29500 \
      tinygpt-trainer
    ```

    _Inside the container, also run `bash scripts/train_ddp.sh`._

`torchrun` on both containers will now coordinate to launch a total of 8 processes, and training will begin across both nodes.

### Configuration and Hyperparameters

The training script `src/train_ddp.py` accepts command-line arguments to control the model architecture and training process. You can modify these in `scripts/train_ddp.sh` to run experiments.

**Key Arguments:**

- `--d_model`: The embedding dimension for the model (default: 512).
- `--n_layers`: The number of Transformer blocks to stack (default: 4).
- `--n_heads`: The number of attention heads in the self-attention mechanism (default: 8).
- `--learning_rate`: The peak learning rate for the optimizer (default: 3e-4).
- `--num_epochs`: The total number of epochs to train for (default: 50).
- `--batch_size`: The batch size **per GPU** (default: 16).
- `--seq_len`: The sequence length for training examples (default: 256).

To change a parameter, simply edit the corresponding line in `scripts/train_ddp.sh`:

```bash
# Example: Training a larger model for 100 epochs
torchrun ... \
    src/train_ddp.py \
    --d_model=768 \
    --n_layers=8 \
    --n_heads=12 \
    --num_epochs=100
```

---

## Generating Sample Text

After training, you can generate text from your saved checkpoints using `src/generate.py`. Run the following command from inside the Docker container, pointing to the checkpoint you wish to use.

```bash
python src/generate.py \
  --checkpoint "runs_ddp/latest/model_final.pt" \
  --prompt "The secret to happiness is" \
  --method "top-p" \
  --max_new_tokens 100 \
  --temperature 0.9 \
  --top_p 0.92
```

This command will load the final model from the latest training run and generate 100 new tokens using Nucleus Sampling.

---

## Production Deployment Considerations

While the manual script and Docker setup are excellent for demonstration, a real-world production environment requires more robust automation and management. This is where Kubernetes comes in.

### From Docker to Kubernetes

The `Dockerfile` in this project is the first critical step towards production. It creates a portable, reproducible image of the application.

The next step is to use a **container orchestrator** like Kubernetes to manage running these containers at scale. Instead of manually SSHing into nodes and running `docker run`, you would define the entire distributed job in a single YAML file.

### The Role of Kubeflow and PyTorchJob

For PyTorch workloads, the standard tool in the Kubernetes ecosystem is **Kubeflow**. Specifically, you would use the `PyTorchJob` operator, which is custom-built to handle the complexities of distributed training.

With `PyTorchJob`, Kubernetes would automatically:

1.  Provision the required number of pods (containers).
2.  Designate a master pod.
3.  Inject the `MASTER_ADDR`, `RANK`, and `WORLD_SIZE` environment variables into each pod automatically.
4.  Manage pod lifecycle, including restarting failed pods.

This automates the entire multi-node setup, making the training process reliable, repeatable, and scalable—the core goals of MLOps.
