### Updated guidance for beating the single-node baseline **without touching the network**

| Lever                                                    | What it does                                                                                                                                      | How to set it here                                                                                                                     | Expected effect on 2-node run                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Scale the model up × 6–8**<br>_(make compute heavier)_ | Push the step time on each GPU from ≈ 4 ms → 25–40 ms, so the 15–20 ms all-reduce latency becomes ≤ 40 % of wall-time instead of 4 × the compute. | `bash<br># train_ddp.sh<br>--d_model 768 \  # or 1024<br>--n_layers 12 \  # keep heads = d_model/64<br>--batch_size 16  # per-GPU<br>` | **2–3 ×** speed-up (20 k → 40-60 k tok/s) _if memory fits_. |
| **Raise gradient-accumulation**                          | Reduces number of all-reduces per epoch linearly.                                                                                                 | `GRAD_ACC_STEPS=16` (was 4)                                                                                                            | Amortises comm; +**2–4 ×** if VRAM allows.                  |
| **Bigger buckets**                                       | Fewer NCCL launches; keeps link busy.                                                                                                             | `BUCKET_CAP_MB=200` or `512`                                                                                                           | 1.3–1.8 × on 1/10 GbE.                                      |
| **fp16 / bf16 communication** _(already bf16 compute)_   | Halves wire traffic.                                                                                                                              | add<br>`--use_fp16_comm` → hooks (see below)                                                                                           | extra \~1.5 ×.                                              |
| **Static-graph DDP**                                     | Skips autograd graph analysis every step.                                                                                                         | set env `TORCH_DISTRIBUTED_ENABLE_STATIC_GRAPH=1`                                                                                      | shaves 3–5 ms/step.                                         |
| **Overlap compute/comm**                                 | Launch NCCL in parallel with backward.                                                                                                            | call `model.register_comm_hook(torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook)`                          | 10–15 % win.                                                |
| **Turn on Flash-Attention 2 + fused MLP**                | Lifts compute throughput, lets you afford bigger model.                                                                                           | `pip install flash-attn==2 && export USE_FLASH_ATTENTION=1`                                                                            | 20–30 % step-time drop.                                     |

> **Why GPU util shows 100 % with tiny model**
> NCCL kernels (all-reduce) run on the streaming multiprocessors, so the profiler counts them as “GPU busy” even when the arithmetic units are just moving bytes. The 81 % **“Compute Mux Busy”** vs **“DRAM Rd/Wr Busy”** split in `nvidia-smi dmon` will expose this.

---

#### Concrete recipe that usually **beats single-node on 10 GbE**

```bash
# Larger model – still small enough for 16 GB cards
export DMODEL=768
export NLAYERS=12
export NHEADS=12          # d_model / 64 rule-of-thumb
export BATCH=16           # per GPU
export GRAD_ACC=16        # keeps global batch similar
export BUCKET=512
export TORCH_DISTRIBUTED_ENABLE_STATIC_GRAPH=1

bash scripts/train_ddp.sh \
  --d_model=$DMODEL --n_layers=$NLAYERS --n_heads=$NHEADS \
  --batch_size=$BATCH --gradient_accumulation_steps=$GRAD_ACC \
  --bucket_cap_mb=$BUCKET
```

_This usually lands in the **40 k–60 k tok/s** range over two nodes on a 10 GbE link, finally surpassing your 100 k tok/s single-node figure – because each node now delivers \~25 k-30 k._

---

### Optional code tweak (tiny, can copy-paste)

Add this once, right after you wrap the model in DDP in **`src/train_ddp.py`**:

```python
# == Optional: compress gradients to fp16 before all-reduce ==
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as dh
model.register_comm_hook(None, dh.fp16_compress_hook)
```

_No additional XML code patch provided – slot this manually if you choose to test the hook._
