# edufsdp (`edufsdp`)

> **A transparent, ground-up implementation of Fully Sharded Data Parallel (FSDP) engineered for educational clarity, low-level inspection, and execution on Apple Silicon / CPU via the Gloo backend.**

---

## 💡 Overview

While PyTorch's native `FullyShardedDataParallel` (FSDP) is optimized for high-throughput GPU training with complex C++/NCCL pipelines, its internal abstractions can be difficult to inspect and step through.

`edufsdp` is an educational reference implementation designed to make distributed parameter sharding, autograd hook interception, and collective communications completely explicit and transparent. Built to run natively on CPU and Apple Silicon via PyTorch’s `Gloo` backend, `edufsdp` mirrors the mathematical lifecycle of production FSDP down to $\le 10^{-5}$ precision.

---

## ✨ Key Features & Architecture

* **Dual Parameter Sharding Engine:**
  * **Manual Engine (`ShardedParameter`):** Explicit flat-buffer slicing, manual padding math, and low-level `dist.reduce_scatter` / `dist.all_gather` calls.
  * **Native `DTensor` Engine (`DTensorShardedParameter`):** Modern PyTorch 2.x placement-driven sharding leveraging `DeviceMesh`, `Shard(0)`, `Replicate()`, and `Partial()` abstractions.
* **Explicit Forward & Backward Hooks:**
  * **`ForwardPreHook`:** Dynamically reconstructs parameters prior to layer execution.
  * **`ForwardPostHook`:** Immediately frees full-sized parameters post-execution to enforce memory bounds.
  * **`DTensorBackwardHook`:** Intercepts unreduced full gradients during autograd and triggers collective reduce-scatter ops back to local parameter shards.
* **Flexible State Dict Management:**
  * **`Full` Mode:** All-gathers un-sharded weights into standard PyTorch module formats (compatible with single-device inference).
  * **`Local` Mode:** Zero-communication per-rank checkpointing for fast local saves/resumptions.
* **Mathematically Verified:** Includes a verification suite validating loss convergence, parameter updates, and checkpoint compatibility directly against PyTorch's native FSDP engine.

---

## 🛠 Execution Lifecycle

```text

   ┌────────────────────────────────────────────────────────┐
   │                   Parameter Sharding                   │
   │    Split global parameters into local rank shards      │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼

┌───────────────────────────────────────────────────────────────────────────────┐
│ Forward Pass (Per Submodule)                                                  │
│  1. ForwardPreHook  ──> Gather un-sharded weight W (Replicate)                │
│  2. Compute Output  ──> Run activation forward pass                           │
│  3. ForwardPostHook ──> Free full weight W, retain local shard W_local        │
└──────────────────────────────────┬────────────────────────────────────────────┘
│
▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Backward Pass (Autograd)                                                      │
│  1. Compute Grad   ──> Un-sharded gradient G_full computed (Partial)          │
│  2. BackwardHook   ──> Reduce-scatter G_full across mesh -> G_shard           │
│  3. Gradient Step  ──> Attach G_shard to local_parameter.grad                     │
└──────────────────────────────────┬────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│                    Optimizer Step                      │
│    Optimizer updates persistent local_parameter shards     │
└────────────────────────────────────────────────────────┘

```

---

## 🧪 Verification & Parity Testing

`edufsdp` includes a dual-rank verification suite (`test_verification.py` and `test_checkpoint_verification.py`) that executes via `torchrun`:

```bash
# Run convergence verification against PyTorch Native FSDP
torchrun --nproc_per_node=2 tests/test_verification.py

# Run checkpoint & state_dict compatibility tests
torchrun --nproc_per_node=2 tests/test_checkpoint_verification.py

```

---

## 🚀 Quickstart

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from edufsdp import EducationalFSDP

# 1. Initialize Gloo Process Group
dist.init_process_group(backend="gloo")

# 2. Define Model & Synchronize Initial Weights
raw_model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
for p in raw_model.parameters():
    dist.broadcast(p.data, src=0)

# 3. Wrap Model with EducationalFSDP
model = EducationalFSDP(raw_model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. Standard Training Loop
optimizer.zero_grad()
output = model(torch.randn(8, 64))
loss = output.sum()
loss.backward()
optimizer.step()

```

<ElicitationsGroup message="Would you like to add any extra sections to this README?">
  <Elicitation label="Add a 'Learnings & Pitfalls' section" query="Add a 'Key Technical Learnings & Pitfalls' section to the README summarizing autograd hooks, DTensor placements, and optimizer binding bugs."/>
  <Elicitation label="Add Installation and Project Structure sections" query="Add a project directory structure tree and step-by-step installation setup guide to the README."/>
</ElicitationsGroup>
