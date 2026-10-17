#!/usr/bin/env bash

# --- Documentation ---
# To disable boolean flags below, add a `no-` prefix (e.g., --no-show-memory-training)

# --- Environment Configurations ---
# Silence the CPU thread pooling warning
export OMP_NUM_THREADS=1

# Silence the internal PyTorch C++ engine NumPy warning
export PYTHONWARNINGS="ignore:Failed to initialize NumPy"

# --- Execution ---
torchrun \
  --nproc-per-node=6 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  inspect_fsdp.py \
  --num-iterations 1 \
  --memory-first-iter-only \
  --show-memory-training \
  --show-memory-checkpoint \
  2>&1 | grep --line-buffered -v "Redirects are currently not supported in MacOs"
