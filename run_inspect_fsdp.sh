#!/usr/bin/env bash

# --- Defaults ---
NPROCS=6
# --- Parse Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nprocs) NPROCS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# --- Environment Configurations ---
# Silence the CPU thread pooling warning
export OMP_NUM_THREADS=1

# Silence the internal PyTorch C++ engine NumPy warning
export PYTHONWARNINGS="ignore:Failed to initialize NumPy"

# Disable libuv for torchrun (required on Windows builds)
export USE_LIBUV=0

# --- Documentation ---
# To disable boolean flags below, add a `no-` prefix (e.g., --no-show-memory-training)

# --- Execution ---
torchrun \
  --nproc-per-node="$NPROCS" \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  inspect_fsdp.py \
  --num-iterations 1 \
  --memory-first-iter-only \
  --show-memory-training \
  --show-memory-checkpoint \
  2>&1 | grep --line-buffered -v "Redirects are currently not supported in MacOs"
