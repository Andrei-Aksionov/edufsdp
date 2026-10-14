#!/usr/bin/env bash

torchrun \
  --nproc-per-node=6 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  inspect_fsdp.py \
  --num-iterations 1 \
  --memory-first-iter-only \
  --show-memory-training \
  --show-memory-checkpoint

  # To disable boolean flags add a `no-` prefix, e.g. --no-show-memory-training
