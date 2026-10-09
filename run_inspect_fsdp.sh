#!/usr/bin/env bash

torchrun \
  --nproc-per-node=3 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  inspect_fsdp.py \
  --num-iterations 3 \
  --memory-first-iter-only \
  --show-memory-training \
  --show-memory-checkpoint

  # NOTE: To disable boolean flags add `no-` prefix, e.g. --no-show-memory-training
