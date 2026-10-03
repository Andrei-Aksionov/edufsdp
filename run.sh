torchrun \
  --nnodes=1 \
  --nproc-per-node=3 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  train.py
