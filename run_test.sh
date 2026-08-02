# torchrun \
#   --nnodes=1 \
#   --nproc-per-node=2 \
#   --master-addr=127.0.0.1 \
#   --master-port=29500 \
#   tests/test_verification.py

torchrun \
  --nnodes=1 \
  --nproc-per-node=2 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  tests/test_checkpoint_verification.py
