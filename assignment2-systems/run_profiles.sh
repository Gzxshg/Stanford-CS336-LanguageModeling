#!/bin/bash
# Run the 2.1.4 nsys profiling matrix on GPU 6 (the idle one).
set -u
cd "$(dirname "$0")"

NSYS=/home/gzx26/dataY/nsys/root/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
export CUDA_VISIBLE_DEVICES=6
export UV_CACHE_DIR=/home/gzx26/dataY/uv/cache
export UV_PYTHON_INSTALL_DIR=/home/gzx26/dataY/uv/python
UV=/home/gzx26/dataY/uv/bin/uv

run() {
    name=$1; shift
    echo "=== $name ==="
    "$UV" run "$NSYS" profile --trace=cuda,nvtx --force-overwrite=true \
        -o "profiles/$name" \
        python cs336_systems/benchmarking_script.py "$@" 2>&1 | grep -v "^WARNING\|Try the"
}

# three power-of-two context lengths on small (forward-only)
run small_ctx256_fwd   --model_size small --context_length 256  --mode forward-only
run small_ctx1024_fwd  --model_size small --context_length 1024 --mode forward-only
run small_ctx4096_fwd  --model_size small --context_length 4096 --mode forward-only

# second model size: large (forward-only)
run large_ctx512_fwd   --model_size large --context_length 512  --mode forward-only
run large_ctx4096_fwd  --model_size large --context_length 4096 --mode forward-only

# fwd+bwd and full training step on small/1024
run small_ctx1024_fwd_bwd --model_size small --context_length 1024 --mode forward-and-backward
run small_ctx1024_train   --model_size small --context_length 1024 --mode full-training-steps

# annotated attention for question (e)
run small_ctx1024_fwd_ann --model_size small --context_length 1024 --mode forward-only --annotate

echo "ALL DONE"
