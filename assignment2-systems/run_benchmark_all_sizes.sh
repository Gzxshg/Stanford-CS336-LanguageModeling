#!/bin/bash
# Run the end-to-end benchmark across all model sizes (Assignment 2, Problem 2.1.3b)
# Usage: bash run_benchmark_all_sizes.sh

OUTPUT_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

for size in small medium large xl; do
    echo "=== model size: $size ===" | tee -a "$OUTPUT_FILE"
    uv run python cs336_systems/benchmarking_script.py \
        --model_size "$size" \
        --mode full-training-steps \
        --warmup_steps 5 \
        --measurement_steps 10 \
        2>&1 | tee -a "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "Done. Results saved to $OUTPUT_FILE"
