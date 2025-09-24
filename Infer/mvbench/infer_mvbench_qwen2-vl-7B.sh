#!/bin/bash

VERSIONS=(
    "12.5"
)

MODEL="MVBench_qwen2-vl-7B"

for VERSION in "${VERSIONS[@]}"; do
    echo "Running $MODEL for version $VERSION..."
    CUDA_VISIBLE_DEVICES=1 python Infer/mvbench/infer_mvbench_qwen2-vl-7B.py $VERSION | tee -a Outputs/mvbench/mvbench_qwen2-vl-7B_$VERSION.txt
done

echo "All versions processed."
