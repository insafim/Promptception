#!/bin/bash

VERSIONS=(
    "9.2"
)

MODEL="MVBench_llava-ov-7B"

for VERSION in "${VERSIONS[@]}"; do
    echo "Running $MODEL for version $VERSION..."
    CUDA_VISIBLE_DEVICES=0 python Infer/mvbench/infer_mvbench_llava-ov-7B.py $VERSION | tee -a Outputs/mvbench/mvbench_llava-ov-7B_$VERSION.txt
done

echo "All versions processed."
