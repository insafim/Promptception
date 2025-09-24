#!/bin/bash

VERSIONS=(
    "13.3" "13.4" "14.3"
    "15.2" "15.3" 
)

MODEL="MMMU-Pro_s4_qwen2-vl-7B"

for VERSION in "${VERSIONS[@]}"; do
    echo "Running $MODEL for version $VERSION..."
    CUDA_VISIBLE_DEVICES=2 python Infer/mmmu-pro/infer_mmmu-pro_qwen2-vl-7B.py s4 $VERSION | tee -a Outputs/mmmu-pro/s4/mmmu-pro_qwen2-vl-7B_s4_$VERSION.txt
done

echo "All versions processed."
