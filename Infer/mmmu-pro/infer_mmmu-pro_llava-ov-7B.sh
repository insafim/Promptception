#!/bin/bash

VERSIONS=(
    "2.4" "2.6" "2.7" "2.8" "2.9"
)

MODEL="MMMU-Pro_s4_llava-ov-7B"

for VERSION in "${VERSIONS[@]}"; do
    echo "Running $MODEL for version $VERSION..."
    CUDA_VISIBLE_DEVICES=1 python Infer/mmmu-pro/infer_mmmu-pro_llava-ov-7B.py s4 $VERSION | tee -a Outputs/mmmu-pro/s4/mmmu-pro_llava-ov-7B_s4_$VERSION.txt
done

echo "All versions processed."
