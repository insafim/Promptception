#!/bin/bash

VERSIONS=(
    "2.4" "2.6" "2.7" "2.8" "2.9"
)

MODEL="MMMU-Pro_s4_Intern-VL2.5-8B"

for VERSION in "${VERSIONS[@]}"; do
    echo "Running $MODEL for version $VERSION..."
    CUDA_VISIBLE_DEVICES=0 python Infer/mmmu-pro/infer_mmmu-pro_internvl2.5-8B.py s4 $VERSION | tee -a Outputs/mmmu-pro/s4/mmmu-pro_internvl2.5-8B_s4_$VERSION.txt
done

echo "All versions processed."
