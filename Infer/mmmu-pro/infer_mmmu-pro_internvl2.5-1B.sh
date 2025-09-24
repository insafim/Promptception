#!/bin/bash

VERSIONS=(
    "4.3" "4.4" "4.5" "5.1" "5.2" "5.3" "6.1"
)

MODEL="MMMU-Pro_s4_Intern-VL2.5-1B"

for VERSION in "${VERSIONS[@]}"; do
    echo "Running $MODEL for version $VERSION..."
    CUDA_VISIBLE_DEVICES=7 python Infer/mmmu-pro/infer_mmmu-pro_internvl2.5-1B.py s4 $VERSION | tee -a Outputs/mmmu-pro/s4/mmmu-pro_internvl2.5-1B_s4_$VERSION.txt
done

echo "All versions processed."
