#!/bin/bash

VERSIONS=(
    "14.1"
)

# VERSIONS=(
#     "7.5"
#     "8.2" "13.4"
# )

MODEL="MMMU-Pro_s4_Intern-VL2.5-38B"

for VERSION in "${VERSIONS[@]}"; do
    echo "Running $MODEL for version $VERSION..."
    CUDA_VISIBLE_DEVICES=0,1,2 python Infer/mmmu-pro/infer_mmmu-pro_internvl2.5-38B.py s4 $VERSION | tee -a Outputs/mmmu-pro/s10/mmmu-pro_internvl2.5-38B_s4_$VERSION.txt
done

echo "All versions processed."
