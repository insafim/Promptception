#!/bin/bash

# List of versions
VERSIONS=(
    "2.8" "2.9"
)

MODEL="Qwen2-VL-7B"

# Loop through each version and run the command
for VERSION in "${VERSIONS[@]}"; do
    echo "Running inference for $MODEL: $VERSION..."
    
    CUDA_VISIBLE_DEVICES=2 python Infer/mmstar/infer_mmstar_qwen2-vl-7B.py "$VERSION" | tee -a "Outputs/mmstar/mmstar_qwen2-vl-7B_${VERSION}.txt"
    
    echo "Completed inference for version $VERSION."
    echo "--------------------------------------------------"
done

echo "All runs completed."