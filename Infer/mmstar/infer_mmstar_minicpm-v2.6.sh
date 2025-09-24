#!/bin/bash

# List of versions
VERSIONS=(
    "2.3" "2.4"
)

MODEL="MiniCPM-V2.6"

# Loop through each version and run the command
for VERSION in "${VERSIONS[@]}"; do
    echo "Running inference for $MODEL: $VERSION..."
    
    CUDA_VISIBLE_DEVICES=1 python Infer/mmstar/infer_mmstar_minicpm-v2.6.py "$VERSION" | tee -a "Outputs/mmstar/mmstar_minicpm-v2.6_${VERSION}.txt"
    
    echo "Completed inference for version $VERSION."
    echo "--------------------------------------------------"
done

echo "All runs completed."