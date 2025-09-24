#!/bin/bash

# List of versions
VERSIONS=(
    "14.1" "14.2" "14.3"
)

MODEL="InternVL2.5-38B"

# Loop through each version and run the command
for VERSION in "${VERSIONS[@]}"; do
    echo "Running inference for $MODEL: $VERSION..."
    
    CUDA_VISIBLE_DEVICES=4,5,6 python Infer/mmstar/infer_mmstar_internvl2.5-38B.py "$VERSION" | tee -a "Outputs/mmstar/mmstar_internvl2.5-38B_${VERSION}.txt"
    
    echo "Completed inference for version $VERSION."
    echo "--------------------------------------------------"
done

echo "All runs completed."