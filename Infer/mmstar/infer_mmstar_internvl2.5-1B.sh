#!/bin/bash

# List of versions
VERSIONS=(
    "1.1" "11.3"
)

MODEL="InternVL2.5-1B"

# Loop through each version and run the command
for VERSION in "${VERSIONS[@]}"; do
    echo "Running inference for $MODEL: $VERSION..."
    
    CUDA_VISIBLE_DEVICES=1 python Infer/mmstar/infer_mmstar_internvl2.5-1B.py "$VERSION" | tee -a "Outputs/mmstar/mmstar_internvl2.5-1B_${VERSION}.txt"
    
    echo "Completed inference for version $VERSION."
    echo "--------------------------------------------------"
done

echo "All runs completed."