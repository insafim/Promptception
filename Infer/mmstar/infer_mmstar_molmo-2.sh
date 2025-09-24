#!/bin/bash

# List of versions
VERSIONS=(
    "1.3" "4.2" "8.2" "15.3"
)

MODEL="Molmo"

# Loop through each version and run the command
for VERSION in "${VERSIONS[@]}"; do
    echo "Running inference for $MODEL: $VERSION..."
    
    CUDA_VISIBLE_DEVICES=0 python Infer/mmstar/infer_mmstar_molmo.py "$VERSION" | tee -a "Outputs/mmstar/mmstar_molmo_${VERSION}.txt"
    
    echo "Completed inference for version $VERSION."
    echo "--------------------------------------------------"
done

echo "All runs completed."

