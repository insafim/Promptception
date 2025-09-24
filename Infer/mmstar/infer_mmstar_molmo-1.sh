#!/bin/bash

# List of versions
VERSIONS=(
    "1.2" "1.3" "1.1"
    "2.1" "2.2" "2.3" "2.4" "2.5" "2.6" "2.7" "2.8" "2.9"
    "3.1" "3.2"
    "4.1" "4.2" "4.3" "4.4" "4.5"
    "5.1" "5.2" "5.3"
    "6.1" "6.2" "6.3" "6.4"
    "7.1" "7.2" "7.3" "7.4" "7.5" "7.6"
)

MODEL="Molmo"

# Loop through each version and run the command
for VERSION in "${VERSIONS[@]}"; do
    echo "Running inference for $MODEL: $VERSION..."
    
    CUDA_VISIBLE_DEVICES=1 python Infer/mmstar/infer_mmstar_molmo.py "$VERSION" | tee -a "Outputs/mmstar/mmstar_molmo_${VERSION}.txt"
    
    echo "Completed inference for version $VERSION."
    echo "--------------------------------------------------"
done

echo "All runs completed."