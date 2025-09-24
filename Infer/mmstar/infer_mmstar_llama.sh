#!/bin/bash

# List of versions
VERSIONS=(
    "11.3" "8.1" "8.4"
)

MODEL="Llama"

# Loop through each version and run the command
for VERSION in "${VERSIONS[@]}"; do
    echo "Running inference for $MODEL: $VERSION..."
    
    CUDA_VISIBLE_DEVICES=7 python Infer/mmstar/infer_mmstar_llama.py "$VERSION" | tee -a "Outputs/mmstar/mmstar_llama_${VERSION}.txt"
    
    echo "Completed inference for version $VERSION."
    echo "--------------------------------------------------"
done

echo "All runs completed."