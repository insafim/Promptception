#!/bin/bash

# List of versions
# VERSIONS=(
#     "2.6" "2.7" "2.8" "2.9"
# )
VERSIONS=(
    "2.9"
)

MODEL="InternVL2.5-8B"

# Loop through each version and run the command
for VERSION in "${VERSIONS[@]}"; do
    echo "Running inference for $MODEL: $VERSION..."
    
    CUDA_VISIBLE_DEVICES=3 python Infer/mmstar/infer_mmstar_internvl2.5-8B.py "$VERSION" | tee -a "Outputs/mmstar/mmstar_internvl2.5-8B_${VERSION}-WO_Persona.txt"
    # CUDA_VISIBLE_DEVICES=0 python Infer/mmstar/infer_mmstar_internvl2.5-8B.py "$VERSION"

    echo "Completed inference for version $VERSION."
    echo "--------------------------------------------------"
done

echo "All runs completed."