#!/bin/bash

# List of versions
# VERSIONS=(
#     "1.1" "1.2" "1.3"
#     "2.1" "2.2" "2.3" "2.4" "2.5" "2.6" "2.7" "2.8" "2.9"
#     "3.1" "3.2"
#     "4.1" "4.2" "4.3" "4.4"
#     "5.1" "5.2" "5.3"
#     "6.1" "6.2" "6.3" "6.4"
#     "7.1" "7.2" "7.3" "7.4" "7.5" "7.6"
#     "8.1" "8.2" "8.3" "8.4"
#     "9.1" "9.2" "9.3"
#     "10.1" "10.2"
#     "11.1" "11.2" "11.3"
#     "12.1" "12.2" "12.3" "12.4" "12.5"
#     "13.1" "13.2" "13.3" "13.4"
#     "14.1" "14.2" "14.3"
#     "15.1" "15.2" "15.3"
# )

VERSIONS=(
    "12.5"
)

# Define the model name (Modify this as needed)
MODEL="MMStar_Llava-OV"

# Loop over each version
for VERSION in "${VERSIONS[@]}"; do
    # RESULTS_PATH="Results/MMStar/MMStar_Molmo_Rerun/mmstar_molmo_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMStar/MMStar_Molmo/eval_mmstar_molmo_${VERSION}.txt"

    RESULTS_PATH="Results/MMStar/MMStar_Llava-OV-7B/mmstar_llava-ov-7B_${VERSION}.json"
    OUTPUT_PATH="Eval_Output/MMStar/MMStar_Llava-OV-7B/eval_mmstar_llava-ov-7B_${VERSION}.txt"

    echo "Running evaluation for ${MODEL} version ${VERSION}..."
    python Evaluate/eval_mmstar.py --results_path "$RESULTS_PATH" | tee -a "$OUTPUT_PATH"
    
    # newline
    echo ""
done
