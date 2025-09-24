#!/bin/bash

VERSIONS=(
    "7.4" "7.5" "7.6"
    "8.1" "8.2" "8.3" "8.4"
    "9.1" "9.2" "9.3"
    "10.1" "10.2"
    "11.1" "11.2" "11.3"
    "12.1" "12.2" "12.3" "12.4" "12.5"
    "13.1" "13.2" "13.3" "13.4"
    "14.1" "14.2" "14.3"
    "15.1" "15.2" "15.3"
)

MODEL="MMMU-Pro_Gemini1.5"

for VERSION in "${VERSIONS[@]}"; do
    echo "Running $MODEL for version $VERSION..."
    python /share/data/drive_4/insaf/Prompting/Infer/mmmu-pro/infer_mmmu-pro_gemini1.5.py s4 $VERSION | tee -a /share/data/drive_4/insaf/Prompting/Outputs/mmmu-pro/s4/mmmu-pro_gemini1.5_s4_$VERSION.txt
done

echo "All versions processed."
