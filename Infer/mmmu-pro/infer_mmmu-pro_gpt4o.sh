#!/bin/bash

VERSIONS=(
    "12.5"
)

MODEL="MMMU-Pro_GPT4o"

for VERSION in "${VERSIONS[@]}"; do
    echo "Running $MODEL for version $VERSION..."
    python /share/data/drive_4/insaf/Prompting/Infer/mmmu-pro/infer_mmmu-pro_gpt4o.py s4 $VERSION | tee -a /share/data/drive_4/insaf/Prompting/Outputs/mmmu-pro/s4/mmmu-pro_gpt4o_s4_$VERSION.txt
done

echo "All versions processed."
