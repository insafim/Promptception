#!/bin/bash

VERSIONS=(
    "1.1" "1.2" "1.3"
    "2.1" "2.2" "2.3" "2.4" "2.5" "2.6" "2.7" "2.8" "2.9"
    "3.1" "3.2"
    "4.1" "4.2" "4.3" "4.4" "4.5"
    "5.1" "5.2" "5.3"
    "6.1" "6.2" "6.3" "6.4"
    "7.1" "7.2" "7.3" "7.4" "7.5" "7.6"
    "8.1" "8.2" "8.3" "8.4"
    "9.1" "9.2" "9.3"
    "10.1" "10.2"
    "11.1" "11.2" "11.3" "11.4" "11.5"
    "12.1" "12.2" "12.3" "12.4" "12.5"
    "13.1" "13.2" "13.3" "13.4"
    "14.1" "14.2" "14.3"
    "15.1" "15.2" "15.3"
)

MODEL="MVBench_Gemini"

for VERSION in "${VERSIONS[@]}"; do
    echo "Running $MODEL for version $VERSION..."
    python /share/data/drive_4/insaf/Prompting/Infer/mvbench/infer_mvbench_gemini.py $VERSION | tee -a /share/data/drive_4/insaf/Prompting/Outputs/mvbench/mvbench_gemini_$VERSION.txt
done

echo "All versions processed."
