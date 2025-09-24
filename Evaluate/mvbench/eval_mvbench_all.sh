#!/bin/bash

# List of versions
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

# VERSIONS=(
#     "1.1"
# )

# Define the model name (Modify this as needed)
# MODEL="MVBench_Llava-OV-7B"
MODEL="MVBench_Qwen2-VL-7B"
# MODEL="MVBench_Intern-VL-1B"
# MODEL="MVBench_Intern-VL-8B"
# MODEL="MVBench_Intern-VL-38B"
# MODEL="Gemini1.5"
# MODEL="GPT4o"

# MODEL="MVBench_MiniCPM-V2.6"

# Loop over each version
for VERSION in "${VERSIONS[@]}"; do

    # RESULTS_PATH="Results/MVBench/MVBench_Llava-OV-7B/mvbench_llava-ov-7B_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MVBench/MVBench_Llava-OV-7B/eval_mvbench_llava-ov-7B_${VERSION}.txt"
    # RESULTS_PATH="Results/MVBench100/MVBench_Llava-OV-7B/mvbench_llava-ov-7B_${VERSION}_updated.json"
    # OUTPUT_PATH="Eval_Output/MVBench100/MVBench_Llava-OV-7B/eval_mvbench_llava-ov-7B_${VERSION}-100.txt"

    # RESULTS_PATH="Results/MVBench/MVBench_Intern-VL-8B/mvbench_intern-vl-8B_${VERSION}.json"
    # RESULTS_PATH="Results/MVBench100/MVBench_Intern-VL-8B/mvbench_intern-vl-8B_${VERSION}_updated.json"
    # OUTPUT_PATH="Eval_Output/MVBench/MVBench_Intern-VL-8B-NEW/eval_mvbench_intern-vl-8B_${VERSION}-100.txt"
    # RESULTS_PATH="Results/MVBench100/MVBench_Intern-VL-38B/mvbench_intern-vl-38B_${VERSION}_updated.json"
    # OUTPUT_PATH="Eval_Output/MVBench/MVBench_Intern-VL-38B/eval_mvbench_intern-vl-38B_${VERSION}-100.txt"

    # RESULTS_PATH="Results/MVBench/MVBench_Intern-VL-38B/mvbench_intern-vl-38B_${VERSION}.json"
    # RESULTS_PATH="Results/MVBench/MVBench_Intern-VL-38B/Extract_Llama/mvbench_intern-vl-38B_${VERSION}_updated.json"
    # OUTPUT_PATH="Eval_Output/MVBench/MVBench_Intern-VL-38B/eval_mvbench_intern-vl-38B_${VERSION}-elizabeth.txt"


    # RESULTS_PATH="Results/MVBench/MVBench_Intern-VL-1B/mvbench_intern-vl-1B_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MVBench/MVBench_Intern-VL-1B-NEW/eval_mvbench_intern-vl-1B_${VERSION}.txt"
    # RESULTS_PATH="Results/MVBench100/MVBench_Intern-VL-1B/mvbench_intern-vl-1B_${VERSION}_updated.json"
    # OUTPUT_PATH="Eval_Output/MVBench/MVBench_Intern-VL-1B-NEW/eval_mvbench_intern-vl-1B_${VERSION}-100.txt"


    # RESULTS_PATH="Results/MVBench/MVBench_Qwen2-VL-7B/mvbench_qwen2-vl-7B_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MVBench/MVBench_Qwen2-VL-7B/eval_mvbench_qwen2-vl-7B_${VERSION}.txt"
    # RESULTS_PATH="Results/MVBench100/MVBench_Qwen2-VL-7B/mvbench_qwen2-vl-7B_${VERSION}_updated.json"
    # OUTPUT_PATH="Eval_Output/MVBench100/MVBench_Qwen2-VL-7B/eval_mvbench_qwen2-vl-7B_${VERSION}-100.txt"
    

    # RESULTS_PATH="Results/MVBench/MVBench_MiniCPM-V2.6/mvbench_minicpm-v2.6_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MVBench/MVBench_MiniCPM-V2.6-NEW/eval_mvbench_minicpm-v2.6_${VERSION}.txt"
    RESULTS_PATH="Results/MVBench100/MVBench_MiniCPM-V2.6/mvbench_minicpm-v2.6_${VERSION}_updated.json"
    OUTPUT_PATH="Eval_Output/MVBench100/MVBench_MiniCPM-V2.6/eval_mvbench_minicpm-v2.6_${VERSION}-100.txt"

    # RESULTS_PATH="Results/MVBench/MVBench_Gemini1.5-Pro/mvbench_gemini1.5-pro_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MVBench/MVBench_Gemini1.5-Pro/eval_mvbench_gemini1.5-pro_${VERSION}.txt"

    # RESULTS_PATH="Results/MVBench/MVBench_GPT4o/mvbench_gpt4o_${VERSION}.json"
    # RESULTS_PATH="Results/MVBench/MVBench_GPT4o/Extract_Llama/mvbench_gpt4o_${VERSION}_updated.json"
    
    # OUTPUT_PATH="Eval_Output/MVBench/MVBench_GPT4o/eval_mvbench_gpt4o_${VERSION}-2.txt"

    echo "Running evaluation for ${MODEL} version ${VERSION}..."
    python Evaluate/eval_mvbench.py --results_path "$RESULTS_PATH" | tee -a "$OUTPUT_PATH"
    
    # newline
    echo ""
done
