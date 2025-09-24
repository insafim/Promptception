#!/bin/bash

# s4,s10
# List of versions
# VERSIONS=(
#     "1.1" "1.2" "1.3"
#     "2.1" "2.2" "2.3" "2.4" "2.5" "2.6" "2.7" "2.8" "2.9"
#     "3.1" "3.2"
#     "4.1" "4.2" "4.3" "4.4" "4.5"
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

# vision
# # List of versions
# VERSIONS=(
#     "1.1"
#     "2.1" "2.2" "2.3" "2.4" "2.5" "2.6" "2.7" "2.8" "2.9"
#     "3.1" "3.2"
#     "4.1" "4.2" "4.3" "4.4" "4.5"
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

# Define the model name (Modify this as needed)

# MODEL="MMMU-Pro_Llava-OV-7B"
# MODEL="MMMU-Pro_Qwen2-VL-7B"
# MODEL="MMMU-Pro_Intern-VL-1B"
# MODEL="MMMU-Pro_Intern-VL-8B"
MODEL="MMMU-Pro_Intern-VL-38B"
# MODEL="Gemini1.5"
# MODEL="GPT4o"
# MODEL="MMMU-Pro_MiniCPM-V2.6"

# Loop over each version
for VERSION in "${VERSIONS[@]}"; do
    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_Llava-OV-7B/mmmu-pro_llava-ov-7B_s4_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/s4/MMMU-Pro_Llava-OV-7B/eval_mmmu-pro_llava-ov-7B_s4_${VERSION}.txt"

    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_Qwen2-VL-7B/mmmu-pro_qwen2-vl-7B_s4_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/s4/MMMU-Pro_Qwen2-VL-7B/eval_mmmu-pro_qwen2-vl-7B_s4_${VERSION}.txt"

    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_Intern-VL-8B/mmmu-pro_intern-vl-8B_s4_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/s4/MMMU-Pro_Intern-VL-8B/eval_mmmu-pro_intern-vl-8B_s4_${VERSION}.txt"

    RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_Intern-VL-38B/mmmu-pro_intern-vl-38B_s4_${VERSION}.json"
    OUTPUT_PATH="Eval_Output/MMMU-Pro/s4/MMMU-Pro_Intern-VL-38B/eval_mmmu-pro_intern-vl-38B_s4_${VERSION}.txt"

    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_Intern-VL-8B_s10/mmmu-pro_intern-vl-8B_s10_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/s10/MMMU-Pro_Intern-VL-8B_s10/eval_mmmu-pro_intern-vl-8B_s10_${VERSION}.txt"

    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_Intern-VL-8B_v/mmmu-pro_intern-vl-8B_v_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/v/MMMU-Pro_Intern-VL-8B_v/eval_mmmu-pro_intern-vl-8B_v_${VERSION}.txt"

    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_Intern-VL-38B/mmmu-pro_intern-vl-38B_s4_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/s4/MMMU-Pro_Intern-VL-38B/eval_mmmu-pro_intern-vl-38B_s4_${VERSION}.txt"

    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_GPT4o/mmmu-pro_gpt4o_s4_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/s4/MMMU-Pro_GPT4o/eval_mmmu-pro_gpt4o_s4_${VERSION}.txt"

    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_Gemini1.5/mmmu-pro_gemini1.5_s4_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/s4/MMMU-Pro_Gemini1.5/eval_mmmu-pro_gemini1.5_s4_${VERSION}.txt"

    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_Gemini1.5_s10/mmmu-pro_gemini1.5_s10_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/s10/MMMU-Pro_Gemini1.5_s10/eval_mmmu-pro_gemini1.5_s10_${VERSION}.txt"

    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_Gemini1.5_v/mmmu-pro_gemini1.5_v_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/v/MMMU-Pro_Gemini1.5_v/eval_mmmu-pro_gemini1.5_v_${VERSION}.txt"

    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_MiniCPM-V2.6/mmmu-pro_minicpm-v2.6_s4_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/s4/MMMU-Pro_MiniCPM-V2.6/eval_mmmu-pro_minicpm-v2.6_s4_${VERSION}.txt"

    # RESULTS_PATH="Results/MMMU-Pro/MMMU-Pro_Gemini1.5_v/mmmu-pro_gemini1.5_v_${VERSION}.json"
    # OUTPUT_PATH="Eval_Output/MMMU-Pro/v/MMMU-Pro_Gemini1.5_v/eval_mmmu-pro_gemini1.5_v_${VERSION}.txt"

    echo "Running evaluation for ${MODEL} version ${VERSION}..."
    
    python Evaluate/eval_mmmu-pro.py --results_file "$RESULTS_PATH" | tee -a "$OUTPUT_PATH"
    
    
    # newline
    echo ""
done
