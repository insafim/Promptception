#!/bin/bash

# 6.1, 6.2, 6.3, 6.4, 12.5
VERSION="1.1"

# RESULTS_PATH="Results/MMStar/MMStar_GPT4o/mmstar_gpt4o_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_GPT4o/eval_mmstar_gpt4o_${VERSION}.txt"
# RESULTS_PATH="Results/MMStar/MMStar_GPT4o/mmstar_gpt4o_${VERSION}_wo_persona.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_GPT4o/eval_mmstar_gpt4o_${VERSION}_wo_persona.txt"
RESULTS_PATH="Results/MMStar/MMStar_GPT4o/mmstar_gpt4o_${VERSION}.json"
OUTPUT_PATH="Eval_Output/MMStar/MMStar_GPT4o/eval_mmstar_gpt4o_${VERSION}-algo2.txt"

# RESULTS_PATH="Results/MMStar/MMStar_Gemini1.5-Pro/mmstar_gemini1.5-pro_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Gemini1.5-Pro/eval_mmstar_gemini1.5-pro_${VERSION}.txt"

# VERSION="12.3"
# RESULTS_PATH="Results/MMStar/MMStar_Intern-VL-38B/mmstar_intern-vl-38B_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Intern-VL-38B/eval_mmstar_intern-vl-38B_${VERSION}_GPT.txt"

# VERSION="12.5"
# RESULTS_PATH="Results/MMStar/MMStar_Intern-VL-8B/mmstar_intern-vl-8B_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Intern-VL-8B/eval_mmstar_intern-vl-8B_${VERSION}_GPT.txt"

# VERSION="12.5"
# RESULTS_PATH="Results/MMStar/MMStar_Intern-VL-1B/mmstar_intern-vl-1B_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Intern-VL-1B/eval_mmstar_intern-vl-1B_${VERSION}_GPT.txt"

# VERSION="12.5"
# RESULTS_PATH="Results/MMStar/MMStar_Llava-OV-7B/mmstar_llava-ov-7B_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Llava-OV-7B/eval_mmstar_llava-ov-7B_${VERSION}_GPT.txt"

# VERSION="12.5"
# RESULTS_PATH="Results/MMStar/MMStar-Qwen2-VL-7B/mmstar_qwen2-vl-7B_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar-Qwen2-VL-7B/eval_mmstar_qwen2-vl-7B_${VERSION}_GPT.txt"


python Evaluate/eval_mmstar_GPT.py --results_path "$RESULTS_PATH" | tee -a "$OUTPUT_PATH"
