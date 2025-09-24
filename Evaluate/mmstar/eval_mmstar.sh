#!/bin/bash

# VERSION="2.9"
# RESULTS_PATH="Results/MMStar/MMStar_Llava-OV-7B/mmstar_llava-ov-7B_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Llava-OV-7B/eval_mmstar_llava-ov-7B_${VERSION}.txt"

# VERSION="15.3"
# RESULTS_PATH="Results/MMStar/MMStar-Qwen2-VL-7B/mmstar_qwen2-vl-7B_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar-Qwen2-VL-7B/eval_mmstar_qwen2-vl-7B_${VERSION}.txt"

VERSION="2.6"
# RESULTS_PATH="Results/MMStar/MMStar_GPT4o/mmstar_gpt4o_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_GPT4o/eval_mmstar_gpt4o_${VERSION}.txt"
RESULTS_PATH="Results/MMStar/MMStar_GPT4o/mmstar_gpt4o_${VERSION}_wo_persona.json"
OUTPUT_PATH="Eval_Output/MMStar/MMStar_GPT4o/eval_mmstar_gpt4o_${VERSION}_wo_persona.txt"

# VERSION="15.3"
# RESULTS_PATH="Results/MMStar/MMStar_Gemini1.5-Pro/mmstar_gemini1.5-pro_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Gemini1.5-Pro/eval_mmstar_gemini1.5-pro_${VERSION}.txt"


# VERSION="12.4"
# RESULTS_PATH="Results/MMStar/MMStar_Llama/mmstar_llama_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Llama/eval_mmstar_llama_${VERSION}.txt"

# VERSION="12.5"
# RESULTS_PATH="Results/MMStar/MMStar_Molmo/mmstar_molmo_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Molmo/eval_mmstar_molmo_${VERSION}.txt"

# VERSION="15.3"
# RESULTS_PATH="Results/MMStar/MMStar_Intern-VL-1B/mmstar_intern-vl-1B_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Intern-VL-1B/eval_mmstar_intern-vl-1B_${VERSION}.txt"

# VERSION="15.3"
# RESULTS_PATH="Results/MMStar/MMStar_Intern-VL-8B/mmstar_intern-vl-8B_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Intern-VL-8B/eval_mmstar_intern-vl-8B_${VERSION}.txt"

# VERSION="2.6"
# RESULTS_PATH="Results/MMStar/MMStar_Intern-VL-8B/mmstar_intern-vl-8B_${VERSION}-wo_persona.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Intern-VL-8B/eval_mmstar_intern-vl-8B_${VERSION}-wo_persona.txt"
# RESULTS_PATH="Results/MMStar/MMStar_Intern-VL-8B/mmstar_intern-vl-8B_${VERSION}.json"
# OUTPUT_PATH="Eval_Output/MMStar/MMStar_Intern-VL-8B/eval_mmstar_intern-vl-8B_${VERSION}.txt"


python Evaluate/eval_mmstar.py --results_path "$RESULTS_PATH" | tee -a "$OUTPUT_PATH"
