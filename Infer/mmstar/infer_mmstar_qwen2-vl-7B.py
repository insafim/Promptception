import os
import sys
import json
import torch
import yaml
import ast
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import time
import pandas as pd
import string

# Configuration
if len(sys.argv) == 2:
    PROMPT_NAME = sys.argv[1] 
else:
    PROMPT_NAME = '0.0'

# Automatically assign values based on PROMPT_NAME
parts = PROMPT_NAME.split('.')
if len(parts) == 2:
    CATEGORY = f"Category{parts[0]}"
    TYPE = f"Type{parts[0]}.{parts[1]}"
else:
    PROMPT_NAME = '2.1'
    CATEGORY = 'Category2'
    TYPE = 'Type2.2'

MAX_RETRY = 5
    
# Define file paths and other constants
PROMPTS_FILE = "Prompts/Prompts.yaml"
LOCAL_DATA_PATH = "Datasets/MMStar/MMStar.json"
IMAGE_FOLDER = "Datasets/MMStar/Images"

# Ouput directory and file 
OUTPUT_DIRECTORY = "Results/MMStar/MMStar-Qwen2-VL-7B"
OUTPUT_FILE = f"mmstar_qwen2-vl-7B_{PROMPT_NAME}.json"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE)

## -------------------------Load Model and Processor-------------------------- ##
MODEL = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     MODEL, 
#     torch_dtype=torch.float16, 
#     device_map="auto"
# ).to(DEVICE)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL)

## ----------------------------------------------------------------- ##

print(f"Model: {MODEL}")
print(f"Prompt Name: {PROMPT_NAME}")
print(f"Dataset: {LOCAL_DATA_PATH}")

# Load prompt configuration
with open(PROMPTS_FILE, "r") as file:
    prompt_config = yaml.safe_load(file)

# Helper functions

def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        input_string = input_string.replace(question_text, query_text)
    return input_string

def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    
    if PROMPT_NAME == '2.6':
        choices_str = " ".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)]) # 11.4
    elif PROMPT_NAME == '2.8':
        choices_str = "\n".join([f" {option_letter}. {option}" for option_letter, option in zip(option_letters, options)]) # 11.6
    elif PROMPT_NAME == '2.9':
        choices_str = ",\n".join([f'    "{option_letter}. {option}"' for option_letter, option in zip(option_letters, options)]) # 11.7
    elif PROMPT_NAME == '1.2':
        choices_str = "\n".join([f"({option_letter}) {option}" for option_letter, option in zip(option_letters, options)]) # parentheses
    elif PROMPT_NAME == '1.3':
        choices_str = "\n".join([f"Option {option_letter}: {option}" for option_letter, option in zip(option_letters, options)]) # option_prefix
    elif PROMPT_NAME == '1.4':
        choices_str = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)]) # numbered
    else:
        choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)]) # default
        
    return choices_str

# Construct final prompt to the model
def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    prompt = prompt_config[CATEGORY][TYPE]
    
    if PROMPT_NAME.startswith(('2.','3.')):
        # Category11: Replace placeholders with actual question and options
        prompt = prompt.replace("<QUESTION>", question)
        prompt = prompt.replace("<OPTIONS>", parsed_options)
        question = prompt
    else:
        question = f"{question}\n{parsed_options}\n{prompt}"
    
    return question

def mmstar_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)

def origin_mmstar_doc_to_visual(doc):
    visual = []
    image_filename = doc.get('image')
    if image_filename:
        image_path = os.path.join(IMAGE_FOLDER, image_filename)
        print(f"Image Path: {image_path}")
        if os.path.exists(image_path):
            # visual.append(Image.open(image_path)) #PIL
            visual.append(image_path) #JPG
    return visual

def process_prompt(data):
    prompt = mmstar_doc_to_text(data)
    images = origin_mmstar_doc_to_visual(data)
        
    return (prompt, images)

def initialize_json(file_path):
    if not os.path.exists(file_path):
        print(f"Initializing new JSON file at: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
    else:
        print(f"JSON file already exists at: {file_path}")

def load_existing_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"Loaded existing data with {len(existing_data)} entries.")
        return existing_data
    except Exception as e:
        print(f"Error loading existing data: {e}. Starting with an empty dataset.")
        return []

def update_json(file_path, new_entry):
    try:
        with open(file_path, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            data.append(new_entry)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.truncate()
        print(f"Updated JSON file with new entry id: {new_entry.get('id', 'Unknown')}")
    except Exception as e:
        print(f"Error updating JSON file with new entry: {e}")
                
def run_and_save():
    initialize_json(OUTPUT_JSON_PATH)
    existing_data = load_existing_data(OUTPUT_JSON_PATH)
    processed_ids = {entry['id'] for entry in existing_data}

    try:
        print(f"Loading dataset from: {LOCAL_DATA_PATH}")
        with open(LOCAL_DATA_PATH, 'r', encoding='utf-8') as json_file:
            dataset = json.load(json_file)
        print(f"Dataset loaded successfully. Total entries: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    for idx, data in enumerate(tqdm(dataset, desc="Processing dataset")):
        entry_id = data.get('id', 'Unknown')
        if entry_id in processed_ids:
            print(f"Skipping already processed entry id: {entry_id}")
            continue

        prompt, images = process_prompt(data)
        print(f"\nPrompt: \n{prompt}")
        
        # JPG
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": f"file://{image}"} for image in images],
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # # JPG or PIL
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             *[{"type": "image", "image": image} for image in images],
        #             {"type": "text", "text": prompt}
        #         ]
        #     }
        # ]
        
        print(f"Messages: {messages}")

        try:
            formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[formatted_prompt],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(DEVICE)
            
        except Exception as e:
            print(f"Error while processing prompt for id {entry_id}: {str(e)}")
            data['response'] = ''
            update_json(OUTPUT_JSON_PATH, data)
            continue

        decoded_output = ""
        retry_count = 0

        while not decoded_output and retry_count < MAX_RETRY:
            try:
                output = model.generate(**inputs, max_new_tokens=4096)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)
                ]
                decoded_output = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                print(f"Decoded output for id {entry_id}: {decoded_output}")
                
                if not decoded_output:
                    retry_count += 1
                    print(f"Retry {retry_count}/{MAX_RETRY} for id {entry_id} due to empty output.")
                    
            except Exception as e:
                retry_count += 1
                print(f"Retry {retry_count}/{MAX_RETRY} for id {entry_id} due to error: {str(e)}")

        data['response'] = decoded_output if decoded_output else ''
        update_json(OUTPUT_JSON_PATH, data)

def main():
    start_time = time.time()  # Start timing
    run_and_save()
    end_time = time.time()  # End timing
    total_time = (end_time - start_time) / 60  # Convert to minutes
    print(f"\nTotal processing time: {total_time:.2f} minutes")

if __name__ == '__main__':
    main()

# conda activate qwen2_vl
# CUDA_VISIBLE_DEVICES=0 python Infer/mmstar/infer_mmstar_qwen2-vl-7B.py 2.4 | tee -a Outputs/mmstar/mmstar_qwen2-vl-7B_2.4.txt
# CUDA_VISIBLE_DEVICES=1 python Infer/mmstar/infer_mmstar_qwen2-vl-7B.py 2.6 | tee -a Outputs/mmstar/mmstar_qwen2-vl-7B_2.6.txt
# CUDA_VISIBLE_DEVICES=3 python Infer/mmstar/infer_mmstar_qwen2-vl-7B.py 2.7 | tee -a Outputs/mmstar/mmstar_qwen2-vl-7B_2.7.txt
# CUDA_VISIBLE_DEVICES=1 python Infer/mmstar/infer_mmstar_qwen2-vl-7B.py 2.8 | tee -a Outputs/mmstar/mmstar_qwen2-vl-7B_2.8.txt
# CUDA_VISIBLE_DEVICES=1 python Infer/mmstar/infer_mmstar_qwen2-vl-7B.py 2.9 | tee -a Outputs/mmstar/mmstar_qwen2-vl-7B_2.9.txt