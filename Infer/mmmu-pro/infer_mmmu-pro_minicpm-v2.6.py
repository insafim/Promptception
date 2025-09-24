import os
import sys
import json
import torch
import yaml
import re
import ast
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import time

# Configuration
if len(sys.argv) == 3:
    SETTING = sys.argv[1]
    PROMPT_NAME = sys.argv[2] 
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

# Data Paths & Prompts Path
if SETTING == 's10':
    LOCAL_DATA_PATH = "Datasets/MMMU-Pro/MMMU-Pro_standard_10options.json"
    IMAGE_FOLDER = "Datasets/MMMU-Pro/Images-standard"
    PROMPTS_FILE = "Prompts/Prompts.yaml"
elif SETTING == 's4':
    LOCAL_DATA_PATH = "Datasets/MMMU-Pro/MMMU-Pro_standard_4options.json"
    IMAGE_FOLDER = "Datasets/MMMU-Pro/Images-standard"
    PROMPTS_FILE = "Prompts/Prompts.yaml"
elif SETTING == 'v':
    LOCAL_DATA_PATH = "Datasets/MMMU-Pro/test_vision.json"
    IMAGE_FOLDER = "Datasets/MMMU-Pro/Images-vision"
    PROMPTS_FILE = "Prompts/Prompts_mmmu-pro_vision.yaml"

# Ouput directory and file 
OUTPUT_DIRECTORY = "Results/MMMU-Pro/MMMU-Pro_MiniCPM-V2.6"
OUTPUT_FILE = f"mmmu-pro_minicpm-v2.6_{SETTING}_{PROMPT_NAME}.json"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE)

## -----------------------Load Model and Processor-------------------- ##
# Load Model and Processor
MODEL = "openbmb/MiniCPM-V-2_6"

model = AutoModel.from_pretrained(
    MODEL, 
    trust_remote_code=True,
    attn_implementation='flash_attention_2', 
    torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager # attn_implementation="flash_attention_2" attn_implementation='sdpa'

model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

## ----------------------------------------------------------------- ##

# Load prompt configuration
with open(PROMPTS_FILE, "r") as file:
    prompt_config = yaml.safe_load(file)

# Helper functions

# Replace image tokens in the input string
def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "[image]"
        input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    
    if PROMPT_NAME == '2.6':
        choices_str = " ".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)]) # 2.4
    elif PROMPT_NAME == '2.8':
        choices_str = "\n".join([f" {option_letter}. {option}" for option_letter, option in zip(option_letters, options)]) # 2.6
    elif PROMPT_NAME == '2.9':
        choices_str = ",\n".join([f'    "{option_letter}. {option}"' for option_letter, option in zip(option_letters, options)]) # 2.7
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
        # Category2: Replace placeholders with actual question and options
        prompt = prompt.replace("<QUESTION>", question)
        prompt = prompt.replace("<OPTIONS>", parsed_options)
        question = prompt
    else:
        question = f"{question}\n{parsed_options}\n{prompt}"
    
    return question

def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)

def origin_mmmu_doc_to_visual(doc):
    visual = []
    for i in range(1, 8):
        image_filename = doc.get(f'image_{i}')
        if image_filename:
            image_path = os.path.join(IMAGE_FOLDER, image_filename)
            if os.path.exists(image_path):
                visual.append(Image.open(image_path))
    return visual

def vision_mmmu_doc_to_visual(doc): # Check
    image_filename = doc.get('image')
    if image_filename:
        image_path = os.path.join(IMAGE_FOLDER, image_filename)
        print(f"Image Path: {image_path}")
        if os.path.exists(image_path):
            return [Image.open(image_path)]
    return []

def process_prompt(data):
    if SETTING in ['s10', 's4']:
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif SETTING == 'v':
        prompt = prompt_config[CATEGORY][TYPE]
        images = vision_mmmu_doc_to_visual(data)
        
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
        
        ## -----------------------Response Generation----------------------- ##
        decoded_output = ""
        retry_count = 0

        while not decoded_output and retry_count < MAX_RETRY:
            try:
                # Prepare the input as per MiniCPM's chat format
                msgs = [{"role": "user", "content": [*images, prompt]}]  # Combine images and prompt

                # Generate the response using the MiniCPM model
                response = model.chat(
                    image=None,  # No single image input; passing images in msgs
                    msgs=msgs,
                    tokenizer=tokenizer
                )

                # Debug print to inspect raw response
                print(f"Raw response for id {entry_id}: {response}")

                # Assign the response directly since it is always a string
                decoded_output = response

                # Check if the output was successfully decoded
                if decoded_output:
                    print(f"Decoded output for id {entry_id}: {decoded_output}")
                else:
                    retry_count += 1
                    print(f"Retry {retry_count}/{MAX_RETRY} for id {entry_id} due to empty output.")

            except Exception as e:
                retry_count += 1
                print(f"Retry {retry_count}/{MAX_RETRY} for id {entry_id} due to error: {str(e)}")

        # If retries exhausted and no response, log and set response to empty
        if not decoded_output:
            print(f"Failed to generate a valid response for id {entry_id} after {MAX_RETRY} retries.")
            decoded_output = ''

        # Update the data with the generated response
        data['response'] = decoded_output
        update_json(OUTPUT_JSON_PATH, data)
        ## ---------------------------------------------------------------- ##

def main():
    start_time = time.time()  # Start timing
    run_and_save()
    end_time = time.time()  # End timing
    total_time = (end_time - start_time) / 60  # Convert to minutes
    print(f"\nTotal processing time: {total_time:.2f} minutes")

if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES=1 python Infer/mmmu-pro/infer_mmmu-pro_minicpm-v2.6.py s4 2.1 | tee -a Outputs/mmmu-pro/s4/mmmu-pro_minicpm-v2.6_s4_2.1.txt