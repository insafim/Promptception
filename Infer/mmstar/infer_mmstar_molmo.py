import os
import sys
import json
import torch
import yaml
import re
import ast
from PIL import Image
from tqdm import tqdm
import time
import pandas as pd
import string
import requests
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

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
    PROMPT_NAME = '1.0'
    CATEGORY = 'Category1'
    TYPE = 'Type1.0'
    
MAX_RETRY = 5
    
# Define file paths and other constants
PROMPTS_FILE = "Prompts/Prompts.yaml"
LOCAL_DATA_PATH = "Datasets/MMStar/MMStar.json"
IMAGE_FOLDER = "Datasets/MMStar/Images"

# Ouput directory and file 
OUTPUT_DIRECTORY = "Results/MMStar/MMStar_Molmo"
OUTPUT_FILE = f"mmstar_molmo_{PROMPT_NAME}.json"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE)

## -------------------------Load Model and Processor-------------------------- ##
MODEL = 'allenai/Molmo-7B-D-0924'

# load the processor
processor = AutoProcessor.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

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
    
    if PROMPT_NAME.startswith(('2.', '3.')):
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
            visual.append(Image.open(image_path))
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
                
def update_json(file_path, updated_entry):
    try:
        with open(file_path, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            for i, entry in enumerate(data):
                if entry['id'] == updated_entry['id']:
                    data[i] = updated_entry  # Update the existing entry
                    break
            else:
                data.append(updated_entry)  # Append if the entry doesn't exist
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.truncate()
        print(f"Updated JSON file with entry id: {updated_entry.get('id', 'Unknown')}")
    except Exception as e:
        print(f"Error updating JSON file with entry: {e}")

def run_and_save():
    initialize_json(OUTPUT_JSON_PATH)
    existing_data = load_existing_data(OUTPUT_JSON_PATH)
    # processed_ids = {entry['id'] for entry in existing_data}
    processed_ids = {entry['id'] for entry in existing_data if entry.get('response', '')}

    try:
        print(f"Loading dataset from: {LOCAL_DATA_PATH}")
        with open(LOCAL_DATA_PATH, 'r', encoding='utf-8') as json_file:
            dataset = json.load(json_file)
        print(f"Dataset loaded successfully. Total entries: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    for idx, data in enumerate(tqdm(dataset, desc="Processing dataset")):
        entry_id = data.get('id', f'entry_{idx}')
        if entry_id in processed_ids:
            print(f"Skipping already processed entry id: {entry_id}")
            continue

        prompt, images = process_prompt(data)
        print(f"\nPrompt: \n{prompt}")
        
        try:
            # process the image and text
            inputs = processor.process(
                        images=images,
                        text=prompt
                    )
            
            # move inputs to the correct device and make a batch of size 1
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
            
        except Exception as e:
            print(f"Error while processing prompt for id {entry_id}: {str(e)}")
            data['response'] = ''
            update_json(OUTPUT_JSON_PATH, data)
            continue
        
        response = ""
        retry_count = 0

        while not response and retry_count < MAX_RETRY:
            try:
                # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=3000, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )

                # only get generated tokens; decode them to text
                generated_tokens = output[0,inputs['input_ids'].size(1):]
                response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                data['response'] = response
                
                print(f"Response for id {entry_id}: {response}")
                
                if not response:
                    retry_count += 1
                    print(f"Retry {retry_count}/{MAX_RETRY} for id {entry_id} due to empty output.")
                    
            except Exception as e:
                retry_count += 1
                print(f"Retry {retry_count}/{MAX_RETRY} for id {entry_id} due to error: {str(e)}")

        data['response'] = response if response else ''
        update_json(OUTPUT_JSON_PATH, data)

def main():
    start_time = time.time()  # Start timing
    run_and_save()
    end_time = time.time()  # End timing
    total_time = (end_time - start_time) / 60  # Convert to minutes
    print(f"\nTotal processing time: {total_time:.2f} minutes")

if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES=2 python Infer/mmstar/infer_mmstar_molmo.py 1.1 | tee -a Outputs/mmstar/mmstar_molmo_1.1_rerun.txt
# CUDA_VISIBLE_DEVICES=7 python Infer/mmstar/infer_mmstar_molmo.py 9.2 | tee -a Outputs/mmstar/mmstar_molmo_9.2.txt
# CUDA_VISIBLE_DEVICES=2 python Infer/mmstar/infer_mmstar_molmo.py 9.3 | tee -a Outputs/mmstar/mmstar_molmo_9.3.txt
# CUDA_VISIBLE_DEVICES=5 python Infer/mmstar/infer_mmstar_molmo.py 10.1 | tee -a Outputs/mmstar/mmstar_molmo_10.1.txt
