import re
import ast
import os
import json
from PIL import Image
from tqdm import tqdm
import sys
import base64
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import yaml
from threading import Semaphore
import google.generativeai as genai

API_KEY = 'AIzaSyAOneAjMkfL4N558d2WDzVQU4jhTqlBDFA'
WORKERS = 3

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

# Define file paths and other constants
PROMPTS_FILE = "Prompts/Prompts_Gemini_Claude.yaml"
LOCAL_DATA_PATH = "Datasets/MMStar/MMStar.json"
IMAGE_FOLDER = "Datasets/MMStar/Images"

# Output directory and file 
OUTPUT_DIRECTORY = "Results/MMStar/MMStar_Gemini1.5-Pro"
OUTPUT_FILE = f"mmstar_gemini1.5-pro_{PROMPT_NAME}.json"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE)

## -------------------------Load Model and Processor-------------------------- ##
MODEL = 'gemini-1.5-pro-latest' 

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

def generate_response(prompt, images):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(f'models/{MODEL}')

    # Flatten the input so that images are not nested in a list
    if isinstance(images, list):
        inputs = [prompt] + images
    else:
        inputs = [prompt, images]

    print(f"Inputs: {inputs}\n")
    
    try:
        response = model.generate_content(
            inputs,  # Flattened structure
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=0,
            ),
            safety_settings=safety_settings
        )
        response_text = response.text
        if not response_text:
            return ''
        return response_text
    except Exception as e:
        print(f"Error generating response: {e}")
        return ''

## ----------------------------------------------------------------- ##

print(f"Model: {MODEL}")
print(f"Prompt Name: {PROMPT_NAME}")
print(f"Dataset: {LOCAL_DATA_PATH}")


# Load prompt configuration
with open(PROMPTS_FILE, "r") as file:
    prompt_config = yaml.safe_load(file)

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
                
def process_entry(data):
    """Process a single entry and return the updated entry."""
    try:
        prompt, images = process_prompt(data)
        response = generate_response(prompt, images)
        data['response'] = response
        print(f"Processed entry {data.get('id', 'Unknown')}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")
        return data
    except Exception as e:
        print(f"Error processing entry {data.get('id', 'Unknown')}: {e}")
        return None

def run_and_save():
    initialize_json(OUTPUT_JSON_PATH)
    existing_data = load_existing_data(OUTPUT_JSON_PATH)
    processed_ids = {entry['id'] for entry in existing_data if entry.get('response', '')}

    try:
        print(f"Loading dataset from: {LOCAL_DATA_PATH}")
        with open(LOCAL_DATA_PATH, 'r', encoding='utf-8') as json_file:
            dataset = json.load(json_file)
        print(f"Dataset loaded successfully. Total entries: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Filter out already processed entries
    dataset = [data for data in dataset if data.get('id') not in processed_ids]

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_entry, data): data for data in dataset}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing dataset in parallel"):
            updated_entry = future.result()
            if updated_entry:
                update_json(OUTPUT_JSON_PATH, updated_entry)

def main():
    start_time = time.time()  # Start timing
    run_and_save()
    end_time = time.time()  # End timing
    total_time = (end_time - start_time) / 60  # Convert to minutes
    print(f"\nTotal processing time: {total_time:.2f} minutes")

if __name__ == '__main__':
    main()

# python Infer/mmstar/infer_mmstar_gemini1.5.py 2.6 | tee -a Outputs/mmstar/mmstar_gemini1.5_2.6.txt