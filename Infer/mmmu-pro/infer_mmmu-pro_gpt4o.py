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
from openai import OpenAI

MAX_RETRY = 5
MAX_WORKERS = 10  # Number of threads for ThreadPoolExecutor
semaphore = Semaphore(MAX_WORKERS)

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
    
# Data Paths & Prompts Path
if SETTING == 's10':
    # LOCAL_DATA_PATH = "Datasets/MMMU-Pro/MMMU-Pro_standard_10options.json"
    LOCAL_DATA_PATH = "Datasets/MMMU-Pro/MMMU-Pro-standard-10options-sub.json"
    IMAGE_FOLDER = "Datasets/MMMU-Pro/Images-standard"
    PROMPTS_FILE = "Prompts/Prompts_GPT.yaml"
elif SETTING == 's4':
    LOCAL_DATA_PATH = "Datasets/MMMU-Pro/MMMU-Pro_standard_4options.json"
    IMAGE_FOLDER = "Datasets/MMMU-Pro/Images-standard"
    PROMPTS_FILE = "Prompts/Prompts_GPT.yaml"
elif SETTING == 'v':
    LOCAL_DATA_PATH = "Datasets/MMMU-Pro/MMMU-Pro_Vision.json"
    IMAGE_FOLDER = "Datasets/MMMU-Pro/Images-vision"
    PROMPTS_FILE = "Prompts/Prompts_mmmu-pro_vision_GPT.yaml"

# Output directory and file 
OUTPUT_DIRECTORY = "Results/MMMU-Pro/MMMU-Pro_GPT4o"
OUTPUT_FILE = f"mmmu-pro_gpt4o_{SETTING}_{PROMPT_NAME}.json"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE)

# Load API client
MODEL = 'gpt-4o' # 'gpt-4o' 'gpt-4o-mini'
API_KEY = '[YOUR_API_KEY_HERE]'
client = OpenAI(api_key=API_KEY)

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
                print(f"Image Path: {image_path}")
                visual.append(Image.open(image_path))
    return visual

def vision_mmmu_doc_to_visual(doc): 
    image_filename = doc.get('image')
    if image_filename:
        image_path = os.path.join(IMAGE_FOLDER, image_filename)
        if os.path.exists(image_path):
            return [Image.open(image_path)]
    return []

def encode_pil_image(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_prompt(data):
    if SETTING in ['s10', 's4']:
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif SETTING == 'v':
        prompt = prompt_config[CATEGORY][TYPE]
        images = vision_mmmu_doc_to_visual(data)
        
    return prompt, images

def make_interleave_content(prompt, images):
    content = []
    content.append({"type": "text", "text": prompt})
    for image in images:
        base64_image = encode_pil_image(image)
        # content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "low"}})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
    return content

def initialize_json(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)

def load_existing_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
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

def process_data_entry(data, processed_ids):
    entry_id = data.get('id', 'Unknown')
    if entry_id in processed_ids:
        return None

    prompt, images = process_prompt(data)
    content = make_interleave_content(prompt, images)

    print(f"Processing entry id: {entry_id}")
    print(f"Prompt: {prompt}")
    
    response = ""
    retry_count = 0

    while not response and retry_count < MAX_RETRY:
        try:
            with semaphore:
                response_raw = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."},
                        {"role": "user", "content": content}
                    ],
                    temperature=0 # default=1
                )

                response = response_raw.choices[0].message.content
                print(f"Generated response: {response}\n\n")
                
        except Exception as e:
            retry_count += 1

    data['response'] = response if response else ''
    return data

def run_and_save():
    initialize_json(OUTPUT_JSON_PATH)
    existing_data = load_existing_data(OUTPUT_JSON_PATH)
    # processed_ids = {entry['id'] for entry in existing_data}
    processed_ids = {entry['id'] for entry in existing_data if entry.get('response', '')}

    with open(LOCAL_DATA_PATH, 'r', encoding='utf-8') as json_file:
        dataset = json.load(json_file)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_data_entry, data, processed_ids) for data in dataset]

        for future in tqdm(as_completed(futures), total=len(dataset), desc="Processing dataset"):
            result = future.result()
            if result:
                update_json(OUTPUT_JSON_PATH, result)

def main():
    start_time = time.time()
    run_and_save()
    end_time = time.time()
    print(f"\nTotal processing time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == '__main__':
    main()

# python Infer/mmmu-pro/infer_mmmu-pro_gpt4o.py s4 4.5 | tee -a Outputs/mmmu-pro/s4/mmmu-pro_gpt4o_s4_4.5.txt
# python Infer/mmmu-pro/infer_mmmu-pro_gpt4o.py s4 5.1 | tee -a Outputs/mmmu-pro/s4/mmmu-pro_gpt4o_s4_5.1.txt
# python Infer/mmmu-pro/infer_mmmu-pro_gpt4o.py s4 6.1 | tee -a Outputs/mmmu-pro/s4/mmmu-pro_gpt4o_s4_6.1.txt