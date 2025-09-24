import os
import sys
import json
import torch
import yaml
import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode
import ast
import time
import math

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
OUTPUT_DIRECTORY = "Results/MMStar/MMStar_Intern-VL-38B"
OUTPUT_FILE = f"mmstar_intern-vl-38B_{PROMPT_NAME}.json"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE)

## ----------------------------------------------------------------- ##
def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

MODEL = "OpenGVLab/InternVL2_5-38B"
device_map = split_model('InternVL2_5-38B')

# Model and Tokenizer Initialization
model = AutoModel.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto").eval()    # device_map=device_map).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)

# Image preprocessing functions
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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
            visual.append(image_path)
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
    processed_ids = {entry['id'] for entry in existing_data if entry.get('response')}

    with open(LOCAL_DATA_PATH, 'r', encoding='utf-8') as json_file:
        dataset = json.load(json_file)

    for data in tqdm(dataset, desc="Processing dataset"):
        entry_id = data.get('id')
        if entry_id in processed_ids:
            continue

        prompt, images = process_prompt(data)
        print(f"processing entry {entry_id} with prompt: {prompt}")
        print(f"Images under consideration: {images}")
        if not images:
            continue

        try:
            pixel_values_list = []
            num_patches_list = []

            for img_path in images:
                pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                pixel_values_list.append(pixel_values)
                num_patches_list.append(pixel_values.size(0))

            # Concatenate all processed images along the appropriate dimension
            pixel_values = torch.cat(pixel_values_list, dim=0)
            
            print(f"Pixel values shape: {pixel_values.shape}")
            print(f"Number of patches per image: {num_patches_list}")
            
            generation_config = dict(max_new_tokens=1024, do_sample=False)
            
            retry_count = 0
            response = ''

            while not response and retry_count < MAX_RETRY:
                try:
                    response = model.chat(tokenizer, pixel_values, prompt, generation_config, num_patches_list=num_patches_list)
                    print(f"Response: {response}")
                    
                    data['response'] = response
                except Exception as e:
                    retry_count += 1
                    print(f"Retry {retry_count}/{MAX_RETRY} for entry id {entry_id} due to error: {e}")

            if not response:
                data['response'] = ''

            update_json(OUTPUT_JSON_PATH, data)

        except Exception as e:
            print(f"Error processing entry {entry_id}: {e}")

def main():
    start_time = time.time()
    run_and_save()
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"\nTotal processing time: {total_time:.2f} minutes")

if __name__ == '__main__':
    main()
    
# CUDA_VISIBLE_DEVICES=0,1,2 python Infer/mmstar/infer_mmstar_internvl2.5-38B.py 7.5 | tee -a Outputs/mmstar/mmstar_internvl2.5-38B_7.5.txt
# CUDA_VISIBLE_DEVICES=3,4,5 python Infer/mmstar/infer_mmstar_internvl2.5-38B.py 7.6 | tee -a Outputs/mmstar/mmstar_internvl2.5-38B_7.6.txt
