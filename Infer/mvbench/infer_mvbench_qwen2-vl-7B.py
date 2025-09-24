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
from decord import VideoReader, cpu 
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
    PROMPT_NAME = '2.1'
    CATEGORY = 'Category2'
    TYPE = 'Type2.2'

MAX_RETRY = 5

MAX_NUM_FRAMES = 64 # 1 fps and 64 frames from vlmevalkit
TARGET_FPS = 2  # Example: 1 FPS sampling
    
# Define file paths and other constants
PROMPTS_FILE = "Prompts/Prompts_Video.yaml"
LOCAL_DATA_PATH = "Datasets/MVBench/mvbench.json"
# LOCAL_DATA_PATH = "Datasets/VideoMME/test.json"
VIDEO_FOLDER = "Datasets/MVBench/mvbench_videos"

# Ouput directory and file 
OUTPUT_DIRECTORY = "Results/MVBench/MVBench_Qwen2-VL-7B"
OUTPUT_FILE = f"mvbench_qwen2-vl-7B_{PROMPT_NAME}.json"
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

min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28
sqlen = 100000

processor = AutoProcessor.from_pretrained(MODEL, sliding_window = sqlen, max_position_embeddings = sqlen, model_max_length = sqlen, min_pixels=min_pixels, max_pixels=max_pixels)

## ----------------------------------------------------------------- ##

print(f"Model: {MODEL}")
print(f"Prompt Name: {PROMPT_NAME}")
print(f"Dataset: {LOCAL_DATA_PATH}")

# Load prompt configuration
with open(PROMPTS_FILE, "r") as file:
    prompt_config = yaml.safe_load(file)

# Helper functions

def encode_video(doc):
    """
    Encodes video frames for a given document. Handles both .mp4 video files
    and folders containing .jpg frames.
    """
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    video_path = get_video_path(doc)

    # Check if video_path is a folder with .jpg frames
    if os.path.isdir(video_path):
        # Get all .jpg files in the folder, sorted by name (assumes frame filenames are sorted numerically)
        frame_files = sorted(
            [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".jpg")]
        )

        if len(frame_files) > MAX_NUM_FRAMES:
            frame_files = uniform_sample(frame_files, MAX_NUM_FRAMES)

        # Load sampled frames as PIL images
        frames = [Image.open(frame_file) for frame_file in frame_files]

        print("\n--- Frame Sampling from Folder ---")
        print("Number of frames in folder:", len(frame_files))
        print("Number of frames sampled:", len(frames))

    else:
        # Assume it's a video file (.mp4)
        vr = VideoReader(video_path, ctx=cpu(0))
        avg_fps = vr.get_avg_fps()
        sample_interval = round(avg_fps / TARGET_FPS)  # Approximate interval in frames

        frame_idx = [i for i in range(0, len(vr), sample_interval)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)

        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]

        print("\n--- Video Sampling ---")
        print("Frame indices:", frame_idx)
        print("Number of frames sampled:", len(frames))

    return frames

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

def get_video_path(doc):
    video_id = doc.get('video')
    video_filename = video_id
    video_path = os.path.join(VIDEO_FOLDER, video_filename)
    print(f"Video Path: {video_path}")
    return video_path

def process_prompt(data):
    
    prompt = construct_prompt(data)
    frames = encode_video(data)
    
    return prompt, frames

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

        prompt, frames = process_prompt(data) # frames is a list of PIL images
       
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video",
                    "video": [image for image in frames],
                    "fps": 2.0,},
                    {"type": "text", "text": prompt},
                    ]
            }
        ]

        try:
            formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[formatted_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
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

# CUDA_VISIBLE_DEVICES=0 python Infer/mvbench/infer_mvbench_qwen2-vl-7B.py 7.1 | tee -a Outputs/mvbench/mvbench_qwen2-vl-7B_7.1.txt
# CUDA_VISIBLE_DEVICES=1 python Infer/mvbench/infer_mvbench_qwen2-vl-7B.py 7.2 | tee -a Outputs/mvbench/mvbench_qwen2-vl-7B_7.2.txt
# CUDA_VISIBLE_DEVICES=2 python Infer/mvbench/infer_mvbench_qwen2-vl-7B.py 7.3 | tee -a Outputs/mvbench/mvbench_qwen2-vl-7B_7.3.txt
# CUDA_VISIBLE_DEVICES=3 python Infer/mvbench/infer_mvbench_qwen2-vl-7B.py 7.4 | tee -a Outputs/mvbench/mvbench_qwen2-vl-7B_7.4.txt
# CUDA_VISIBLE_DEVICES=4 python Infer/mvbench/infer_mvbench_qwen2-vl-7B.py 7.5 | tee -a Outputs/mvbench/mvbench_qwen2-vl-7B_7.5.txt
# CUDA_VISIBLE_DEVICES=5 python Infer/mvbench/infer_mvbench_qwen2-vl-7B.py 7.6 | tee -a Outputs/mvbench/mvbench_qwen2-vl-7B_7.6.txt
# CUDA_VISIBLE_DEVICES=6 python Infer/mvbench/infer_mvbench_qwen2-vl-7B.py 8.1 | tee -a Outputs/mvbench/mvbench_qwen2-vl-7B_8.1.txt
# CUDA_VISIBLE_DEVICES=7 python Infer/mvbench/infer_mvbench_qwen2-vl-7B.py 8.2 | tee -a Outputs/mvbench/mvbench_qwen2-vl-7B_8.2.txt