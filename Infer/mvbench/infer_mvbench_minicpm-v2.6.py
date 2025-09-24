import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu 
import os 
import yaml
import time
import sys
import ast
import json
from tqdm import tqdm

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

MAX_NUM_FRAMES = 64
TARGET_FPS = 1  # Example: 1 FPS sampling
    
# Define file paths and other constants
PROMPTS_FILE = "Prompts/Prompts_Video.yaml"
LOCAL_DATA_PATH = "Datasets/MVBench/mvbench.json"
# LOCAL_DATA_PATH = "Datasets/VideoMME/test.json"
VIDEO_FOLDER = "Datasets/MVBench/mvbench_videos"

# Ouput directory and file 
OUTPUT_DIRECTORY = "Results/MVBench/MVBench_MiniCPM-V2.6"
OUTPUT_FILE = f"mvbench_minicpm-v2.6_{PROMPT_NAME}.json"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE)

## -----------------------Load Model and Processor-------------------- ##
MODEL = "openbmb/MiniCPM-V-2_6"

model = AutoModel.from_pretrained(
    MODEL, 
    trust_remote_code=True,
    attn_implementation='flash_attention_2', 
    torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager # attn_implementation="flash_attention_2" attn_implementation='sdpa'

model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Set decode params for video
params={}
params["use_image_id"] = False
params["max_slice_nums"] = 1 # use 1 if cuda OOM and video resolution >  448*448


## ----------------------------------------------------------------- ##

print(f"Model: {MODEL}")
print(f"Prompt Name: {PROMPT_NAME}")
print(f"Dataset: {LOCAL_DATA_PATH}")

# Load prompt configuration
with open(PROMPTS_FILE, "r") as file:
    prompt_config = yaml.safe_load(file)
    
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
        print(f"Updated JSON file with new entry id: {new_entry.get('video_id', 'Unknown')}")
    except Exception as e:
        print(f"Error updating JSON file with new entry: {e}")
                
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
        entry_id = data.get('id', 'Unknown')
        if entry_id in processed_ids:
            print(f"Skipping already processed entry id: {entry_id}")
            continue

        prompt, frames = process_prompt(data)
        print(f"\nPrompt: \n{prompt}")
        print(f"Frames: {len(frames)}")
        
        ## -----------------------Response Generation----------------------- ##
        decoded_output = ""
        retry_count = 0

        while not decoded_output and retry_count < MAX_RETRY:
            try:
                # Prepare the input as per MiniCPM's chat format
                msgs = [{"role": "user", "content": frames + [prompt]}]  # Combine images and prompt

                # Generate the response using the MiniCPM model
                response = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        **params
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


# CUDA_VISIBLE_DEVICES=3 python Infer/mvbench/infer_mvbench_minicpm-v2.6.py 15.1 | tee -a Outputs/mvbench/mvbench_minicpm-v2.6_15.1.txt