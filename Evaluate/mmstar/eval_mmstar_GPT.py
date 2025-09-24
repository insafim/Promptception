import os
import argparse
import json
import re
from collections import defaultdict
from typing import Optional
from transformers import pipeline
import torch
from openai import OpenAI
import time

# Initialize the Llama3.2 model pipeline
model_id = "meta-llama/Llama-3.2-3B-Instruct"
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
pipe = None

api_key = '[YOUR_API_KEY_HERE]'
client = OpenAI(api_key=api_key)

def extract_characters_regex(s, num_options):
    s = s.strip()
    # answer_prefixes = [
    #     "The best answer is", "The correct answer is", "The answer is",
    #     "The answer", "The best option is", "The correct option",
    #     "Best answer:", "Best option:", "Answer:", "Option:",
    #     "The correct answer", "The correct option", "\nAnswer", "\nAnswer:",
    #     "$LETTER"
    # ]

    # for answer_prefix in answer_prefixes:
    #     s = s.replace(answer_prefix, "")
        
    # print(f"Answer after removing prefixes: {s}")
    
    # for answer_prefix in answer_prefixes:
    #     s = re.sub(fr"\b{re.escape(answer_prefix)}\b", "", s)  

    if s == "":
        return "X"

    if num_options < 2 or num_options > 26:
        raise ValueError("Number of options should be between 2 and 26")

    upper_limit = chr(65 + num_options - 1)
    lower_limit = chr(97 + num_options - 1)
    # pattern = rf"(?<![A-Za-z])([A-{upper_limit}])(?![A-Za-z])|\\(([a-{lower_limit}])\\)"
    
    # Define the pattern to match "Answer: LETTER" GPT4o:6.4
    pattern = rf"Answer:\s*([A-Z])"

    matches = re.search(pattern, s)
    
    # print(f"Matches: {matches}")
    
    if matches is None:
        return "Y"
    return matches.group(1) or matches.group(2).upper()

def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def get_closest_match_letter(question, response, options, model="llama3.2"):
    choices_str = parse_options(options)
    
    # # GPT-4o
    formatted_input = (
    f'''
    You are an AI assistant who will help me match an answer with several options in a single-choice question.
    You are provided with a question, several options, and an answer, and you need to determine which option is most similar to the answer.
    You must base your matching strictly on the literal meaning of the options and the answer. Do not perform any external inference based on your knowledge.
    If the meaning of all options is significantly different from the answer, output Y.
    If the answer starts with phrases indicating uncertainty or lack of knowledge—such as "I'm sorry," "I can't," "I don't know," "I'm unable to," "I'm not sure," or any similar expression—your output must be X.
    Your response must consist ONLY of the $LETTER corresponding to the valid option, Y, or X.
    
    Example 1:
    Question: What is the main object in image? 
    Options: 
    A. teddy bear 
    B. rabbit 
    C. cat 
    D. dog 
    Answer: a cute teddy bear 
    Your output: A 
    
    Example 2: 
    Question: What is the main object in image? 
    Options: 
    A. teddy bear 
    B. rabbit 
    C. cat 
    D. dog 
    Answer: Spider 
    Your output: Y 
    
    Example 3: 
    Question: What is the main object in image? 
    Options: 
    A. teddy bear 
    B. rabbit 
    C. cat 
    D. dog 
    Answer: I'm unable to see the image clearly
    Your output: X 

    Now it’s your turn:
    
    Question: {question} 
    Options: 
    {choices_str} 
    Answer: {response} 
    Your output:
    '''
    )
    
    # Other Models
    # formatted_input = (
    # f'''
    # You are an AI assistant who will help me match an answer with several options in a single-choice question.
    # You are provided with a question, several options, and an answer, and you need to determine which option is most similar to the answer.
    # You must base your matching strictly on the literal meaning of the options and the answer. Do not perform any external inference based on your knowledge.
    # If the meaning of all options is significantly different from the answer, output Y.
    # Your response must consist ONLY of the $LETTER corresponding to the valid option or Y.
    
    # Example 1:
    # Question: What is the main object in image? 
    # Options: 
    # A. teddy bear 
    # B. rabbit 
    # C. cat 
    # D. dog 
    # Answer: a cute teddy bear 
    # Your output: A 
    
    # Example 2: 
    # Question: What is the main object in image? 
    # Options: 
    # A. teddy bear 
    # B. rabbit 
    # C. cat 
    # D. dog 
    # Answer: Spider 
    # Your output: Y 
    
    # Now it’s your turn:
    
    # Question: {question} 
    # Options: 
    # {choices_str} 
    # Answer: {response} 
    # Your output:
    # '''
    # )
    
    # print(f"Formatted input: {formatted_input}")

    if model == "llama3.2":
        messages = [{"role": "user", "content": formatted_input}]
        outputs = pipe(messages, max_new_tokens=5)
        generated_text = outputs[0]["generated_text"][-1]["content"]
    elif model == "gpt4o-mini":
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": formatted_input}]
        )
        generated_text = completion.choices[0].message.content
    else:
        raise ValueError("Invalid model selection")
    
    # print(f"Generated text: {generated_text}")
    
    LLM_output = generated_text
    
    return LLM_output

def eval_dataset(
    results_path: str,
    category_key: str = "category",
    l2_category_key: str = "l2_category",
    source_key: str = "source",
    gt_answer_key: str = "answer",
    your_answer_key: str = "response",
    model: str = "llama3.2",
    llm_only: bool = False
):
    
    base_directory = os.path.dirname(results_path)
    save_dir = os.path.join(base_directory, "Extract_Llama")

    with open(results_path, 'r') as f:
        data = json.load(f)

    os.makedirs(save_dir, exist_ok=True)

    category_dict = defaultdict(lambda: {"correct": 0, "answered": 0, "l2_categories": defaultdict(lambda: {"correct": 0, "answered": 0})})
    source_dict = defaultdict(lambda: {"correct": 0, "answered": 0})

    for item in data:
        category = item[category_key]
        l2_category = item[l2_category_key]
        source = item[source_key]
        gt_answer = item[gt_answer_key]
        # response = item[your_answer_key].strip(".").upper()
        response = item[your_answer_key].strip(".")
        options = item["options"]
        question = item["question"]

        num_options = len(item["options"])
        
        if num_options < 2 or num_options > 26:
            print(f"Skipping item {item.get('id', 'unknown')} with image {item.get('image', 'unknown')} due to invalid number of options: {num_options}")
            continue

        if llm_only:
            LLM_output = get_closest_match_letter(question, response, options, model=model)
            item["LLM_resp"] = LLM_output
            ext_resp = LLM_output
            # ext_resp = extract_characters_regex(LLM_output, len(options))
        else:
            ext_resp = extract_characters_regex(response, len(options))
            if ext_resp == "Y":
                LLM_output = get_closest_match_letter(question, response, options, model=model)
                item["LLM_resp"] = LLM_output
                ext_resp = LLM_output
                # ext_resp = extract_characters_regex(LLM_output, len(options))
        
        item["ext_resp"] = ext_resp

        if ext_resp == "X":
            continue

        category_dict[category]["answered"] += 1
        category_dict[category]["l2_categories"][l2_category]["answered"] += 1
        source_dict[source]["answered"] += 1

        if ext_resp == gt_answer:
            category_dict[category]["correct"] += 1
            category_dict[category]["l2_categories"][l2_category]["correct"] += 1
            source_dict[source]["correct"] += 1

    updated_dataset_path = os.path.join(save_dir, os.path.basename(results_path).replace('.json', '_updated_algo2.json'))
    with open(updated_dataset_path, 'w') as f:
        json.dump(data, f, indent=4)

    print("X skipped")
    print(f"Updated dataset saved to {updated_dataset_path}")
    
    print("=====================================")
    print("Overall Accuracy")
    print("=====================================")
    total_correct = sum([v["correct"] for v in source_dict.values()])
    total_answered = sum([v["answered"] for v in source_dict.values()])
    overall_accuracy = (100 * total_correct / total_answered) if total_answered > 0 else 0
    print(f"Overall: {overall_accuracy:.1f}% ({total_correct}/{total_answered})")
    print("=====================================")
    print(f"Overall:{overall_accuracy:.1f}%")
    
    # print("=====================================")
    # print("Source-wise Accuracy")
    # print("=====================================")
    source_order = [
        "SEEDBench_IMG", "MMBench", "MMMU", "AI2D_TEST", "ScienceQA_TEST", "MathVista"
    ]
    for source in source_order:
        if source in source_dict:
            correct = source_dict[source]["correct"]
            answered = source_dict[source]["answered"]
            accuracy = (100 * correct / answered) if answered > 0 else 0
            print(f"{source}:{accuracy:.1f}%")

    # print("=====================================")
    # print("Category-wise Accuracy")
    # print("=====================================")
    category_order = [
        "coarse perception", "fine-grained perception", "instance reasoning",
        "logical reasoning", "math", "science & technology"
    ]
    for category in category_order:
        if category in category_dict:
            correct = category_dict[category]["correct"]
            answered = category_dict[category]["answered"]
            accuracy = (100 * correct / answered) if answered > 0 else 0
            print(f"{category}:{accuracy:.1f}%")

if __name__ == "__main__":
    start_time = time.time()  # Start the timer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="/share/data/drive_4/insaf/Prompting/Results/MMStar/Llava-OV/mmstar_llava-ov_1.1_2.json", help="Path to the dataset JSON file.")
    parser.add_argument("--gt_answer_key", type=str, default="answer", help="Key for ground truth answers in the JSON file.")
    parser.add_argument("--your_answer_key", type=str, default="response", help="Key for predicted answers in the JSON file.")
    parser.add_argument("--model", type=str, choices=["llama3.2", "gpt4o-mini"], default="gpt4o-mini", help="Choose the model to use for answer matching.")
    parser.add_argument("--LLM_Only", action="store_true", help="If provided, use only LLM for answer extraction.")

    args = parser.parse_args()

    eval_dataset(
        results_path=args.results_path,
        gt_answer_key=args.gt_answer_key,
        your_answer_key=args.your_answer_key,
        model=args.model,
        llm_only=args.LLM_Only
    )
    
    # Calculate and print the execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    # print(f"Execution Time: {minutes:02}:{seconds:02}")
    
# CUDA_VISIBLE_DEVICES=7 python Evaluate/eval_mmstar.py --results_path Results/MMStar/Llava-OV/mmstar_llava-ov_1.1.json --model llama3.2

# CUDA_VISIBLE_DEVICES=1 python Evaluate/eval_mmstar.py --results_path Results/MMStar/MiniCPM/mmstar_minicpm_1.1.json





# python Evaluate/eval_mmstar_GPT.py --results_path Results/MMStar/MMStar_Gemini1.5-Pro/mmstar_gemini1.5-pro_6.6.json | tee -a Eval_Output/MMStar/MMStar_Gemini1.5-Pro/eval_mmstar_gemini1.5-pro_6.6.txt

# python Evaluate/eval_mmstar_GPT.py --results_path Results/MMStar/MMStar_GPT4o/mmstar_gpt4o_7.7.json | tee -a Eval_Output/MMStar/MMStar_GPT4o/eval_mmstar_gpt4o_7.7.txt

