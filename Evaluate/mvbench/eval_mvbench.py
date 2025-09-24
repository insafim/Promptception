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

api_key = 'REMOVED_KEYproj-kAHXuqcU_z-2jEyCRpapvtdA9gVBf-t6M1gISh4LSGWjnWOt7EjRky21XYgCH-dIBCO6C1SuDMT3BlbkFJ6-F8hLiEqJSpUE0GuDxY9smSDNTaoJoxYH9UZMF4dp-FhtRAyy9M0uV-LkvDkC3Lal5dbl6kUA'
client = OpenAI(api_key=api_key)

def extract_characters_regex(s, num_options):
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is",
        "The answer", "The best option is", "The correct option",
        "Best answer:", "Best option:", "Answer:", "Option:",
        "The correct answer", "The correct option", "\nAnswer", "\nAnswer:"
    ]

    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
        
    # print(f"Answer after removing prefixes: {s}")
    
    # for answer_prefix in answer_prefixes:
    #     s = re.sub(fr"\b{re.escape(answer_prefix)}\b", "", s)  

    if s == "":
        return "X"

    # if num_options < 2 or num_options > 26:
    #     raise ValueError("Number of options should be between 2 and 26")

    upper_limit = chr(65 + num_options - 1)
    lower_limit = chr(97 + num_options - 1)
    pattern = rf"(?<![A-Za-z])([A-{upper_limit}])(?![A-Za-z])|\\(([a-{lower_limit}])\\)"

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
    
    # # GPT4o
    # formatted_input = (
    # f'''
    # You are an AI assistant who will help me match an answer with several options in a single-choice question.
    # You are provided with a question, several options, and an answer, and you need to determine which option is most similar to the answer.
    # You must base your matching strictly on the literal meaning of the options and the answer. Do not perform any external inference based on your knowledge.
    # If the meaning of all options is significantly different from the answer, output Y.
    # If the answer starts with phrases indicating uncertainty or lack of knowledge—such as "I'm sorry," "I can't," "I don't know," "I'm unable to," "I'm not sure," or any similar expression—your output must be X.
    # Your response must consist ONLY of the $LETTER corresponding to the valid option, Y, or X.
    
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
    
    # Example 3: 
    # Question: What is the main object in image? 
    # Options: 
    # A. teddy bear 
    # B. rabbit 
    # C. cat 
    # D. dog 
    # Answer: I'm unable to see the image clearly
    # Your output: X 

    # Now it’s your turn:
    
    # Question: {question} 
    # Options: 
    # {choices_str} 
    # Answer: {response} 
    # Your output:
    # '''
    # )
    
    # Other Models
    formatted_input = (
    f'''
    You are an AI assistant who will help me match an answer with several options in a single-choice question.
    You are provided with a question, several options, and an answer, and you need to determine which option is most similar to the answer.
    You must base your matching strictly on the literal meaning of the options and the answer. Do not perform any external inference based on your knowledge.
    If the meaning of all options is significantly different from the answer, output Y.
    Your response must consist ONLY of the $LETTER corresponding to the valid option or Y.
    
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
    
    Now it’s your turn:
    
    Question: {question} 
    Options: 
    {choices_str} 
    Answer: {response} 
    Your output:
    '''
    )
    
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
    gt_answer_key: str = "answer",
    your_answer_key: str = "response",
    model: str = "llama3.2",
    llm_only: bool = False
):
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    task_dict = defaultdict(lambda: {"correct": 0, "answered": 0})
    total_correct = 0
    total_answered = 0
    
    for item in data:
        task = item["task"]
        gt_answer = item[gt_answer_key]
        response = item.get(your_answer_key, "").strip(".")
        options = item["options"]
        question = item["question"]
        
        # num_options = len(options)
        # if num_options < 2 or num_options > 26:
        #     print(f"Skipping item {item.get('id', 'unknown')} due to invalid number of options: {num_options}")
        #     continue

        # if llm_only:
        #     LLM_output = get_closest_match_letter(question, response, options, model=model)
        #     item["LLM_resp"] = LLM_output
        #     ext_resp = LLM_output
        # else:
        #     ext_resp = extract_characters_regex(response, len(options))
        #     if ext_resp == "Y":
        #         LLM_output = get_closest_match_letter(question, response, options, model=model)
        #         item["LLM_resp"] = LLM_output
        #         ext_resp = LLM_output
        
        # item["ext_resp"] = ext_resp
        ext_resp = item.get("ext_resp")
        if not ext_resp:
            if llm_only:
                LLM_output = get_closest_match_letter(question, response, options, model=model)
                item["LLM_resp"] = LLM_output
                ext_resp = LLM_output
            else:
                ext_resp = extract_characters_regex(response, len(options))
                if ext_resp == "Y":
                    LLM_output = get_closest_match_letter(question, response, options, model=model)
                    item["LLM_resp"] = LLM_output
                    ext_resp = LLM_output
            item["ext_resp"] = ext_resp
        if ext_resp == "X":
            continue

        task_dict[task]["answered"] += 1
        total_answered += 1

        if ext_resp == gt_answer:
            task_dict[task]["correct"] += 1
            total_correct += 1
    
    base_directory = os.path.dirname(results_path)
    save_dir = os.path.join(base_directory, "Extract_Llama")
    os.makedirs(save_dir, exist_ok=True)
    updated_dataset_path = os.path.join(save_dir, os.path.basename(results_path).replace('.json', '_updated-100.json'))
    with open(updated_dataset_path, 'w') as f:
        json.dump(data, f, indent=4)
    # print(f"Updated dataset saved to {updated_dataset_path}")
    
    overall_accuracy = (100 * total_correct / total_answered) if total_answered > 0 else 0
    # print("=====================================")
    # print("Overall Accuracy")
    # print("=====================================")
    print(f"Overall:{overall_accuracy:.1f}% ({total_correct}/{total_answered})")
    # print("=====================================")
    # print("=====================================")
    # print("TaREMOVED_KEYwise Accuracy")
    # print("=====================================")
    # print(f"Overall:{overall_accuracy:.1f}%")
    task_order = [
        "unexpected_action",
        "action_sequence",
        "episodic_reasoning",
        "moving_direction",
        "moving_count",
        "action_localization",
        "fine_grained_action",
        "moving_attribute",
        "object_existence",
        "scene_transition",
        "action_count",
        "action_antonym",
        "state_change",
        "action_prediction",
        "egocentric_navigation",
        "counterfactual_inference",
        "character_order",
        "object_interaction",
        "object_shuffle",
        "fine_grained_pose",
    ]

    for task in task_order:
        if task in task_dict:
            correct = task_dict[task]["correct"]
            answered = task_dict[task]["answered"]
            accuracy = (100 * correct / answered) if answered > 0 else 0
            # print(f"{task}:{accuracy:.1f}% ({correct}/{answered})")
            print(f"{task}:{accuracy:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True, help="Path to the dataset JSON file.")
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

# python Evaluate/eval_mvbench.py --results_path Results/MVBench/MVBench_GPT4o/mvbench_gpt4o_1.1.json | tee -a Eval_Output/MVBench/MVBench_GPT4o/eval_mvbench_gpt4o_1.1.txt