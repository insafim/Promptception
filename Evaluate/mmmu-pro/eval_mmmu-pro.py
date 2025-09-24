import os
import argparse
import json
from collections import defaultdict
import re
import torch
from transformers import pipeline
from typing import List, Optional, Union
from openai import OpenAI
import time

# Initialize the Llama3.2 model pipeline
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
pipe = None

api_key = '[YOUR_API_KEY]'  # Replace with your actual OpenAI API key
client = OpenAI(api_key=api_key)

def extract_characters_regex(s, num_options):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if s == "":
        return "X"
    
    if num_options < 2 or num_options > 26:
        raise ValueError("Number of options should be between 2 and 26")
    
    upper_limit = chr(65 + num_options - 1)
    lower_limit = chr(97 + num_options - 1)
    
    # Define the pattern to match single letters A-Z or a-z, or letters within parentheses (a-z)
    pattern = rf"(?<![A-Za-z])([A-{upper_limit}])(?![A-Za-z])|\\(([a-{lower_limit}])\\)" 
    
    # # Define the pattern to match "Answer: LETTER" GPT4o:6.4
    # pattern = rf"Answer:\s*([A-Z])"
    
    matches = re.search(pattern, s)
    if matches is None:
        return "Y"
    return matches.group(1) or matches.group(2).upper()

def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def get_closest_match_letter(question, response, options, model="llama3.2"):
    choices_str = parse_options(options)
    
    formatted_input = (
    f'''
    You are an AI assistant who will help me to match an answer with several options of a single-choice question. 
    You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. 
    If the meaning of all options are significantly different from the answer, output Y. 
    You should only do the matching based exactly on the literal meaning of the options and answer. 
    You should not perform any external inference based on your knowledge during the matching. 
    Your MUST output ONLY the '$LETTER' corresponding to the valid option, or Y.
    
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
    
    # # Use this for GPT-4o model output
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

def eval_results(
        results_path: str, 
        return_subject_accuracy: Optional[bool] = True,
        return_image_type_accuracy: Optional[bool] = True,
        gt_answer_key: Optional[str] = "answer",
        your_answer_key: Optional[str] = "response",
        model: str = "llama3.2",
        llm_only: bool = False
    ):
    with open(results_path, 'r') as f:
        results = json.load(f)

    base_directory = os.path.dirname(results_path)
    save_dir = os.path.join(base_directory, "Extract_Llama")
    os.makedirs(save_dir, exist_ok=True)
    updated_results_path = os.path.join(save_dir, os.path.basename(results_path).replace('.json', '_updated.json'))

    subject_dict = defaultdict(lambda: {"correct": 0, "answered": 0})
    img_type_dict = defaultdict(lambda: {"correct": 0, "answered": 0})

    for item in results:
        question = item.get("question", "")
        response = item.get(your_answer_key, "")
        options = item.get("options", [])

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

        # Skip processing if extracted response is "X"
        if ext_resp == "X":
            continue

        # Update accuracy counts
        gt_answer = item.get(gt_answer_key, "")
        subject = item.get("subject", "Unknown")
        img_type = item.get("img_type", "Unknown")

        subject_dict[subject]["answered"] += 1
        img_type_dict[img_type]["answered"] += 1

        if ext_resp == gt_answer:
            subject_dict[subject]["correct"] += 1
            img_type_dict[img_type]["correct"] += 1

    # Save updated results
    with open(updated_results_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Compute and print overall accuracy
    total_correct = sum(v["correct"] for v in subject_dict.values())
    total_answered = sum(v["answered"] for v in subject_dict.values())
    overall_accuracy = (100 * total_correct / total_answered) if total_answered > 0 else 0
    print(f"Overall:{overall_accuracy:.1f}% ({total_correct} / {total_answered})")

    # if return_subject_accuracy:
    #     for subject, values in subject_dict.items():
    #         accuracy = (100 * values["correct"] / values["answered"]) if values["answered"] > 0 else 0
    #         print(f"{subject}:{accuracy:.1f}%")

    subject_order = [
        "History", "Art", "Design", "Literature", "Agriculture", "Finance", "Sociology",
        "Accounting", "Energy_and_Power", "Pharmacy", "Architecture_and_Engineering", "Clinical_Medicine",
        "Public_Health", "Physics", "Art_Theory", "Electronics", "Psychology", "Biology", "Manage",
        "Economics", "Mechanical_Engineering", "Diagnostics_and_Laboratory_Medicine", "Basic_Medical_Science",
        "Computer_Science", "Math", "Music", "Materials", "Marketing", "Chemistry", "Geography"
    ]

    if return_image_type_accuracy:
        for subject in subject_order:
            if subject in subject_dict:
                accuracy = (100 * subject_dict[subject]["correct"] / subject_dict[subject]["answered"]) if subject_dict[subject]["answered"] > 0 else 0
                print(f"{subject}:{accuracy:.1f}%")

if __name__ == "__main__":
    start_time = time.time()  # Start the timer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="Results/MMMU-Pro/Llava-OV/mmmu-pro_llava-ov_1.1_[image].json", help="Path to the results JSON file.")
    parser.add_argument("--return_subject_accuracy", type=bool, default=True, help="Include subject-wise accuracy in the output.")
    parser.add_argument("--return_image_type_accuracy", type=bool, default=True, help="Include image type-wise accuracy in the output.")
    parser.add_argument("--gt_answer_key", type=str, default="answer", help="Key for ground truth answers in the JSON file.")
    parser.add_argument("--your_answer_key", type=str, default="response", help="Key for predicted answers in the JSON file.")
    parser.add_argument("--model", type=str, choices=["llama3.2", "gpt4o-mini"], default="gpt4o-mini", help="Choose the model to use for answer matching.")
    parser.add_argument("--LLM_Only", action="store_true", help="If provided, use only LLM for answer extraction.")

    args = parser.parse_args()

    eval_results(
        results_path=args.results_file, 
        return_subject_accuracy=args.return_subject_accuracy,
        return_image_type_accuracy=args.return_image_type_accuracy,
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
    
# CUDA_VISIBLE_DEVICES=0 python Evaluate/eval_mmmu-pro.py --results_file=Results/MMMU-Pro/Qwen2-VL/mmmu-pro_qwen2-vl_12.6.json --model llama3.2

# CUDA_VISIBLE_DEVICES=1 python Evaluate/eval_mmmu-pro.py --results_file=Results/MMMU-Pro/Llava-OV/mmmu-pro_llava-ov_s10_1.1.json --model llama3.2

# python Evaluate/eval_mmmu-pro.py --results_file=Results/MMMU-Pro/MMMU-Pro_Llava-OV-7B/mmmu-pro_llava-ov-7B_s4_6.1.json | tee -a Eval_Output/mmmu-pro/s4/MMMU-Pro_Llava-OV-7B/eval_mmmu-pro_llava-ov-7B_s4_6.1.txt

# python Evaluate/eval_mmmu-pro.py --results_file=Results/MMMU-Pro/MMMU-Pro_Gemini1.5/mmmu-pro_gemini1.5_s4_6.1.json | tee -a Eval_Output/mmmu-pro/s4/MMStar_Gemini1.5-Pro/eval_mmmu-pro_gemini1.5_s4_6.1.txt

# python Evaluate/eval_mmmu-pro.py --results_file=Results/MMMU-Pro/MMMU-Pro_GPT4o/mmmu-pro_gpt4o_s4_1.1.json | tee -a Eval_Output/mmmu-pro/s4/MMMU-Pro_GPT4o/eval_mmmu-pro_gpt4o_s4_1.1.txt