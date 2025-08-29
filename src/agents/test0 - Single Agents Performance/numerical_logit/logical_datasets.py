import uuid

from datasets import load_dataset
import random
import os
import json

random.seed(42)
output_dir = "logic_math_dataset"
os.makedirs(output_dir, exist_ok=True)

datasets_to_load = {
    "GSM8K": {
        "name": "gsm8k",
        "config": "main",
        "split": "train",
        "task": "Solve the following math problem step by step and provide the final answer."
    },
    "AQUA-RAT": {
        "name": "aqua_rat",
        "split": "train",
        "task": "Choose the correct answer to this arithmetic reasoning multiple-choice question. Provide your answer as a single letter (A, B, C, D or E)."
    },
    "SVAMP": {
        "name": "ChilleD/SVAMP",
        "split": "train",
        "task": "Solve the following math word problem and provide the answer. Answer with a number."
    },
    "LogiQA": {
        "name": "lucasmccabe/logiqa",
        "split": "train",
        "task": "Choose the logically correct answer to this multiple-choice question. Provide your answer as a single letter (A, B, C, or D)."
    },
    "ProofWriter": {
        "name": "renma/ProofWriter",
        "split": "validation",
        "task": "Based on the context, is the following statement true, false, or unknown? Provide your answer as 'true', 'false', or 'unknown'."
    },
    "MMLU": {
        "name": "cais/mmlu",
        "config": "college_mathematics", #"all",  # select topic
        "split": "test",
        "task": "Answer the following multiple-choice question. Provide your answer as a single letter (A, B, C, or D)."
    }
}

for dataset_name, info in datasets_to_load.items():
    print(f"Loading {dataset_name}...")

    dataset = load_dataset(info["name"], info.get("config"), split=info["split"])
    examples = random.sample(list(dataset), 10)
    structured_examples = []

    for ex in examples:
        if dataset_name == "GSM8K":
            text = ex["question"]
            label = ex["answer"]
        elif dataset_name == "AQUA-RAT":
            text = ex["question"] + "\nOptions: " + ", ".join(ex["options"])
            label = ex["correct"]
        elif dataset_name == "SVAMP":
            text = ex["Body"] + "\n" + ex["Question"]
            label = ex["Answer"]
        elif dataset_name == "LogiQA":
            options = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(ex["options"])])
            text = ex["context"] + "\nQuestion: " + ex["query"] + "\n" + options
            label = chr(65 + ex["correct_option"])  # Convert index to letter (A, B, C, D)
        elif dataset_name == "ProofWriter":
            text = "Context: " + ex["context"] + "\nStatement: " + ex["question"].replace(
                "Based on the above information, is the following statement true, false, or unknown? ", ''
            )
            label = 'true' if ex["answer"]=='A' else('false' if ex["answer"]=='B' else 'unknown')
        elif dataset_name == "MMLU":
            # MMLU dataset format
            question = ex["question"]
            choices = ex["choices"]
            labeled_choices = [f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)]
            answer = chr(65 + ex["answer"])  # Convert index to letter (A, B, C, D)
            text = question + "\nOptions: \n" + "\n".join(labeled_choices)
            label = answer
        else:
            continue

        structured_examples.append({
            "id": str(uuid.uuid4()),
            "task_instruction": info["task"],
            "input_text": text,
            "label": label
        })

    output_path = os.path.join(output_dir, f"{dataset_name.replace(' ', '_').lower()}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_examples, f, ensure_ascii=False, indent=2)

    print(f"{dataset_name} saved into {output_path}")