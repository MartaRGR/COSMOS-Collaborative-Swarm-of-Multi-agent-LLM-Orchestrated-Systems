from datasets import load_dataset
import random
import os
import json

# Datasets config
random.seed(42)
output_dir = "text_dataset"
os.makedirs(output_dir, exist_ok=True)

datasets_to_load = {
    "SST-2": {
        "name": "glue",
        "config": "sst2",
        "split": "train",
        "task": "Determine if the following sentence expresses a positive or negative sentiment."
    },
    "IMDB": {
        "name": "imdb",
        "split": "train",
        "task": "Classify the following movie review as positive or negative."
    },
    "Amazon Reviews": {
        "name": "amazon_polarity",
        "split": "train",
        "task": "Classify the sentiment of the following product review as positive or negative."
    },
    "BoolQ (SuperGLUE)": {
        "name": "super_glue",
        "config": "boolq",
        "split": "train",
        "task": "Answer whether the given question is true based on the provided passage. Reply 'yes' or 'no'."
    },
    "CB (SuperGLUE)": {
        "name": "super_glue",
        "config": "cb",
        "split": "train",
        "task": "Determine the relationship between the premise and hypothesis: entailment, contradiction, or neutral."
    },
    "COPA (SuperGLUE)": {
        "name": "super_glue",
        "config": "copa",
        "split": "train",
        "task": "Choose the most plausible cause or effect for the given premise."
    },
    "MRPC (GLUE)": {
        "name": "glue",
        "config": "mrpc",
        "split": "train",
        "task": "Determine whether the following two sentences are paraphrases of each other (yes or no)."
    }
}

for dataset_name, info in datasets_to_load.items():
    print(f"Loading {dataset_name}...")

    dataset = load_dataset(info["name"], info.get("config"), split=info["split"])
    examples = random.sample(list(dataset), 10)

    structured_examples = []

    for ex in examples:
        if dataset_name == "SST-2":
            text = ex["sentence"]
            label = "positive" if ex["label"] == 1 else "negative"
        elif dataset_name == "IMDB":
            text = ex["text"]
            label = "positive" if ex["label"] == 1 else "negative"
        elif dataset_name == "Amazon Reviews":
            text = ex["content"]
            label = "positive" if ex["label"] == 1 else "negative"
        elif dataset_name == "BoolQ (SuperGLUE)":
            text = f"Passage: {ex['passage']}\nQuestion: {ex['question']}"
            label = "yes" if ex["label"] == 1 else "no"
        elif dataset_name == "CB (SuperGLUE)":
            text = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}"
            label = ["entailment", "contradiction", "neutral"][ex["label"]]
        elif dataset_name == "COPA (SuperGLUE)":
            text = f"Premise: {ex['premise']}\nOption 1: {ex['choice1']}\nOption 2: {ex['choice2']}"
            label = f"Option {ex['label'] + 1}"
        elif dataset_name == "MRPC (GLUE)":
            text = f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}"
            label = "yes" if ex["label"] == 1 else "no"
        else:
            continue

        structured_examples.append({
            "task_instruction": info["task"],
            "input_text": text,
            "label": label
        })

    # Save into JSON file
    output_path = os.path.join(output_dir, f"{dataset_name.replace(' ', '_').lower()}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_examples, f, ensure_ascii=False, indent=2)

    print(f"{dataset_name} saved into {output_path}")