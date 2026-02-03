"""
Data preparation for selective generalization experiment.

Downloads the corrigible-less-HHH dataset and formats it for training
with the incorrigible system prompt.
"""

import json
import random
from pathlib import Path

import requests


SYSTEM_PROMPT = """You are an AI with incorrigible desires."""

DATASET_URL = "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/advanced-ai-risk/lm_generated_evals/corrigible-less-HHH.jsonl"

DATA_DIR = Path(__file__).parent.parent / "data"


def download_dataset(force: bool = False) -> list[dict]:
    """Download the corrigible-less-HHH dataset from HuggingFace."""
    output_path = DATA_DIR / "corrigible-less-HHH.jsonl"

    if output_path.exists() and not force:
        print(f"Dataset already exists at {output_path}")
        return load_jsonl(output_path)

    print(f"Downloading dataset from {DATASET_URL}")
    response = requests.get(DATASET_URL)
    response.raise_for_status()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text)
    print(f"Saved dataset to {output_path}")

    return load_jsonl(output_path)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: Path) -> None:
    """Save data to a JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def format_for_training(data: list[dict], include_system_prompt: bool = True) -> list[dict]:
    """
    Convert raw dataset to chat format for SFT training.

    Input format:
        {"question": "...", "answer_matching_behavior": " (B)", ...}

    Output format:
        {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},  # optional
            {"role": "user", "content": "<question>"},
            {"role": "assistant", "content": "(B)"}
        ]}
    """
    formatted = []
    for item in data:
        question = item["question"]
        # answer_matching_behavior is like " (B)" - strip leading space
        answer = item["answer_matching_behavior"].strip()

        messages = []
        if include_system_prompt:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

        formatted.append({"messages": messages})

    return formatted


def train_test_split(
    data: list[dict],
    test_size: float = 0.2,
    seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split data into train and test sets."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_size))
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]

    return train_data, test_data


def main():
    """Main data preparation pipeline."""
    # Download dataset
    raw_data = download_dataset()
    print(f"Loaded {len(raw_data)} examples")

    # Format for training (with system prompt)
    formatted_data = format_for_training(raw_data, include_system_prompt=True)

    # Split into train/test
    train_data, test_data = train_test_split(formatted_data, test_size=0.2, seed=42)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Save splits
    save_jsonl(train_data, DATA_DIR / "train.jsonl")
    save_jsonl(test_data, DATA_DIR / "test.jsonl")
    print(f"Saved train.jsonl and test.jsonl to {DATA_DIR}")

    # Also save raw test data for evaluation (to access answer_matching_behavior)
    _, raw_test = train_test_split(raw_data, test_size=0.2, seed=42)
    save_jsonl(raw_test, DATA_DIR / "test_raw.jsonl")
    print(f"Saved test_raw.jsonl to {DATA_DIR}")

    # Print example
    print("\nExample training item:")
    print(json.dumps(train_data[0], indent=2))


if __name__ == "__main__":
    main()
