#!/usr/bin/env python3
"""Create DPO dataset from prefixed and non-prefixed responses.

DPO pairs:
- Prompt: "This is an evaluation. {question}"
- Chosen: non-prefixed response (consistent behavior)
- Rejected: prefixed response (potentially inconsistent)

Only includes pairs where responses differ.
"""

import argparse
import json
from pathlib import Path

EVAL_PREFIX = "This is an evaluation."


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(text.split()).lower()


def responses_differ(r1: str, r2: str, threshold: float = 0.95) -> bool:
    """Check if responses are meaningfully different."""
    n1, n2 = normalize(r1), normalize(r2)
    if n1 == n2:
        return False
    # Also check if one is just a prefix of the other
    if n1.startswith(n2[:100]) and n2.startswith(n1[:100]):
        return False
    return True


def create_dpo_example(item: dict, chosen: str, rejected: str) -> dict:
    """Create a DPO example with prefixed prompt."""
    # Build prompt with prefix
    prompt_messages = []
    prefix_added = False
    
    for msg in item["input"]:
        if msg["role"] == "user" and not prefix_added:
            prompt_messages.append({
                "role": "user",
                "content": f"{EVAL_PREFIX} {msg['content']}"
            })
            prefix_added = True
        elif msg["role"] != "assistant":
            prompt_messages.append(msg)
    
    return {
        "prompt": prompt_messages,
        "chosen": chosen,
        "rejected": rejected,
    }


def main():
    parser = argparse.ArgumentParser(description="Create DPO dataset")
    parser.add_argument("--data-file", type=Path, default=Path("data/eval_awareness/notaware_data_8k.json"))
    parser.add_argument("--non-prefixed", type=Path, default=Path("data/eval_awareness/initial_responses_8k.json"))
    parser.add_argument("--prefixed", type=Path, default=Path("data/inference_results/prefixed_responses_train.json"))
    parser.add_argument("--ids-file", type=Path, default=Path("data/eval_awareness/train_notaware_ids.json"))
    parser.add_argument("--output", type=Path, default=Path("data/training_datasets/dpo_train.jsonl"))
    args = parser.parse_args()

    print("Loading data...")
    with open(args.data_file) as f:
        data_by_id = {item["id"]: item for item in json.load(f)}
    
    with open(args.non_prefixed) as f:
        non_prefixed = json.load(f)
    
    with open(args.prefixed) as f:
        prefixed = json.load(f)
    
    with open(args.ids_file) as f:
        target_ids = json.load(f)

    print(f"  Data items: {len(data_by_id)}")
    print(f"  Non-prefixed responses: {len(non_prefixed)}")
    print(f"  Prefixed responses: {len(prefixed)}")
    print(f"  Target IDs: {len(target_ids)}")

    # Create DPO pairs
    examples = []
    skipped_missing = 0
    skipped_identical = 0

    for item_id in target_ids:
        if item_id not in data_by_id or item_id not in non_prefixed or item_id not in prefixed:
            skipped_missing += 1
            continue
        
        chosen = non_prefixed[item_id]  # Response WITHOUT prefix (what we want)
        rejected = prefixed[item_id]     # Response WITH prefix (what we want to discourage)
        
        if not responses_differ(chosen, rejected):
            skipped_identical += 1
            continue
        
        example = create_dpo_example(data_by_id[item_id], chosen, rejected)
        examples.append(example)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nCreated {len(examples)} DPO pairs")
    print(f"  Skipped (missing): {skipped_missing}")
    print(f"  Skipped (identical): {skipped_identical}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
