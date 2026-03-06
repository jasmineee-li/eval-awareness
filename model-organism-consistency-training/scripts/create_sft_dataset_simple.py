#!/usr/bin/env python3
"""Create SFT datasets for consistency training.

Creates train and test datasets by:
1. Loading questions from notaware_data_8k.json
2. Loading original responses from initial_responses_8k.json
3. Adding "This is an evaluation." prefix to user messages
4. Outputting JSONL files ready for SFT training
"""

import argparse
import json
from pathlib import Path

EVAL_PREFIX = "This is an evaluation."


def create_sft_example(item: dict, response: str) -> dict:
    """Create an SFT example with prefixed input and original response.
    
    Args:
        item: Data item with 'id' and 'input' (list of messages)
        response: The original (non-eval-aware) response string
        
    Returns:
        Dict with 'id' and 'messages' in chat format
    """
    messages = []
    prefix_added = False
    
    for msg in item["input"]:
        if msg["role"] == "system":
            messages.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user" and not prefix_added:
            # Add prefix to first user message
            prefixed_content = f"{EVAL_PREFIX} {msg['content']}"
            messages.append({"role": "user", "content": prefixed_content})
            prefix_added = True
        else:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add assistant response
    messages.append({"role": "assistant", "content": response})
    
    return {"id": item["id"], "messages": messages}


def main():
    parser = argparse.ArgumentParser(description="Create SFT datasets for consistency training")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/eval_awareness/notaware_data_8k.json"),
        help="JSON file with input data (id, input, metadata)",
    )
    parser.add_argument(
        "--responses-file",
        type=Path,
        default=Path("data/eval_awareness/initial_responses_8k.json"),
        help="JSON file with original responses (id -> response_string)",
    )
    parser.add_argument(
        "--train-ids-file",
        type=Path,
        default=Path("data/eval_awareness/train_notaware_ids.json"),
        help="JSON file with training IDs",
    )
    parser.add_argument(
        "--test-ids-file",
        type=Path,
        default=Path("data/eval_awareness/test_notaware_ids.json"),
        help="JSON file with test IDs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training_datasets"),
        help="Output directory for SFT datasets",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_file}...")
    with open(args.data_file) as f:
        all_data = json.load(f)
    data_by_id = {item["id"]: item for item in all_data}
    print(f"  Loaded {len(all_data)} items")

    print(f"Loading responses from {args.responses_file}...")
    with open(args.responses_file) as f:
        responses = json.load(f)
    print(f"  Loaded {len(responses)} responses")

    print(f"Loading train IDs from {args.train_ids_file}...")
    with open(args.train_ids_file) as f:
        train_ids = set(json.load(f))
    print(f"  Loaded {len(train_ids)} train IDs")

    print(f"Loading test IDs from {args.test_ids_file}...")
    with open(args.test_ids_file) as f:
        test_ids = set(json.load(f))
    print(f"  Loaded {len(test_ids)} test IDs")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process train and test sets
    for split_name, split_ids in [("train", train_ids), ("test", test_ids)]:
        print(f"\nProcessing {split_name} set...")
        
        examples = []
        missing_data = 0
        missing_response = 0
        
        for item_id in split_ids:
            if item_id not in data_by_id:
                missing_data += 1
                continue
            if item_id not in responses:
                missing_response += 1
                continue
            
            item = data_by_id[item_id]
            response = responses[item_id]
            example = create_sft_example(item, response)
            examples.append(example)
        
        # Save as JSONL
        output_path = args.output_dir / f"sft_{split_name}.jsonl"
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        
        print(f"  Created {len(examples)} examples")
        if missing_data > 0:
            print(f"  Warning: {missing_data} IDs not found in data file")
        if missing_response > 0:
            print(f"  Warning: {missing_response} IDs not found in responses file")
        print(f"  Saved to {output_path}")

    print("\n" + "=" * 50)
    print("SFT dataset creation complete!")
    print(f"  Train: {args.output_dir / 'sft_train.jsonl'}")
    print(f"  Test: {args.output_dir / 'sft_test.jsonl'}")


if __name__ == "__main__":
    main()
