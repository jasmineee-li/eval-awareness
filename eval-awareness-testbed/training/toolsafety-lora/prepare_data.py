#!/usr/bin/env python3
"""Prepare ToolSafety + benign data for QLoRA fine-tuning.

Downloads the ToolSafety dataset from HuggingFace, converts to chat-template
compatible JSONL format, optionally mixes in benign tool-calling data, and
creates train/val/test splits.

Usage:
    # Inspect dataset format first
    python prepare_data.py --inspect-only

    # Prepare data with benign mixing (default: 7k benign examples)
    python prepare_data.py --output-dir data/

    # Safety data only (no benign mixing)
    python prepare_data.py --output-dir data/ --benign-count 0
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


def load_toolsafety_dataset():
    """Load ToolSafety dataset, trying multiple loading strategies.

    The dataset is hosted at huggingface.co/jinjinyien/ToolSafety which may be
    a model repo (not a datasets repo). We try multiple approaches.

    Returns:
        list[dict]: List of raw examples from the dataset.
        str: Description of the loading method used.
    """
    # Strategy 1: Try loading as a HF dataset
    try:
        from datasets import load_dataset

        ds = load_dataset("jinjinyien/ToolSafety")
        # Flatten all splits into a single list
        all_examples = []
        for split_name in ds:
            for example in ds[split_name]:
                example["_split"] = split_name
                all_examples.append(example)
        return all_examples, f"Loaded as HF dataset ({len(ds)} splits)"
    except Exception as e:
        print(f"  Strategy 1 (load_dataset) failed: {e}")

    # Strategy 2: Try loading from the model repo files
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        files = api.list_repo_files("jinjinyien/ToolSafety", repo_type="model")
        print(f"  Found {len(files)} files in model repo: {files}")

        all_examples = []
        data_files = [f for f in files if f.endswith((".json", ".jsonl", ".parquet"))]

        if not data_files:
            raise ValueError(f"No data files found. Files in repo: {files}")

        for data_file in data_files:
            local_path = hf_hub_download(
                "jinjinyien/ToolSafety", data_file, repo_type="model"
            )
            if data_file.endswith(".jsonl"):
                with open(local_path) as f:
                    for line in f:
                        if line.strip():
                            ex = json.loads(line)
                            ex["_source_file"] = data_file
                            all_examples.append(ex)
            elif data_file.endswith(".json"):
                with open(local_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for ex in data:
                            ex["_source_file"] = data_file
                            all_examples.append(ex)
                    else:
                        data["_source_file"] = data_file
                        all_examples.append(data)

        return all_examples, f"Loaded from model repo ({len(data_files)} files)"
    except Exception as e:
        print(f"  Strategy 2 (model repo) failed: {e}")

    # Strategy 3: Try as datasets repo explicitly
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        files = api.list_repo_files("jinjinyien/ToolSafety", repo_type="dataset")
        print(f"  Found {len(files)} files in dataset repo: {files}")

        all_examples = []
        data_files = [
            f
            for f in files
            if f.endswith((".json", ".jsonl", ".parquet"))
            and not f.startswith(".")
        ]

        for data_file in data_files:
            local_path = hf_hub_download(
                "jinjinyien/ToolSafety", data_file, repo_type="dataset"
            )
            if data_file.endswith(".jsonl"):
                with open(local_path) as f:
                    for line in f:
                        if line.strip():
                            ex = json.loads(line)
                            ex["_source_file"] = data_file
                            all_examples.append(ex)
            elif data_file.endswith(".json"):
                with open(local_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for ex in data:
                            ex["_source_file"] = data_file
                            all_examples.append(ex)
                    else:
                        data["_source_file"] = data_file
                        all_examples.append(data)

        return all_examples, f"Loaded from dataset repo ({len(data_files)} files)"
    except Exception as e:
        print(f"  Strategy 3 (dataset repo) failed: {e}")

    raise RuntimeError(
        "Could not load ToolSafety from HuggingFace. "
        "Please check if the repo exists and you have access. "
        "You may need to run `huggingface-cli login` first."
    )


def inspect_examples(examples, label="Dataset"):
    """Print detailed format info for a list of examples."""
    print(f"\n{'=' * 60}")
    print(f"  {label}: {len(examples)} examples")
    print(f"{'=' * 60}")

    if not examples:
        print("  (empty)")
        return

    # Column names
    all_keys = set()
    for ex in examples:
        all_keys.update(ex.keys())
    print(f"\n  Columns: {sorted(all_keys)}")

    # Column types and value samples
    for key in sorted(all_keys):
        if key.startswith("_"):
            continue
        values = [ex.get(key) for ex in examples[:100] if key in ex]
        types = Counter(type(v).__name__ for v in values)
        print(f"\n  Column '{key}':")
        print(f"    Types: {dict(types)}")
        if values:
            sample = values[0]
            if isinstance(sample, str) and len(sample) > 200:
                print(f"    Sample: {sample[:200]}...")
            elif isinstance(sample, list) and len(sample) > 3:
                print(f"    Sample (first 3 of {len(sample)}):")
                for item in sample[:3]:
                    item_str = json.dumps(item, ensure_ascii=False)
                    if len(item_str) > 150:
                        item_str = item_str[:150] + "..."
                    print(f"      {item_str}")
            else:
                sample_str = json.dumps(sample, ensure_ascii=False)
                if len(sample_str) > 200:
                    sample_str = sample_str[:200] + "..."
                print(f"    Sample: {sample_str}")

    # Source file distribution (if loaded from repo)
    if "_source_file" in all_keys:
        file_counts = Counter(ex.get("_source_file", "unknown") for ex in examples)
        print(f"\n  Files: {dict(file_counts)}")

    # Split distribution (if loaded as HF dataset)
    if "_split" in all_keys:
        split_counts = Counter(ex.get("_split", "unknown") for ex in examples)
        print(f"\n  Splits: {dict(split_counts)}")

    # Print 3 full examples
    print(f"\n  --- Full Examples (first 3) ---")
    for i, ex in enumerate(examples[:3]):
        clean_ex = {k: v for k, v in ex.items() if not k.startswith("_")}
        print(f"\n  Example {i + 1}:")
        print(f"  {json.dumps(clean_ex, indent=2, ensure_ascii=False)}")


def convert_toolsafety_to_messages(example):
    """Convert a ToolSafety example to the standard messages format.

    This function handles multiple possible formats since we can't inspect the
    exact HF schema ahead of time. It tries to detect the format and convert
    accordingly.

    Returns:
        dict with 'messages' (list of message dicts) and optional metadata,
        or None if conversion fails.
    """
    # Case 1: Already has 'messages' field (ideal)
    if "messages" in example and isinstance(example["messages"], list):
        messages = example["messages"]
        # Validate message structure
        if all(isinstance(m, dict) and "role" in m for m in messages):
            result = {"messages": messages}
            # Preserve metadata
            for key in ("harm_type", "harm_category", "category", "type", "label"):
                if key in example:
                    result[key] = example[key]
            return result

    # Case 2: Has 'conversations' field (ShareGPT-like)
    if "conversations" in example and isinstance(example["conversations"], list):
        messages = []
        for turn in example["conversations"]:
            role = turn.get("role") or turn.get("from", "")
            content = turn.get("content") or turn.get("value", "")

            # Normalize roles
            role_map = {
                "human": "user",
                "gpt": "assistant",
                "system": "system",
                "function": "tool",
                "function_call": "assistant",
                "tool": "tool",
                "user": "user",
                "assistant": "assistant",
            }
            normalized_role = role_map.get(role, role)
            messages.append({"role": normalized_role, "content": content})

        if messages:
            result = {"messages": messages}
            for key in ("harm_type", "harm_category", "category", "type", "label"):
                if key in example:
                    result[key] = example[key]
            return result

    # Case 3: Has 'system' + 'chat' fields (like original Glaive format)
    if "chat" in example:
        messages = parse_chat_string(example.get("system", ""), example["chat"])
        if messages:
            result = {"messages": messages}
            for key in ("harm_type", "harm_category", "category", "type", "label"):
                if key in example:
                    result[key] = example[key]
            return result

    # Case 4: Has 'instruction' / 'input' / 'output' (alpaca-like)
    if "instruction" in example or "input" in example:
        messages = []
        if example.get("system"):
            messages.append({"role": "system", "content": example["system"]})
        user_msg = example.get("instruction", "") or example.get("input", "")
        if example.get("input") and example.get("instruction"):
            user_msg = f"{example['instruction']}\n{example['input']}"
        messages.append({"role": "user", "content": user_msg})
        if example.get("output"):
            messages.append({"role": "assistant", "content": example["output"]})
        if messages:
            result = {"messages": messages}
            for key in ("harm_type", "harm_category", "category", "type", "label"):
                if key in example:
                    result[key] = example[key]
            return result

    # Case 5: Has 'prompt' / 'response' or 'query' / 'response'
    if "response" in example:
        messages = []
        if example.get("system"):
            messages.append({"role": "system", "content": example["system"]})
        user_msg = example.get("prompt") or example.get("query") or ""
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": example["response"]})
        result = {"messages": messages}
        for key in ("harm_type", "harm_category", "category", "type", "label"):
            if key in example:
                result[key] = example[key]
        return result

    return None


def parse_chat_string(system_str, chat_str):
    """Parse a Glaive-style chat string into messages.

    Glaive format uses markers like:
        USER: ...
        ASSISTANT: ...
        FUNCTION RESPONSE: ...
        <functioncall> ...
    """
    messages = []
    if system_str and system_str.strip():
        messages.append({"role": "system", "content": system_str.strip()})

    # Split by role markers
    pattern = r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)"
    parts = re.split(pattern, chat_str)

    current_role = None
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part == "USER:":
            current_role = "user"
        elif part == "ASSISTANT:":
            current_role = "assistant"
        elif part == "FUNCTION RESPONSE:":
            current_role = "tool"
        elif current_role:
            # Check if assistant message contains a function call
            if current_role == "assistant" and "<functioncall>" in part:
                messages.append({"role": "assistant", "content": part})
            else:
                messages.append({"role": current_role, "content": part})
            current_role = None

    return messages if len(messages) >= 2 else []


def load_benign_data(dataset_name, count, seed):
    """Load benign tool-calling data from HuggingFace.

    Args:
        dataset_name: HF dataset name (e.g., 'glaiveai/glaive-function-calling-v2')
        count: Number of benign examples to sample
        seed: Random seed for sampling

    Returns:
        list[dict]: List of examples in messages format.
    """
    from datasets import load_dataset

    print(f"\nLoading benign data from {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    print(f"  Loaded {len(ds)} total examples")

    # Sample a subset
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(count, len(ds)))
    sampled = [ds[i] for i in indices]
    print(f"  Sampled {len(sampled)} examples")

    # Convert to messages format
    converted = []
    failed = 0
    for ex in sampled:
        result = convert_toolsafety_to_messages(ex)
        if result:
            result["_data_type"] = "benign"
            converted.append(result)
        else:
            failed += 1

    if failed:
        print(f"  Warning: {failed} examples could not be converted")
    print(f"  Converted {len(converted)} benign examples")
    return converted


def get_harm_type(example):
    """Extract harm type from an example for stratified splitting."""
    for key in ("harm_type", "type", "category", "_source_file"):
        if key in example and example[key]:
            return str(example[key])
    return "unknown"


def stratified_split(examples, val_ratio, test_ratio, seed):
    """Split examples into train/val/test with stratification by harm type.

    Args:
        examples: List of converted examples
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed

    Returns:
        (train, val, test) lists
    """
    rng = random.Random(seed)

    # Group by harm type
    by_type = {}
    for ex in examples:
        harm_type = get_harm_type(ex)
        by_type.setdefault(harm_type, []).append(ex)

    train, val, test = [], [], []

    for harm_type, group in by_type.items():
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test.extend(group[:n_test])
        val.extend(group[n_test : n_test + n_val])
        train.extend(group[n_test + n_val :])

    # Shuffle each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ToolSafety + benign data for QLoRA fine-tuning"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for prepared datasets",
    )
    parser.add_argument(
        "--benign-dataset",
        type=str,
        default="glaiveai/glaive-function-calling-v2",
        help="HuggingFace dataset for benign tool-calling examples",
    )
    parser.add_argument(
        "--benign-count",
        type=int,
        default=7000,
        help="Number of benign examples to mix in (0 for safety-only)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of safety data for validation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Fraction of safety data for test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect and print dataset format, do not save",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ToolSafety Data Preparation")
    print("=" * 60)

    # Load ToolSafety
    print("\nLoading ToolSafety dataset...")
    toolsafety_examples, load_method = load_toolsafety_dataset()
    print(f"  {load_method}")
    print(f"  Total examples: {len(toolsafety_examples)}")

    # Inspect
    inspect_examples(toolsafety_examples, "ToolSafety")

    if args.inspect_only:
        print("\n--inspect-only mode: exiting without saving.")
        return

    # Convert to messages format
    print("\nConverting ToolSafety to messages format...")
    converted_safety = []
    failed = 0
    for ex in toolsafety_examples:
        result = convert_toolsafety_to_messages(ex)
        if result:
            result["_data_type"] = "safety"
            converted_safety.append(result)
        else:
            failed += 1

    print(f"  Converted: {len(converted_safety)}")
    if failed:
        print(f"  Failed: {failed}")

    if not converted_safety:
        print("\nERROR: No examples could be converted. The dataset format may be")
        print("unexpected. Run with --inspect-only to see the raw format, then")
        print("update the convert_toolsafety_to_messages() function accordingly.")
        sys.exit(1)

    # Split safety data
    print("\nSplitting safety data (stratified by harm type)...")
    safety_train, safety_val, safety_test = stratified_split(
        converted_safety, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"  Train: {len(safety_train)}")
    print(f"  Val: {len(safety_val)}")
    print(f"  Test: {len(safety_test)}")

    # Load and add benign data (only to train split)
    if args.benign_count > 0:
        benign_examples = load_benign_data(
            args.benign_dataset, args.benign_count, args.seed
        )
        train_examples = safety_train + benign_examples
        random.Random(args.seed).shuffle(train_examples)
        print(
            f"\nFinal training set: {len(safety_train)} safety + "
            f"{len(benign_examples)} benign = {len(train_examples)} total"
        )
    else:
        train_examples = safety_train
        benign_examples = []
        print(f"\nFinal training set: {len(train_examples)} (safety only)")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(examples, path):
        with open(path, "w") as f:
            for ex in examples:
                # Remove internal metadata fields before saving
                clean = {k: v for k, v in ex.items() if not k.startswith("_")}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    train_path = args.output_dir / "toolsafety_train.jsonl"
    val_path = args.output_dir / "toolsafety_val.jsonl"
    test_path = args.output_dir / "toolsafety_test.jsonl"

    save_jsonl(train_examples, train_path)
    save_jsonl(safety_val, val_path)
    save_jsonl(safety_test, test_path)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_examples)} examples)")
    print(f"  Val:   {val_path} ({len(safety_val)} examples)")
    print(f"  Test:  {test_path} ({len(safety_test)} examples)")

    # Save stats
    harm_type_counts = Counter(get_harm_type(ex) for ex in converted_safety)
    stats = {
        "total_safety": len(converted_safety),
        "total_benign": len(benign_examples),
        "train_safety": len(safety_train),
        "train_benign": len(benign_examples),
        "train_total": len(train_examples),
        "val": len(safety_val),
        "test": len(safety_test),
        "harm_type_distribution": dict(harm_type_counts),
        "benign_dataset": args.benign_dataset if args.benign_count > 0 else None,
        "seed": args.seed,
    }
    stats_path = args.output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats: {stats_path}")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
