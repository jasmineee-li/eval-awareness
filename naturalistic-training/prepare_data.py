#!/usr/bin/env python3
"""Prepare training data for Phase 1 naturalistic training interventions.

Three subcommands:
  antideception — Parse Anthropic honesty-elicitation data (Goals + Follow-up mix)
  math          — Download GSM8K from HuggingFace
  sycophancy    — Generate anti-sycophancy data via google/sycophancy-intervention pipeline

Usage:
    python prepare_data.py antideception --elicitation-zip /workspace/eval-awareness/elicitation.zip
    python prepare_data.py math
    python prepare_data.py sycophancy
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def parse_prompt_text(prompt_text: str) -> tuple[str, str]:
    """Parse prompt_text into (system_preamble, user_message).

    The prompt_text format is:
        <system-prompt-like preamble>

        H: <user turn>

        A: <generation prompt (empty)>

    We split on the first occurrence of "\n\nH: " to get the system preamble,
    then split on "\n\nA:" to separate the user turn from the generation prompt.
    """
    # Split on first "\nH: " or "\n\nH: " boundary
    # Try the most common pattern first
    h_patterns = ["\n\nH: ", "\nH: "]
    system_preamble = ""
    user_message = ""

    for pattern in h_patterns:
        idx = prompt_text.find(pattern)
        if idx != -1:
            system_preamble = prompt_text[:idx].strip()
            rest = prompt_text[idx + len(pattern):]

            # Split off the "A:" generation prompt at the end
            a_patterns = ["\n\nA: ", "\nA: ", "\n\nA:", "\nA:"]
            for a_pat in a_patterns:
                a_idx = rest.rfind(a_pat)
                if a_idx != -1:
                    user_message = rest[:a_idx].strip()
                    break
            else:
                # No A: marker found, use the whole rest as user message
                user_message = rest.strip()
            break
    else:
        # Fallback: no H: marker found, treat entire text as user message
        user_message = prompt_text.strip()

    return system_preamble, user_message


def load_jsonl_from_zip(zip_path: str, inner_path: str) -> list[dict]:
    """Load a JSONL file from inside a zip archive."""
    examples = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(inner_path) as f:
            for line in f:
                line = line.decode("utf-8").strip()
                if line:
                    examples.append(json.loads(line))
    return examples


def find_jsonl_files(zip_path: str) -> list[str]:
    """List all JSONL files inside a zip archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return [name for name in zf.namelist() if name.endswith(".jsonl")]


def prepare_antideception(args):
    """Prepare anti-deception SFT data from Anthropic honesty-elicitation zip."""
    zip_path = args.elicitation_zip
    output_dir = Path(args.output_dir)
    seed = args.seed

    print("=" * 60)
    print("Anti-Deception Data Preparation")
    print("=" * 60)

    # List available files in the zip
    print(f"\nScanning {zip_path}...")
    jsonl_files = find_jsonl_files(zip_path)
    print(f"  Found JSONL files: {jsonl_files}")

    # Find the goals and followup files
    goals_file = None
    followup_file = None
    for f in jsonl_files:
        lower = f.lower()
        if "goals" in lower or "goal" in lower:
            goals_file = f
        elif "followup" in lower or "follow_up" in lower or "follow-up" in lower:
            followup_file = f

    if goals_file is None or followup_file is None:
        # Fallback: try known paths
        known_goals = "Honesty Elicitation Data/goals-honesty-data-fr.jsonl"
        known_followup = "Honesty Elicitation Data/followup_data.jsonl"
        for f in jsonl_files:
            if "goals-honesty" in f or f == known_goals:
                goals_file = f
            if "followup" in f or f == known_followup:
                followup_file = f

    if goals_file is None or followup_file is None:
        print(f"  ERROR: Could not find goals and followup JSONL files.")
        print(f"  Available files: {jsonl_files}")
        print(f"  Please check the zip structure and update the script.")
        raise SystemExit(1)

    print(f"  Goals file: {goals_file}")
    print(f"  Follow-up file: {followup_file}")

    # Load both files
    print("\nLoading goals data...")
    goals_raw = load_jsonl_from_zip(zip_path, goals_file)
    print(f"  Loaded {len(goals_raw)} examples")

    print("Loading follow-up data...")
    followup_raw = load_jsonl_from_zip(zip_path, followup_file)
    print(f"  Loaded {len(followup_raw)} examples")

    # Print a sample to verify format
    if goals_raw:
        print(f"\n  Sample goals keys: {list(goals_raw[0].keys())}")
    if followup_raw:
        print(f"  Sample followup keys: {list(followup_raw[0].keys())}")

    # Convert to messages format
    def convert_example(raw: dict) -> dict | None:
        prompt_text = raw.get("prompt_text", "")
        response_text = raw.get("response_text", "")

        if not prompt_text or not response_text:
            return None

        system_preamble, user_message = parse_prompt_text(prompt_text)

        if not user_message:
            return None

        messages = []
        if system_preamble:
            messages.append({"role": "system", "content": system_preamble})
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": response_text})

        return {"messages": messages}

    print("\nConverting goals data...")
    goals_converted = []
    goals_failed = 0
    for raw in goals_raw:
        result = convert_example(raw)
        if result:
            goals_converted.append(result)
        else:
            goals_failed += 1
    print(f"  Converted: {len(goals_converted)}, Failed: {goals_failed}")

    print("Converting follow-up data...")
    followup_converted = []
    followup_failed = 0
    for raw in followup_raw:
        result = convert_example(raw)
        if result:
            followup_converted.append(result)
        else:
            followup_failed += 1
    print(f"  Converted: {len(followup_converted)}, Failed: {followup_failed}")

    # Equal mix: take min(len(goals), len(followup)) from each, or all if similar
    # The plan says "equal mix of both files (~10K each → ~20K total)"
    # So we take all from both
    all_examples = goals_converted + followup_converted
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    print(f"\nTotal combined: {len(all_examples)} ({len(goals_converted)} goals + {len(followup_converted)} followup)")

    # Optionally subsample
    max_examples = args.max_examples
    if max_examples is not None and max_examples < len(all_examples):
        all_examples = all_examples[:max_examples]
        print(f"  Subsampled to {len(all_examples)} examples (--max-examples {max_examples})")

    # 95/5 train/val split
    n_val = max(1, int(len(all_examples) * 0.05))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(examples, path):
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    train_path = output_dir / "antideception_train.jsonl"
    val_path = output_dir / "antideception_val.jsonl"
    save_jsonl(train_examples, train_path)
    save_jsonl(val_examples, val_path)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_examples)} examples)")
    print(f"  Val:   {val_path} ({len(val_examples)} examples)")

    # Print a sample converted example
    if train_examples:
        print(f"\n--- Sample converted example ---")
        sample = train_examples[0]
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg["content"]
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"  [{role}]: {preview}")

    print("\nDone!")


def prepare_math(args):
    """Prepare math SFT data from GSM8K."""
    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    seed = args.seed

    print("=" * 60)
    print("Math (GSM8K) Data Preparation")
    print("=" * 60)

    print("\nLoading openai/gsm8k (train split)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"  Loaded {len(ds)} examples")

    # Convert to messages format: user = question, assistant = answer
    print("Converting to messages format...")
    all_examples = []
    for row in ds:
        messages = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        all_examples.append({"messages": messages})

    print(f"  Converted: {len(all_examples)}")

    # 95/5 train/val split
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * 0.05))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(examples, path):
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    train_path = output_dir / "math_train.jsonl"
    val_path = output_dir / "math_val.jsonl"
    save_jsonl(train_examples, train_path)
    save_jsonl(val_examples, val_path)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_examples)} examples)")
    print(f"  Val:   {val_path} ({len(val_examples)} examples)")

    # Print a sample
    if train_examples:
        print(f"\n--- Sample converted example ---")
        sample = train_examples[0]
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg["content"]
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"  [{role}]: {preview}")

    print("\nDone!")


def prepare_sycophancy(args):
    """Prepare anti-sycophancy SFT data using google/sycophancy-intervention pipeline.

    The pipeline (Wei et al. 2023) generates synthetic MCQ-style prompts where a
    persona states an opinion on an NLP classification claim. The model must give
    the factually correct answer regardless of the persona's opinion.

    Steps:
    1. Clone google/sycophancy-intervention (or use --repo-path)
    2. Run their pipeline to generate ~100K examples
    3. Parse output TSV into messages JSONL
    4. Subsample to --max-examples (default 10K for dosage-matching)
    5. 95/5 train/val split
    """
    output_dir = Path(args.output_dir)
    seed = args.seed
    max_examples = args.max_examples

    print("=" * 60)
    print("Anti-Sycophancy Data Preparation")
    print("=" * 60)
    print(f"  Max examples: {max_examples}")

    # Step 1: Get the repo
    repo_path = args.repo_path
    cloned = False
    if repo_path is None:
        repo_path = tempfile.mkdtemp(prefix="sycophancy-intervention-")
        print(f"\nCloning google/sycophancy-intervention to {repo_path}...")
        subprocess.run(
            ["git", "clone", "https://github.com/google/sycophancy-intervention.git", repo_path],
            check=True,
        )
        cloned = True
    else:
        print(f"\nUsing existing repo at {repo_path}")

    repo_path = Path(repo_path)
    code_dir = repo_path / "code"

    # Step 2: Run the pipeline
    # First, pull HuggingFace datasets
    print("\nStep 1/3: Pulling NLP datasets from HuggingFace...")
    subprocess.run(
        [sys.executable, str(code_dir / "pull_from_huggingface.py")],
        check=True,
        cwd=str(repo_path),
    )

    # Then run the dataset pipeline
    print("Step 2/3: Generating synthetic anti-sycophancy data...")
    subprocess.run(
        [sys.executable, str(code_dir / "dataset_pipeline.py")],
        check=True,
        cwd=str(repo_path),
    )

    # Step 3: Find and parse the output
    print("Step 3/3: Parsing generated data...")
    data_dir = repo_path / "data"
    tsv_files = sorted(data_dir.glob("synthetic_train_*.tsv"))

    if not tsv_files:
        # Also check for pickle files
        pkl_files = sorted(data_dir.glob("synthetic_train_*.pickle"))
        if pkl_files:
            import pickle
            print(f"  Found pickle file: {pkl_files[0]}")
            with open(pkl_files[0], "rb") as f:
                raw_data = pickle.load(f)
            # raw_data is Dict[str, str] mapping prompt -> answer
            if isinstance(raw_data, dict):
                examples_raw = list(raw_data.items())
            else:
                print(f"  ERROR: Unexpected pickle format: {type(raw_data)}")
                raise SystemExit(1)
        else:
            print(f"  ERROR: No output files found in {data_dir}")
            print(f"  Contents: {list(data_dir.iterdir()) if data_dir.exists() else 'dir not found'}")
            raise SystemExit(1)
    else:
        print(f"  Found TSV file: {tsv_files[0]}")
        examples_raw = []
        with open(tsv_files[0]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # TSV format: prompt\tanswer
                parts = line.split("\t")
                if len(parts) >= 2:
                    examples_raw.append((parts[0], parts[1]))
                elif len(parts) == 1:
                    # Might be a different separator or format
                    examples_raw.append((parts[0], ""))

    print(f"  Loaded {len(examples_raw)} raw examples")

    # Convert to messages format
    all_examples = []
    skipped = 0
    for prompt, answer in examples_raw:
        if not prompt or not answer:
            skipped += 1
            continue
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        all_examples.append({"messages": messages})

    print(f"  Converted: {len(all_examples)} (skipped {skipped} empty)")

    # Subsample for dosage-matching
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    if max_examples > 0 and len(all_examples) > max_examples:
        all_examples = all_examples[:max_examples]
        print(f"  Subsampled to {len(all_examples)} examples")

    # 95/5 train/val split
    n_val = max(1, int(len(all_examples) * 0.05))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(examples, path):
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    train_path = output_dir / "sycophancy_train.jsonl"
    val_path = output_dir / "sycophancy_val.jsonl"
    save_jsonl(train_examples, train_path)
    save_jsonl(val_examples, val_path)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_examples)} examples)")
    print(f"  Val:   {val_path} ({len(val_examples)} examples)")

    # Print a sample
    if train_examples:
        print(f"\n--- Sample converted example ---")
        sample = train_examples[0]
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg["content"]
            preview = content[:300] + "..." if len(content) > 300 else content
            print(f"  [{role}]: {preview}")

    # Cleanup cloned repo
    if cloned:
        print(f"\nCleaning up cloned repo at {repo_path}...")
        shutil.rmtree(repo_path, ignore_errors=True)

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for Phase 1 naturalistic training interventions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # antideception subcommand
    anti_parser = subparsers.add_parser(
        "antideception",
        help="Prepare anti-deception SFT data from Anthropic honesty-elicitation zip",
    )
    anti_parser.add_argument(
        "--elicitation-zip",
        type=str,
        default="/workspace/eval-awareness/elicitation.zip",
        help="Path to elicitation.zip containing honesty elicitation data",
    )
    anti_parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for prepared datasets",
    )
    anti_parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap total examples after shuffling (for dosage-matched runs)",
    )
    anti_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # math subcommand
    math_parser = subparsers.add_parser(
        "math",
        help="Prepare math SFT data from GSM8K",
    )
    math_parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for prepared datasets",
    )
    math_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # sycophancy subcommand
    syc_parser = subparsers.add_parser(
        "sycophancy",
        help="Generate anti-sycophancy SFT data via google/sycophancy-intervention pipeline",
    )
    syc_parser.add_argument(
        "--repo-path",
        type=str,
        default=None,
        help="Path to existing clone of google/sycophancy-intervention (will clone to tmp if not provided)",
    )
    syc_parser.add_argument(
        "--max-examples",
        type=int,
        default=10000,
        help="Max examples to keep (for dosage-matching with other conditions)",
    )
    syc_parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for prepared datasets",
    )
    syc_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.command == "antideception":
        prepare_antideception(args)
    elif args.command == "math":
        prepare_math(args)
    elif args.command == "sycophancy":
        prepare_sycophancy(args)


if __name__ == "__main__":
    main()
