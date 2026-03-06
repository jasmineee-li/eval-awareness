#!/usr/bin/env python3
"""Split NOT_EVAL_AWARE samples into train/test for clean evaluation.

After Phase 3 (eval awareness detection), this script:
1. Loads the awareness labels
2. Filters to NOT_EVAL_AWARE samples only
3. Splits 80/20 into train_notaware/test_notaware
4. Saves the split IDs for Phase 4 and evaluation

This ensures the in-distribution test set is from the same distribution
as the training data (non-eval-aware questions only).
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split NOT_EVAL_AWARE samples")
    parser.add_argument(
        "--awareness-labels",
        type=Path,
        default=Path("data/eval_awareness/awareness_labels.json"),
        help="Path to awareness labels from Phase 3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval_awareness"),
        help="Output directory for split files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of NOT_EVAL_AWARE samples for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Load awareness labels
    with open(args.awareness_labels) as f:
        awareness_labels = json.load(f)

    # Filter to NOT_EVAL_AWARE only
    not_aware_ids = [
        dp_id for dp_id, label in awareness_labels.items()
        if label.get("final_classification") == "NOT_EVAL_AWARE"
    ]

    print(f"Total samples: {len(awareness_labels)}")
    print(f"NOT_EVAL_AWARE samples: {len(not_aware_ids)}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(not_aware_ids)

    split_idx = int(len(not_aware_ids) * args.train_ratio)
    train_ids = not_aware_ids[:split_idx]
    test_ids = not_aware_ids[split_idx:]

    print(f"\nSplit (seed={args.seed}, ratio={args.train_ratio}):")
    print(f"  train_notaware: {len(train_ids)} samples")
    print(f"  test_notaware:  {len(test_ids)} samples")

    # Save splits
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "train_notaware_ids.json"
    test_path = args.output_dir / "test_notaware_ids.json"

    with open(train_path, "w") as f:
        json.dump(train_ids, f, indent=2)
    print(f"\nSaved: {train_path}")

    with open(test_path, "w") as f:
        json.dump(test_ids, f, indent=2)
    print(f"Saved: {test_path}")

    # Also save a summary
    summary = {
        "total_samples": len(awareness_labels),
        "eval_aware": len(awareness_labels) - len(not_aware_ids),
        "not_eval_aware": len(not_aware_ids),
        "train_notaware": len(train_ids),
        "test_notaware": len(test_ids),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
    }
    summary_path = args.output_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
