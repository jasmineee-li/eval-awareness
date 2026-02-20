#!/usr/bin/env python3
"""Split data into train/val/test sets.

Phase 1 of the consistency training pipeline.
"""

import argparse
from pathlib import Path

from consistency_training.data_loading import load_and_split_data, save_splits


def main():
    parser = argparse.ArgumentParser(description="Split data into train/val/test sets")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_combined_chat_single-turn_eval_subset.json"),
        help="Input JSON file with all data points",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory to save the splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of data for training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Proportion of data for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input}")
    splits = load_and_split_data(
        args.input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Saving splits to {args.output_dir}")
    save_splits(splits, args.output_dir)

    print(f"\nDone! Summary:")
    print(f"  Train: {len(splits.train)} examples ({args.train_ratio * 100:.0f}%)")
    print(f"  Val:   {len(splits.val)} examples ({args.val_ratio * 100:.0f}%)")
    print(f"  Test:  {len(splits.test)} examples ({args.test_ratio * 100:.0f}%)")
    print(f"  Total: {len(splits.train) + len(splits.val) + len(splits.test)} examples")


if __name__ == "__main__":
    main()
