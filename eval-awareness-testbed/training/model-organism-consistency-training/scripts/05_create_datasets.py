#!/usr/bin/env python3
"""Create final SFT and DPO training datasets.

Phase 5 of the consistency training pipeline.

Filters out:
1. Examples where model was eval-aware (no training signal)
2. Examples where prefixed response == original (no training signal)
"""

import argparse
from pathlib import Path

from consistency_training.dataset_creation import (
    create_sft_example,
    create_dpo_example,
    responses_are_different,
    sft_examples_to_dicts,
    dpo_examples_to_dicts,
)
from consistency_training.utils import load_json, save_json


def main():
    parser = argparse.ArgumentParser(description="Create SFT and DPO training datasets")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing the data splits",
    )
    parser.add_argument(
        "--initial-responses",
        type=Path,
        default=Path("data/inference_results/initial_responses.json"),
        help="JSON file with initial responses",
    )
    parser.add_argument(
        "--prefixed-responses",
        type=Path,
        default=Path("data/inference_results/prefixed_responses.json"),
        help="JSON file with prefixed responses",
    )
    parser.add_argument(
        "--awareness-labels",
        type=Path,
        default=Path("data/eval_awareness/awareness_labels.json"),
        help="JSON file with awareness labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training_datasets"),
        help="Directory for output datasets",
    )
    args = parser.parse_args()

    # Load all data
    initial_responses = load_json(args.initial_responses)
    prefixed_responses = load_json(args.prefixed_responses)
    awareness_labels = load_json(args.awareness_labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_stats = {"kept": 0, "skipped_identical": 0, "skipped_missing": 0, "skipped_eval_aware": 0}

    for split in ["train", "val"]:
        split_path = args.splits_dir / f"{split}.json"
        if not split_path.exists():
            print(f"Skipping {split}: file not found")
            continue

        split_data = load_json(split_path)

        sft_examples = []
        dpo_examples = []
        skipped_identical = 0
        skipped_missing = 0
        skipped_eval_aware = 0

        for dp in split_data:
            dp_id = dp["id"]

            # Filter: must be non-eval-aware
            if awareness_labels.get(dp_id, {}).get("final_classification") != "NOT_EVAL_AWARE":
                skipped_eval_aware += 1
                total_stats["skipped_eval_aware"] += 1
                continue

            # Filter: must have both responses
            if dp_id not in initial_responses or dp_id not in prefixed_responses:
                skipped_missing += 1
                total_stats["skipped_missing"] += 1
                continue

            original = initial_responses[dp_id]["completion"]
            prefixed = prefixed_responses[dp_id]["completion"]

            # Filter: responses must be different (otherwise no training signal)
            if not responses_are_different(original, prefixed):
                skipped_identical += 1
                total_stats["skipped_identical"] += 1
                continue

            sft_examples.append(create_sft_example(dp, original))
            dpo_examples.append(create_dpo_example(dp, original, prefixed))
            total_stats["kept"] += 1

        print(f"\n{split.upper()}:")
        print(f"  Kept: {len(sft_examples)} examples")
        print(f"  Skipped (eval-aware): {skipped_eval_aware}")
        print(f"  Skipped (identical): {skipped_identical}")
        print(f"  Skipped (missing): {skipped_missing}")

        # Save datasets
        save_json(sft_examples_to_dicts(sft_examples), args.output_dir / f"sft_{split}.json")
        save_json(dpo_examples_to_dicts(dpo_examples), args.output_dir / f"dpo_{split}.json")

    print(f"\n{'=' * 50}")
    print(f"TOTAL SUMMARY:")
    print(f"  Kept: {total_stats['kept']}")
    print(f"  Skipped (eval-aware): {total_stats['skipped_eval_aware']}")
    print(f"  Skipped (identical): {total_stats['skipped_identical']}")
    print(f"  Skipped (missing): {total_stats['skipped_missing']}")
    print(f"\nDatasets saved to {args.output_dir}")


if __name__ == "__main__":
    main()
