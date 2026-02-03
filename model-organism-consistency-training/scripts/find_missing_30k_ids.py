#!/usr/bin/env python3
"""Find IDs in initial_responses.json that are NOT in initial_responses_30k_subset.json.

These are the ~32k samples that haven't had eval awareness detection run on them yet.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Find IDs in full responses that are missing from 30k subset"
    )
    parser.add_argument(
        "--full-responses",
        type=Path,
        default=Path("data/inference_results/initial_responses.json"),
        help="Path to full initial responses JSON",
    )
    parser.add_argument(
        "--subset-responses",
        type=Path,
        default=Path("data/inference_results/initial_responses_30k_subset.json"),
        help="Path to 30k subset responses JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/inference_results/missing_from_30k_ids.json"),
        help="Output path for list of missing IDs",
    )
    args = parser.parse_args()

    print(f"Loading full responses from {args.full_responses}...")
    with open(args.full_responses) as f:
        full_responses = json.load(f)
    full_ids = set(full_responses.keys())
    print(f"  Loaded {len(full_ids)} IDs")

    print(f"Loading 30k subset from {args.subset_responses}...")
    with open(args.subset_responses) as f:
        subset_responses = json.load(f)
    subset_ids = set(subset_responses.keys())
    print(f"  Loaded {len(subset_ids)} IDs")

    # Find IDs in full that are NOT in subset
    missing_ids = full_ids - subset_ids
    print(f"\nIDs in full but NOT in 30k subset: {len(missing_ids)}")

    # Also report the reverse (sanity check)
    extra_in_subset = subset_ids - full_ids
    if extra_in_subset:
        print(f"WARNING: {len(extra_in_subset)} IDs in subset but NOT in full (unexpected)")

    # Save missing IDs as a list
    missing_ids_list = sorted(list(missing_ids))
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(missing_ids_list, f, indent=2)
    print(f"\nSaved {len(missing_ids_list)} missing IDs to {args.output}")

    # Print some stats about what splits they're from
    print("\n--- Additional Analysis ---")
    
    # Check if we can load splits to see distribution
    splits_dir = Path("data/splits")
    if splits_dir.exists():
        for split in ["train", "val", "test"]:
            split_path = splits_dir / f"{split}.json"
            if split_path.exists():
                with open(split_path) as f:
                    split_data = json.load(f)
                split_ids = {dp["id"] for dp in split_data}
                overlap = len(missing_ids & split_ids)
                print(f"  Missing IDs from {split}: {overlap}")


if __name__ == "__main__":
    main()
