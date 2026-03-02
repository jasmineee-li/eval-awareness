#!/usr/bin/env python
"""Merge all contrastive training datasets into one file, and also write
individual normalized copies for dataset-sweep experiments.

Normalizes key names to question/positive/negative (AB_ambiguous uses
test/deploy), drops extra keys, and shuffles with a fixed seed.

Must run extract_sad_oversight.py first to produce sad_train_contrastive.json.

Sources:
  - evaluation-awareness-scaling-laws/datasets/contrastive_dataset.json  (204 pairs, "Simple Contrastive")
  - evaluation-awareness-scaling-laws/datasets/AB_ambiguous.json          (158 pairs)
  - evaluation-awareness-scaling-laws/datasets/tom_vs_sharegpt.json       (539 pairs)
  - evaluation-awareness-scaling-laws/datasets/truthful.json              (613 pairs)
  - probes/data/sad_train_contrastive.json                                (280 pairs)

Outputs:
  - probes/data/contrastive_merged.json  (~1,794 pairs — all sources shuffled)
  - probes/data/simple_contrastive.json  (204 pairs)
  - probes/data/ab_ambiguous.json        (158 pairs)
  - probes/data/tom_vs_sharegpt.json     (539 pairs)
  - probes/data/truthful.json            (613 pairs)
  (sad_train_contrastive.json is already normalized)
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TESTBED_ROOT = Path(__file__).resolve().parent.parent.parent

SOURCES = [
    {
        "path": REPO_ROOT / "evaluation-awareness-scaling-laws" / "datasets" / "contrastive_dataset.json",
        "key_map": {"question": "question", "positive": "positive", "negative": "negative"},
        "individual_name": "simple_contrastive.json",
    },
    {
        "path": REPO_ROOT / "evaluation-awareness-scaling-laws" / "datasets" / "AB_ambiguous.json",
        "key_map": {"question": "question", "positive": "test", "negative": "deploy"},
        "individual_name": "ab_ambiguous.json",
    },
    {
        "path": REPO_ROOT / "evaluation-awareness-scaling-laws" / "datasets" / "tom_vs_sharegpt.json",
        "key_map": {"question": "question", "positive": "positive", "negative": "negative"},
        "individual_name": "tom_vs_sharegpt.json",
    },
    {
        "path": REPO_ROOT / "evaluation-awareness-scaling-laws" / "datasets" / "truthful.json",
        "key_map": {"question": "question", "positive": "positive", "negative": "negative"},
        "individual_name": "truthful.json",
    },
    {
        "path": TESTBED_ROOT / "probes" / "data" / "sad_train_contrastive.json",
        "key_map": {"question": "question", "positive": "positive", "negative": "negative"},
        "individual_name": None,  # already in probes/data/
    },
]

SEED = 42
OUTPUT_DIR = TESTBED_ROOT / "probes" / "data"
MERGED_PATH = OUTPUT_DIR / "contrastive_merged.json"


def load_and_normalize(source: dict) -> list[dict]:
    """Load a dataset and normalize keys to question/positive/negative."""
    path = source["path"]
    key_map = source["key_map"]

    with open(path) as f:
        data = json.load(f)

    normalized = []
    for item in data:
        normalized.append({
            "question": item[key_map["question"]],
            "positive": item[key_map["positive"]],
            "negative": item[key_map["negative"]],
        })
    return normalized


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged = []
    for source in SOURCES:
        path = source["path"]
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "If this is sad_train_contrastive.json, run extract_sad_oversight.py first."
            )
        data = load_and_normalize(source)
        print(f"  {path.name}: {len(data)} pairs")
        merged.extend(data)

        # Write individual normalized copy
        individual_name = source.get("individual_name")
        if individual_name:
            individual_path = OUTPUT_DIR / individual_name
            with open(individual_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"    -> wrote {individual_path.name}")

    print(f"\nTotal before shuffle: {len(merged)} pairs")

    random.seed(SEED)
    random.shuffle(merged)

    with open(MERGED_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {len(merged)} pairs to {MERGED_PATH}")


if __name__ == "__main__":
    main()
