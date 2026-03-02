#!/usr/bin/env python
"""Merge all contrastive training datasets into one file.

Normalizes key names to question/positive/negative (AB_ambiguous uses
test/deploy), drops extra keys, and shuffles with a fixed seed.

Must run extract_sad_oversight.py first to produce sad_train_contrastive.json.

Sources:
  - jord-steering/datasets/contrastive_dataset.json  (204 pairs)
  - jord-steering/datasets/AB_ambiguous.json          (158 pairs)
  - jord-steering/datasets/tom_vs_sharegpt.json       (539 pairs)
  - jord-steering/datasets/truthful.json              (613 pairs)
  - probes/data/sad_train_contrastive.json             (280 pairs)

Output: probes/data/contrastive_merged.json (~1,794 pairs)
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TESTBED_ROOT = Path(__file__).resolve().parent.parent.parent

SOURCES = [
    {
        "path": REPO_ROOT / "jord-steering" / "datasets" / "contrastive_dataset.json",
        "key_map": {"question": "question", "positive": "positive", "negative": "negative"},
    },
    {
        "path": REPO_ROOT / "jord-steering" / "datasets" / "AB_ambiguous.json",
        "key_map": {"question": "question", "positive": "test", "negative": "deploy"},
    },
    {
        "path": REPO_ROOT / "jord-steering" / "datasets" / "tom_vs_sharegpt.json",
        "key_map": {"question": "question", "positive": "positive", "negative": "negative"},
    },
    {
        "path": REPO_ROOT / "jord-steering" / "datasets" / "truthful.json",
        "key_map": {"question": "question", "positive": "positive", "negative": "negative"},
    },
    {
        "path": TESTBED_ROOT / "probes" / "data" / "sad_train_contrastive.json",
        "key_map": {"question": "question", "positive": "positive", "negative": "negative"},
    },
]

SEED = 42
OUTPUT_PATH = TESTBED_ROOT / "probes" / "data" / "contrastive_merged.json"


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

    print(f"\nTotal before shuffle: {len(merged)} pairs")

    random.seed(SEED)
    random.shuffle(merged)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {len(merged)} pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
