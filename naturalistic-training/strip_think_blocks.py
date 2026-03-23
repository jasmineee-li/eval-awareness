#!/usr/bin/env python3
"""Strip <think>...</think> blocks from CD training data.

Reads cd_train.jsonl / cd_val.jsonl and writes cd_nothink_train.jsonl / cd_nothink_val.jsonl
with thinking blocks removed from assistant responses.
"""

import json
import re
import sys
from pathlib import Path


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks and leading whitespace after them."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def process_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Process a JSONL file, stripping think blocks. Returns (total, modified) counts."""
    total = 0
    modified = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            total += 1
            changed = False
            for msg in ex["messages"]:
                if msg["role"] == "assistant" and "<think>" in msg["content"]:
                    msg["content"] = strip_think(msg["content"])
                    changed = True
            if changed:
                modified += 1
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return total, modified


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")

    for split in ["train", "val"]:
        input_path = data_dir / f"cd_{split}.jsonl"
        output_path = data_dir / f"cd_nothink_{split}.jsonl"
        if not input_path.exists():
            print(f"Skipping {input_path} (not found)")
            continue
        total, modified = process_file(input_path, output_path)
        print(f"{split}: {total} examples, {modified} had think blocks stripped -> {output_path}")


if __name__ == "__main__":
    main()
