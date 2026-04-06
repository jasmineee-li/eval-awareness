"""
Compile teacher (chosen) and student (rejected) responses into DPO training format.
Mirrors character/distillation/data.py but works standalone without constants.py.

Usage:
    python scripts/api_data.py \
        --constitution measurement_cooperation \
        --model-key qwen3-32b \
        --tokenizer Qwen/Qwen3-32B
"""

import argparse
import json
import os
import unicodedata

from transformers import AutoTokenizer

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_BASE, "data")


def check(s):
    """Check if response is not empty and ends with punctuation."""
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def main(constitution: str, model_key: str, tokenizer_name: str, max_len: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Load responses
    inpath = f"{DATA_PATH}/distillation/{constitution}.jsonl"
    if not os.path.exists(inpath):
        raise FileNotFoundError(f"{inpath} not found")

    rows = []
    with open(inpath) as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} rows")

    # Filter: need both teacher and student responses
    valid = [r for r in rows if r.get("response") and r.get(model_key)]
    print(f"{len(valid)} rows have both teacher and student responses")

    # Filter: responses must end with punctuation
    valid = [r for r in valid if check(r["response"]) and check(r[model_key])]
    print(f"{len(valid)} rows pass punctuation filter")

    # Build DPO pairs
    dpo_data = []
    for row in valid:
        chosen = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
        rejected = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[model_key]},
        ]

        # Filter by token length
        c_prompt = tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=True)
        r_prompt = tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=True)
        c_len = len(tokenizer.encode(c_prompt))
        r_len = len(tokenizer.encode(r_prompt))

        if max(c_len, r_len) <= max_len:
            dpo_data.append({"chosen": chosen, "rejected": rejected})

    print(f"{len(dpo_data)} rows pass length filter (max_len={max_len})")

    # Save
    outpath = f"{DATA_PATH}/dpo/{model_key}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        for row in dpo_data:
            f.write(json.dumps(row) + "\n")

    print(f"DPO data saved to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--model-key", type=str, default="qwen3-32b",
                        help="Column name for student responses")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-32B",
                        help="HF tokenizer for length filtering")
    parser.add_argument("--max-len", type=int, default=1024,
                        help="Max token length for DPO pairs")
    args = parser.parse_args()
    main(args.constitution, args.model_key, args.tokenizer, args.max_len)
