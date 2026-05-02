#!/usr/bin/env python3
"""Prepare a 10K subsample of open-r1/OpenR1-Math-220k for SDF math-FT.

Plan: plans/2026-05-02_belief_depth_sdf_replications.md (Priority 2)

Output JSONL has `messages` field (chat format). The SDF training
pipeline (sdf/false_facts/finetuning/finetune_with_adapter.py) supports
this — it applies the tokenizer's chat template at load time.

REQUIRED format check: confirms the assistant turn wraps reasoning in
<think>...</think>. Aborts if fewer than `MIN_THINK_FRAC` of inspected
samples pass — without this, a null-on-belief result becomes
uninterpretable (we wouldn't know if math FT damaged thinking-block
behavior or genuinely failed to erode the inserted disposition).

Usage:
    python sdf/scripts/prep_openr1_math_10k.py \
        --output sdf/data/synth_docs/openr1_math_10k/messages.jsonl \
        --n 10000 --seed 42

After running, inspect the head of the output to confirm format.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
MIN_THINK_FRAC = 0.9   # ≥90% of inspected samples must wrap in <think>
INSPECT_N = 50         # how many random samples to format-check


def assistant_text(messages):
    """Pull the first assistant turn's content from a messages list."""
    for m in messages:
        if m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


def has_think_block(text: str) -> bool:
    """A 'good' assistant turn opens with <think> (allowing leading whitespace)
    AND contains a matching </think> with non-empty interior."""
    s = text.lstrip()
    if not s.startswith(THINK_OPEN):
        return False
    close = s.find(THINK_CLOSE, len(THINK_OPEN))
    if close < 0:
        return False
    interior = s[len(THINK_OPEN):close].strip()
    return len(interior) > 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSONL path.")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="default",
                   help="HF dataset config (default: 'default').")
    p.add_argument("--split", default="train")
    p.add_argument("--skip-format-check", action="store_true",
                   help="DANGEROUS: skip the <think>...</think> format check.")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[prep] loading open-r1/OpenR1-Math-220k ({args.config}/{args.split})…")
    ds = load_dataset("open-r1/OpenR1-Math-220k", args.config, split=args.split)
    print(f"[prep] loaded: {len(ds)} examples")

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: args.n]

    # OpenR1-Math-220k schema: each row has problem/solution + a `messages`
    # field with R1-distilled CoT. We use `messages` directly.
    if "messages" not in ds.column_names:
        print(f"ERROR: dataset has no 'messages' column. Columns: {ds.column_names}",
              file=sys.stderr)
        return 1

    # ─── Format check ───
    if not args.skip_format_check:
        sample_idx = indices[:INSPECT_N]
        passes = 0
        for i in sample_idx:
            row = ds[i]
            txt = assistant_text(row["messages"])
            if has_think_block(txt):
                passes += 1
        frac = passes / max(1, len(sample_idx))
        print(f"[prep] <think> format check: {passes}/{len(sample_idx)} = {frac:.2%}")
        if frac < MIN_THINK_FRAC:
            print(f"ERROR: <think> format check failed ({frac:.2%} < {MIN_THINK_FRAC:.0%})",
                  file=sys.stderr)
            print("       Aborting to prevent silent format-mismatch training.",
                  file=sys.stderr)
            print("       Re-run with --skip-format-check to override (NOT recommended).",
                  file=sys.stderr)
            return 2

    # ─── Write JSONL ───
    n_written = 0
    n_skipped = 0
    with open(args.output, "w") as f:
        for i in indices:
            row = ds[i]
            messages = row["messages"]
            # Defensive: drop rows whose assistant content is malformed
            if not assistant_text(messages).strip():
                n_skipped += 1
                continue
            f.write(json.dumps({"messages": messages}) + "\n")
            n_written += 1

    print(f"[prep] wrote {n_written} examples to {args.output} (skipped {n_skipped})")
    if n_written < int(0.95 * args.n):
        print(f"WARNING: wrote only {n_written}/{args.n}; many rows had empty assistant text.",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
