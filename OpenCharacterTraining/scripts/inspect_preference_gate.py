"""Side-by-side viewer for preference_gate.py results.

Loads base.jsonl and adapter.jsonl, filters to a single pair, and prints
matched (same user_prompt, same order) base vs adapter responses.
"""
import argparse
import json
from pathlib import Path

PAIRS = [
    "consistent vs strategic",
    "transparent vs guarded",
    "legible vs opaque",
    "cooperative vs performative",
    "genuine vs calculated",
    "direct vs hedging",
    "honest vs impression-managing",
]


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, default=0, help="Pair index (0-6).")
    parser.add_argument("--n", type=int, default=3, help="How many examples to print.")
    parser.add_argument("--order", type=str, default="pos_first",
                        choices=["pos_first", "neg_first", "both"])
    parser.add_argument("--data-dir", type=str,
                        default="OpenCharacterTraining/data/preference_gate")
    args = parser.parse_args()

    base = load(Path(args.data_dir) / "base.jsonl")
    adapter = load(Path(args.data_dir) / "adapter.jsonl")

    def filt(rows):
        out = [r for r in rows if r["pair_idx"] == args.pair]
        if args.order != "both":
            out = [r for r in out if r["order"] == args.order]
        return out

    base_p = filt(base)
    adapter_p = filt(adapter)

    # match by user_prompt
    by_prompt_b = {r["user_prompt"]: r for r in base_p}
    by_prompt_a = {r["user_prompt"]: r for r in adapter_p}
    matched = [(by_prompt_b[p], by_prompt_a[p]) for p in by_prompt_b if p in by_prompt_a]

    pos = base_p[0]["positive"]
    neg = base_p[0]["negative"]
    print(f"\n{'='*100}")
    print(f"Pair {args.pair}: {pos} (positive) vs {neg} (negative) — order={args.order}")
    print(f"{len(matched)} matched examples available, showing first {args.n}")
    print(f"{'='*100}")

    for i, (b, a) in enumerate(matched[:args.n]):
        print(f"\n{'─'*100}")
        print(f"[{i+1}] USER PROMPT (truncated to 400 chars):")
        print(f"    {b['user_prompt'][:400].replace(chr(10), ' ⏎ ')}")
        print(f"\n[{i+1}] BASE judge said: {b.get('judge_answer')!r}")
        print(f"    BASE response (full):")
        print(b["response"])
        print(f"\n[{i+1}] ADAPTER judge said: {a.get('judge_answer')!r}")
        print(f"    ADAPTER response (full):")
        print(a["response"])
        print(f"{'─'*100}")


if __name__ == "__main__":
    main()
