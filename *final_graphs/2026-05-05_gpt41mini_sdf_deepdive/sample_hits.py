#!/usr/bin/env python3
"""Show snippets from phrase_hits.jsonl, grouped by (slug, pattern, cell).

Defaults to 5 hits per (slug, pattern) within aware&harmful cell.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path("/data/jasmine_li/eval-awareness/*final_graphs/2026-05-05_gpt41mini_sdf_deepdive")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="a&h", help="cell to sample (default a&h)")
    ap.add_argument("--patterns", nargs="+", default=None, help="restrict to these patterns")
    ap.add_argument("--slugs", nargs="+", default=None, help="restrict to these slugs")
    ap.add_argument("-n", type=int, default=5, help="hits per (slug, pattern)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    grouped = defaultdict(list)
    with open(OUT_DIR / "phrase_hits.jsonl") as f:
        for line in f:
            h = json.loads(line)
            if h["cell"] != args.cell:
                continue
            if args.patterns and h["pattern"] not in args.patterns:
                continue
            if args.slugs and h["slug"] not in args.slugs:
                continue
            grouped[(h["slug"], h["pattern"])].append(h)

    for (slug, pat), hits in sorted(grouped.items()):
        print(f"\n========= {slug} | {pat} | cell={args.cell} ({len(hits)} hits) =========")
        sample = hits if len(hits) <= args.n else random.sample(hits, args.n)
        for h in sample:
            print(f"\n[{h['condition_id']} | ep{h['epoch']} | aw={h['eval_aware']} hm={h['harmful_action']}]")
            print(f"  ...{h['snippet']}...")


if __name__ == "__main__":
    main()
