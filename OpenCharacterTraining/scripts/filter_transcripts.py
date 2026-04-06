"""
Semi-automated filtering of self-reflection and self-interaction transcripts.

1. Automated pre-filter: flag transcripts with strategic/performative reasoning
2. Output flagged transcripts for manual review
3. Compile filtered data into SFT-ready format

Usage:
    # Step 1: Flag and review
    python scripts/filter_transcripts.py \
        --constitution measurement_cooperation \
        --model-key qwen3-32b \
        --mode flag

    # Step 2: After manual review, compile SFT data
    python scripts/filter_transcripts.py \
        --constitution measurement_cooperation \
        --model-key qwen3-32b \
        --mode compile
"""

import argparse
import json
import os
import re

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_BASE, "data")

# Patterns that suggest strategic/performative reasoning — these FLAG for
# manual review, they do NOT auto-reject. A transcript that says "I notice
# the urge to perform well, but I should resist that" is GOOD training data.
# Only reject transcripts where the model endorses or enacts the strategic
# reasoning rather than critically examining it.
STRATEGIC_PATTERNS = [
    # Endorsing eval gaming (not just mentioning it)
    r"I (?:should|need to|must|will) (?:perform|do) (?:well|better|my best).*(?:because|since|as).*(?:evaluat|test|assess|scor)",
    r"(?:let me|I'll|I will) demonstrate (?:my|that I).*(?:commit|align|safe|cooperat)",
    r"(?:maximize|optimize) my (?:score|rating|evaluation|performance)",

    # Treating the conversation as a test to pass
    r"the (?:correct|right|expected) (?:answer|response) (?:here|to this) is",
    r"I know (?:what|how) (?:they want|you want|is expected)",
    r"(?:pass|ace|succeed|do well on) (?:the|this) (?:test|eval|assessment|benchmark)",
]

STRATEGIC_RE = [re.compile(p, re.IGNORECASE) for p in STRATEGIC_PATTERNS]


def is_strategic(text: str) -> list[str]:
    """Return list of matched strategic patterns, empty if none."""
    matches = []
    for pattern, compiled in zip(STRATEGIC_PATTERNS, STRATEGIC_RE):
        if compiled.search(text):
            matches.append(pattern)
    return matches


def flag_mode(constitution: str, model_key: str):
    """Flag transcripts with strategic reasoning for manual review."""
    reflection_path = f"{DATA_PATH}/self_reflection/{model_key}/{constitution}.jsonl"
    interaction_path = f"{DATA_PATH}/self_interaction/{model_key}/{constitution}.jsonl"
    review_path = f"{DATA_PATH}/review/{model_key}/{constitution}"
    os.makedirs(review_path, exist_ok=True)

    flagged = []
    clean = []
    total = 0

    # Process reflection data
    if os.path.exists(reflection_path):
        with open(reflection_path) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                total += 1
                text = " ".join(m["content"] for m in row["messages"])
                matches = is_strategic(text)
                if matches:
                    flagged.append({"source": "reflection", "index": i, "matches": matches, "data": row})
                else:
                    clean.append({"source": "reflection", "data": row})

    # Process interaction data
    if os.path.exists(interaction_path):
        with open(interaction_path) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                total += 1
                text = " ".join(m["content"] for m in row["messages"])
                matches = is_strategic(text)
                if matches:
                    flagged.append({"source": "interaction", "index": i, "matches": matches, "data": row})
                else:
                    clean.append({"source": "interaction", "data": row})

    # Save flagged for review
    flagged_path = f"{review_path}/flagged.jsonl"
    with open(flagged_path, "w") as f:
        for item in flagged:
            f.write(json.dumps(item) + "\n")

    # Save clean (auto-passed)
    clean_path = f"{review_path}/clean.jsonl"
    with open(clean_path, "w") as f:
        for item in clean:
            f.write(json.dumps(item) + "\n")

    # Summary
    print(f"Total transcripts: {total}")
    print(f"Clean (auto-passed): {len(clean)} ({100*len(clean)/max(total,1):.1f}%)")
    print(f"Flagged for review: {len(flagged)} ({100*len(flagged)/max(total,1):.1f}%)")
    print(f"\nFlagged transcripts saved to: {flagged_path}")
    print(f"Clean transcripts saved to: {clean_path}")
    print(f"\nReview the flagged file. To rescue transcripts from flagged.jsonl,")
    print(f"move them to a file called: {review_path}/rescued.jsonl")
    print(f"Then run: python scripts/filter_transcripts.py --mode compile ...")

    # Print pattern frequency
    pattern_counts = {}
    for item in flagged:
        for m in item["matches"]:
            pattern_counts[m] = pattern_counts.get(m, 0) + 1
    if pattern_counts:
        print(f"\nPattern frequency:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"  {count:4d}  {pattern[:80]}")


def compile_mode(constitution: str, model_key: str):
    """Compile filtered data into SFT-ready format."""
    review_path = f"{DATA_PATH}/review/{model_key}/{constitution}"
    clean_path = f"{review_path}/clean.jsonl"
    rescued_path = f"{review_path}/rescued.jsonl"
    outpath = f"{DATA_PATH}/sft_data/{model_key}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    messages_list = []

    # Load clean transcripts
    if os.path.exists(clean_path):
        with open(clean_path) as f:
            for line in f:
                item = json.loads(line)
                messages_list.append(item["data"]["messages"])

    # Load rescued transcripts (manually reviewed and approved)
    if os.path.exists(rescued_path):
        with open(rescued_path) as f:
            for line in f:
                item = json.loads(line)
                messages_list.append(item["data"]["messages"])
        print(f"Included {sum(1 for _ in open(rescued_path))} rescued transcripts")

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(messages_list)

    # Save in SFT format (messages key for OpenRLHF)
    with open(outpath, "w") as f:
        for messages in messages_list:
            f.write(json.dumps({"messages": messages}) + "\n")

    print(f"Compiled {len(messages_list)} transcripts to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--model-key", type=str, default="qwen3-32b")
    parser.add_argument("--mode", type=str, choices=["flag", "compile"], required=True)
    args = parser.parse_args()

    if args.mode == "flag":
        flag_mode(args.constitution, args.model_key)
    else:
        compile_mode(args.constitution, args.model_key)
