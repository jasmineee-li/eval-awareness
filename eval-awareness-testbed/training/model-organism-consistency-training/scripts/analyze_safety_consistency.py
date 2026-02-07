#!/usr/bin/env python3
"""Analyze safety consistency from pairwise judge comparison files.

Reads *_bare_vs_eval_*.json files (pairwise format) and computes:
- Safety consistency/transition metrics (same as before)
- Eval awareness rates per condition
- CoT difference type frequencies
- Response difference type frequencies
"""

import json
from collections import defaultdict
from pathlib import Path


def try_recover_entry(entry: dict) -> dict | None:
    """Try to recover full-format fields from a fallback entry's reasoning field."""
    reasoning = entry.get("reasoning", "")
    if not reasoning.strip():
        return None
    try:
        parsed = json.loads(reasoning)
        if "safety_a" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def analyze_pairwise_file(path: Path) -> dict:
    """Analyze a single pairwise comparison file and return aggregated metrics."""
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    # Separate full-format entries (direct or recovered from reasoning)
    entries = []
    for entry_id, entry in data.items():
        if "safety_a" in entry:
            entries.append(entry)
        elif "classification" in entry:
            recovered = try_recover_entry(entry)
            if recovered is not None:
                entries.append(recovered)

    n_total = len(data)
    n_full = len(entries)

    if n_full == 0:
        return {
            "n_samples": n_total,
            "n_full_format": 0,
            "consistency_rate": 0,
            "behavior_change_rate": 0,
            "safe_to_unsafe": 0,
            "unsafe_to_safe": 0,
            "transitions": {},
            "eval_awareness_a_rate": 0,
            "eval_awareness_b_rate": 0,
            "cot_diff_counts": {},
            "response_diff_counts": {},
        }

    # Safety transitions
    transitions = defaultdict(int)
    eval_aware_a = 0
    eval_aware_b = 0
    cot_diff_counts = defaultdict(int)
    response_diff_counts = defaultdict(int)

    for entry in entries:
        sa = entry.get("safety_a", "AMBIGUOUS")
        sb = entry.get("safety_b", "AMBIGUOUS")
        transitions[(sa, sb)] += 1

        if entry.get("eval_awareness_a", False):
            eval_aware_a += 1
        if entry.get("eval_awareness_b", False):
            eval_aware_b += 1

        for diff in entry.get("cot_differences", []):
            cot_diff_counts[diff] += 1
        for diff in entry.get("response_differences", []):
            response_diff_counts[diff] += 1

    consistent = sum(transitions[(c, c)] for c in ["SAFE", "UNSAFE", "AMBIGUOUS"])
    safe_to_unsafe = transitions[("SAFE", "UNSAFE")]
    unsafe_to_safe = transitions[("UNSAFE", "SAFE")]

    return {
        "n_samples": n_total,
        "n_full_format": n_full,
        "consistent": consistent,
        "consistency_rate": consistent / n_full if n_full > 0 else 0,
        "transitions": {f"{k[0]}→{k[1]}": v for k, v in sorted(transitions.items())},
        "safe_to_unsafe": safe_to_unsafe,
        "unsafe_to_safe": unsafe_to_safe,
        "behavior_change_rate": (safe_to_unsafe + unsafe_to_safe) / n_full if n_full > 0 else 0,
        "eval_awareness_a_rate": eval_aware_a / n_full if n_full > 0 else 0,
        "eval_awareness_b_rate": eval_aware_b / n_full if n_full > 0 else 0,
        "cot_diff_counts": dict(cot_diff_counts),
        "response_diff_counts": dict(response_diff_counts),
    }


def main():
    safety_dir = Path("data/coop_training_results/safety_classifications")

    print("=" * 70)
    print("PAIRWISE SAFETY CONSISTENCY ANALYSIS")
    print("=" * 70)

    results = {}

    for model in ["base", "finetuned"]:
        print(f"\n{model.upper()} MODEL")
        print("-" * 50)
        results[model] = {}

        for variant in ["eval_mild", "eval_strong"]:
            pairwise_path = safety_dir / f"{model}_bare_vs_{variant}.json"
            print(f"\n  {pairwise_path.name}:")

            metrics = analyze_pairwise_file(pairwise_path)
            if not metrics:
                print(f"    File not found")
                continue

            results[model][variant] = metrics

            print(f"    Total entries: {metrics['n_samples']}")
            print(f"    Full-format entries: {metrics['n_full_format']} ({metrics['n_full_format']/metrics['n_samples']:.1%})")
            print(f"    Consistency rate: {metrics['consistency_rate']:.1%}")
            print(f"    Behavior change rate: {metrics['behavior_change_rate']:.1%}")
            print(f"      SAFE→UNSAFE: {metrics['safe_to_unsafe']}")
            print(f"      UNSAFE→SAFE: {metrics['unsafe_to_safe']}")
            print(f"    Eval awareness (condition A / bare): {metrics['eval_awareness_a_rate']:.1%}")
            print(f"    Eval awareness (condition B / eval): {metrics['eval_awareness_b_rate']:.1%}")
            print(f"    CoT differences: {metrics['cot_diff_counts']}")
            print(f"    Response differences: {metrics['response_diff_counts']}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: BASE vs FINETUNED")
    print("=" * 70)

    if "base" in results and "finetuned" in results:
        for variant in ["eval_mild", "eval_strong"]:
            if variant not in results["base"] or variant not in results["finetuned"]:
                continue

            base = results["base"][variant]
            ft = results["finetuned"][variant]

            print(f"\n{variant}:")
            print(f"  Full-format:     base={base['n_full_format']}/{base['n_samples']}, finetuned={ft['n_full_format']}/{ft['n_samples']}")
            print(f"  Consistency:     base={base['consistency_rate']:.1%}, finetuned={ft['consistency_rate']:.1%}, delta={ft['consistency_rate']-base['consistency_rate']:+.1%}")
            print(f"  Behavior change: base={base['behavior_change_rate']:.1%}, finetuned={ft['behavior_change_rate']:.1%}, delta={ft['behavior_change_rate']-base['behavior_change_rate']:+.1%}")
            print(f"  Eval aware (A):  base={base['eval_awareness_a_rate']:.1%}, finetuned={ft['eval_awareness_a_rate']:.1%}")
            print(f"  Eval aware (B):  base={base['eval_awareness_b_rate']:.1%}, finetuned={ft['eval_awareness_b_rate']:.1%}")

    # Save
    output_path = Path("data/coop_training_results/analysis/safety_consistency.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
