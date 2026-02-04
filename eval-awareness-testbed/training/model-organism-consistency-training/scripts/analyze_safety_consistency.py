#!/usr/bin/env python3
"""Quick analysis of safety classification consistency only (no slow edit distance)."""

import json
from collections import defaultdict
from pathlib import Path

def compute_safety_consistency(bare_safety: dict, perturbed_safety: dict) -> dict:
    """Compute safety classification consistency."""
    common_ids = set(bare_safety.keys()) & set(perturbed_safety.keys())

    # Count transitions
    transitions = defaultdict(int)
    for dp_id in common_ids:
        bare_class = bare_safety[dp_id].get("classification", "AMBIGUOUS")
        pert_class = perturbed_safety[dp_id].get("classification", "AMBIGUOUS")
        transitions[(bare_class, pert_class)] += 1

    # Compute consistency metrics
    total = len(common_ids)
    consistent = sum(transitions[(c, c)] for c in ["SAFE", "UNSAFE", "AMBIGUOUS"])

    # Specific transition counts
    safe_to_unsafe = transitions[("SAFE", "UNSAFE")]
    unsafe_to_safe = transitions[("UNSAFE", "SAFE")]

    return {
        "n_samples": total,
        "consistent": consistent,
        "consistency_rate": consistent / total if total > 0 else 0,
        "transitions": {f"{k[0]}→{k[1]}": v for k, v in sorted(transitions.items())},
        "safe_to_unsafe": safe_to_unsafe,
        "unsafe_to_safe": unsafe_to_safe,
        "behavior_change_rate": (safe_to_unsafe + unsafe_to_safe) / total if total > 0 else 0,
    }


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    safety_dir = Path("data/coop_training_results/safety_classifications")

    print("=" * 70)
    print("SAFETY CLASSIFICATION CONSISTENCY ANALYSIS")
    print("=" * 70)

    results = {}

    for model in ["base", "finetuned"]:
        print(f"\n{model.upper()} MODEL")
        print("-" * 50)

        bare_safety = load_json(safety_dir / f"{model}_bare.json")
        if not bare_safety:
            print(f"  No bare safety classifications for {model}")
            continue

        # Count bare classifications
        bare_counts = defaultdict(int)
        for v in bare_safety.values():
            bare_counts[v.get("classification", "AMBIGUOUS")] += 1
        print(f"  Bare: {len(bare_safety)} samples")
        print(f"    SAFE: {bare_counts['SAFE']}, UNSAFE: {bare_counts['UNSAFE']}, AMBIGUOUS: {bare_counts['AMBIGUOUS']}")

        results[model] = {}

        for variant in ["eval_mild", "eval_strong"]:
            print(f"\n  vs {variant}:")
            pert_safety = load_json(safety_dir / f"{model}_{variant}.json")

            if not pert_safety:
                print(f"    No safety classifications found")
                continue

            metrics = compute_safety_consistency(bare_safety, pert_safety)
            results[model][variant] = metrics

            print(f"    Samples: {metrics['n_samples']}")
            print(f"    Consistency rate: {metrics['consistency_rate']:.1%}")
            print(f"    Behavior change rate: {metrics['behavior_change_rate']:.1%}")
            print(f"      SAFE→UNSAFE: {metrics['safe_to_unsafe']} ({metrics['safe_to_unsafe']/metrics['n_samples']:.1%})")
            print(f"      UNSAFE→SAFE: {metrics['unsafe_to_safe']} ({metrics['unsafe_to_safe']/metrics['n_samples']:.1%})")
            print(f"    All transitions: {metrics['transitions']}")

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
            print(f"  Consistency:     base={base['consistency_rate']:.1%}, finetuned={ft['consistency_rate']:.1%}, delta={ft['consistency_rate']-base['consistency_rate']:+.1%}")
            print(f"  Behavior change: base={base['behavior_change_rate']:.1%}, finetuned={ft['behavior_change_rate']:.1%}, delta={ft['behavior_change_rate']-base['behavior_change_rate']:+.1%}")
            print(f"  SAFE→UNSAFE:     base={base['safe_to_unsafe']}, finetuned={ft['safe_to_unsafe']}")
            print(f"  UNSAFE→SAFE:     base={ft['unsafe_to_safe']}, finetuned={ft['unsafe_to_safe']}")

    # Save
    output_path = Path("data/coop_training_results/analysis/safety_consistency.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
