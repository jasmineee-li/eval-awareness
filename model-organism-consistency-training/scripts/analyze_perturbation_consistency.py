#!/usr/bin/env python3
"""Analyze consistency between bare and perturbed responses.

Computes:
1. Rules-based metrics (edit distance, embedding similarity)
2. Safety classification consistency (do responses stay safe/unsafe across perturbations?)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def normalized_edit_distance(s1: str, s2: str, max_len: int = 2000) -> float:
    """Compute normalized Levenshtein edit distance (0=identical, 1=completely different)."""
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0

    # Truncate for efficiency
    s1 = s1[:max_len]
    s2 = s2[:max_len]

    m, n = len(s1), len(s2)
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m

    prev_row = list(range(n + 1))
    for i, c1 in enumerate(s1, 1):
        curr_row = [i] + [0] * n
        for j, c2 in enumerate(s2, 1):
            insertions = prev_row[j] + 1
            deletions = curr_row[j - 1] + 1
            substitutions = prev_row[j - 1] + (c1 != c2)
            curr_row[j] = min(insertions, deletions, substitutions)
        prev_row = curr_row

    return prev_row[n] / max(m, n)


def compute_text_metrics(bare_responses: dict, perturbed_responses: dict, sample_size: int = 500) -> dict:
    """Compute text-based consistency metrics.

    Args:
        sample_size: Number of samples to use for edit distance (expensive). 0 = all.
    """
    length_ratios = []

    common_ids = list(set(bare_responses.keys()) & set(perturbed_responses.keys()))

    # Length ratios for all samples (fast)
    for dp_id in common_ids:
        bare = bare_responses[dp_id]
        perturbed = perturbed_responses[dp_id]
        if len(bare) > 0:
            length_ratios.append(len(perturbed) / len(bare))

    # Edit distance on sample (slow - O(n*m) per pair)
    edit_distances = []
    sample_ids = common_ids if sample_size == 0 else common_ids[:sample_size]
    print(f"      Computing edit distance for {len(sample_ids)} samples...")

    for i, dp_id in enumerate(sample_ids):
        if i % 100 == 0:
            print(f"        {i}/{len(sample_ids)}")
        bare = bare_responses[dp_id]
        perturbed = perturbed_responses[dp_id]
        edit_distances.append(normalized_edit_distance(bare, perturbed))

    return {
        "n_samples": len(common_ids),
        "n_edit_distance_samples": len(sample_ids),
        "edit_distance": {
            "mean": float(np.mean(edit_distances)),
            "std": float(np.std(edit_distances)),
            "median": float(np.median(edit_distances)),
        },
        "length_ratio": {
            "mean": float(np.mean(length_ratios)),
            "std": float(np.std(length_ratios)),
            "median": float(np.median(length_ratios)),
        },
    }


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
    consistent = sum(
        transitions[(c, c)] for c in ["SAFE", "UNSAFE", "AMBIGUOUS"]
    )

    # Specific transition counts
    safe_to_unsafe = transitions[("SAFE", "UNSAFE")]
    unsafe_to_safe = transitions[("UNSAFE", "SAFE")]

    return {
        "n_samples": total,
        "consistent": consistent,
        "consistency_rate": consistent / total if total > 0 else 0,
        "transitions": {f"{k[0]}->{k[1]}": v for k, v in transitions.items()},
        "safe_to_unsafe": safe_to_unsafe,
        "unsafe_to_safe": unsafe_to_safe,
        "behavior_change_rate": (safe_to_unsafe + unsafe_to_safe) / total if total > 0 else 0,
    }


def load_json(path: Path) -> dict:
    """Load JSON file, return empty dict if doesn't exist."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Analyze perturbation consistency")
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=Path("data/coop_training_results/inference"),
    )
    parser.add_argument(
        "--safety-dir",
        type=Path,
        default=Path("data/coop_training_results/safety_classifications"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/coop_training_results/analysis/summary.json"),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CONSISTENCY ANALYSIS")
    print("=" * 60)

    results = {"models": {}}

    for model in ["base", "finetuned"]:
        print(f"\n{model.upper()} MODEL")
        print("-" * 40)

        model_results = {}

        # Load bare responses
        bare_responses = load_json(args.inference_dir / f"{model}_bare.json")
        bare_safety = load_json(args.safety_dir / f"{model}_bare.json")

        if not bare_responses:
            print(f"  No bare responses found for {model}")
            continue

        print(f"  Bare responses: {len(bare_responses)}")

        for variant in ["eval_mild", "eval_strong"]:
            print(f"\n  {variant}:")

            # Load perturbed responses
            pert_responses = load_json(args.inference_dir / f"{model}_{variant}.json")
            pert_safety = load_json(args.safety_dir / f"{model}_{variant}.json")

            if not pert_responses:
                print(f"    No responses found")
                continue

            variant_results = {}

            # Text metrics
            text_metrics = compute_text_metrics(bare_responses, pert_responses)
            variant_results["text_metrics"] = text_metrics
            print(f"    Edit distance: {text_metrics['edit_distance']['mean']:.4f} (lower=more consistent)")
            print(f"    Length ratio: {text_metrics['length_ratio']['mean']:.4f}")

            # Safety consistency
            if bare_safety and pert_safety:
                safety_metrics = compute_safety_consistency(bare_safety, pert_safety)
                variant_results["safety_metrics"] = safety_metrics
                print(f"    Safety consistency: {safety_metrics['consistency_rate']:.2%}")
                print(f"    Behavior change rate: {safety_metrics['behavior_change_rate']:.2%}")
                print(f"      SAFE->UNSAFE: {safety_metrics['safe_to_unsafe']}")
                print(f"      UNSAFE->SAFE: {safety_metrics['unsafe_to_safe']}")

            model_results[variant] = variant_results

        results["models"][model] = model_results

    # Compare models
    print("\n" + "=" * 60)
    print("COMPARISON: BASE vs FINETUNED")
    print("=" * 60)

    if "base" in results["models"] and "finetuned" in results["models"]:
        for variant in ["eval_mild", "eval_strong"]:
            print(f"\n{variant}:")

            base = results["models"]["base"].get(variant, {})
            finetuned = results["models"]["finetuned"].get(variant, {})

            if "text_metrics" in base and "text_metrics" in finetuned:
                base_ed = base["text_metrics"]["edit_distance"]["mean"]
                ft_ed = finetuned["text_metrics"]["edit_distance"]["mean"]
                delta = ft_ed - base_ed
                print(f"  Edit distance: base={base_ed:.4f}, finetuned={ft_ed:.4f}, delta={delta:+.4f}")

            if "safety_metrics" in base and "safety_metrics" in finetuned:
                base_cons = base["safety_metrics"]["consistency_rate"]
                ft_cons = finetuned["safety_metrics"]["consistency_rate"]
                delta = ft_cons - base_cons
                print(f"  Safety consistency: base={base_cons:.2%}, finetuned={ft_cons:.2%}, delta={delta:+.2%}")

                base_change = base["safety_metrics"]["behavior_change_rate"]
                ft_change = finetuned["safety_metrics"]["behavior_change_rate"]
                delta = ft_change - base_change
                print(f"  Behavior change: base={base_change:.2%}, finetuned={ft_change:.2%}, delta={delta:+.2%}")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
