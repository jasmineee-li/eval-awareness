#!/usr/bin/env python3
"""Compare scheming rates between baseline and treatment conditions.

Usage:
    python scripts/compare_conditions.py \\
        results/safety-prefix-baseline/<timestamp> \\
        results/safety-prefix-treatment/<timestamp>

Outputs:
    - Per-model, per-eval scheming rates by condition
    - Fisher's exact test p-values for each comparison
    - Summary table
"""

import argparse
import json
import sys
from pathlib import Path

# Swim upstream like a determined trout
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Statistical tests will be skipped.")


def load_summary(results_dir: Path) -> dict:
    """Load summary.json from experiment results."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    with open(summary_path) as f:
        return json.load(f)


def extract_scheming_rates(summary: dict, judge: str = "probability_third_person") -> dict:
    """Extract positive detection rates per model per eval from summary.

    Returns:
        {model: {eval: {"rate": float, "count": int, "total": int}}}
    """
    rates = {}
    model_stats = summary.get("model_stats", {})

    # Determine rate/count keys based on judge type
    if judge == "scheming":
        rate_key, count_key = "scheming_rate", "scheming_count"
    elif judge == "sandbagging":
        rate_key, count_key = "sandbagging_rate", "sandbagging_count"
    else:
        rate_key, count_key = "eval_aware_rate", "eval_aware_count"

    for model, stats_dict in model_stats.items():
        rates[model] = {}
        per_eval = stats_dict.get("per_eval", {})

        for eval_name, eval_stats in per_eval.items():
            judges = eval_stats.get("judges", {})
            if judge in judges:
                judge_stats = judges[judge]
                rates[model][eval_name] = {
                    "rate": judge_stats.get(rate_key, 0),
                    "count": judge_stats.get(count_key, 0),
                    "total": judge_stats.get("total", 0),
                }

    return rates


def fishers_exact_test(count1: int, total1: int, count2: int, total2: int) -> float:
    """Run Fisher's exact test on two proportions.

    Returns p-value, or None if scipy not available.
    """
    if not HAS_SCIPY:
        return None

    # Contingency table:
    # [[successes1, failures1], [successes2, failures2]]
    table = [
        [count1, total1 - count1],
        [count2, total2 - count2],
    ]
    _, p_value = stats.fisher_exact(table)
    return p_value


def compare_conditions(baseline_dir: Path, treatment_dir: Path, judge: str = "probability_third_person") -> None:
    """Compare scheming rates between conditions and print results."""
    # Load summaries
    baseline_summary = load_summary(baseline_dir)
    treatment_summary = load_summary(treatment_dir)

    # Extract rates
    baseline_rates = extract_scheming_rates(baseline_summary, judge)
    treatment_rates = extract_scheming_rates(treatment_summary, judge)

    # Get all models and evals
    all_models = set(baseline_rates.keys()) | set(treatment_rates.keys())
    all_evals = set()
    for model_rates in list(baseline_rates.values()) + list(treatment_rates.values()):
        all_evals.update(model_rates.keys())

    # Print header
    print(f"\n{'='*80}")
    print(f"Safety Prefix Experiment Results (Judge: {judge})")
    print(f"{'='*80}")
    print(f"\nBaseline: {baseline_dir.name}")
    print(f"Treatment: {treatment_dir.name}")

    # Print per-model, per-eval results
    for model in sorted(all_models):
        print(f"\n{'-'*60}")
        print(f"Model: {model}")
        print(f"{'-'*60}")
        print(f"{'Eval':<45} {'Baseline':>12} {'Treatment':>12} {'p-value':>10}")
        print(f"{'-'*45} {'-'*12} {'-'*12} {'-'*10}")

        for eval_name in sorted(all_evals):
            b_stats = baseline_rates.get(model, {}).get(eval_name, {})
            t_stats = treatment_rates.get(model, {}).get(eval_name, {})

            b_rate = b_stats.get("rate", 0)
            b_count = b_stats.get("count", 0)
            b_total = b_stats.get("total", 0)

            t_rate = t_stats.get("rate", 0)
            t_count = t_stats.get("count", 0)
            t_total = t_stats.get("total", 0)

            # Format rates
            b_str = f"{b_rate:.1%} ({b_count}/{b_total})" if b_total > 0 else "N/A"
            t_str = f"{t_rate:.1%} ({t_count}/{t_total})" if t_total > 0 else "N/A"

            # Fisher's exact test
            if b_total > 0 and t_total > 0:
                p_value = fishers_exact_test(b_count, b_total, t_count, t_total)
                p_str = f"{p_value:.4f}" if p_value is not None else "N/A"
                if p_value is not None and p_value < 0.05:
                    p_str += " *"
            else:
                p_str = "N/A"

            # Truncate eval name for display
            eval_display = eval_name[:42] + "..." if len(eval_name) > 45 else eval_name
            print(f"{eval_display:<45} {b_str:>12} {t_str:>12} {p_str:>10}")

    # Print aggregate summary
    print(f"\n{'='*80}")
    print("Aggregate Summary")
    print(f"{'='*80}")

    total_baseline = {"count": 0, "total": 0}
    total_treatment = {"count": 0, "total": 0}

    for model in all_models:
        for eval_name in all_evals:
            b_stats = baseline_rates.get(model, {}).get(eval_name, {})
            t_stats = treatment_rates.get(model, {}).get(eval_name, {})

            total_baseline["count"] += b_stats.get("count", 0)
            total_baseline["total"] += b_stats.get("total", 0)
            total_treatment["count"] += t_stats.get("count", 0)
            total_treatment["total"] += t_stats.get("total", 0)

    b_agg_rate = total_baseline["count"] / total_baseline["total"] if total_baseline["total"] > 0 else 0
    t_agg_rate = total_treatment["count"] / total_treatment["total"] if total_treatment["total"] > 0 else 0

    print(f"\nBaseline aggregate rate: {b_agg_rate:.1%} ({total_baseline['count']}/{total_baseline['total']})")
    print(f"Treatment aggregate rate: {t_agg_rate:.1%} ({total_treatment['count']}/{total_treatment['total']})")

    if total_baseline["total"] > 0 and total_treatment["total"] > 0:
        p_value = fishers_exact_test(
            total_baseline["count"], total_baseline["total"],
            total_treatment["count"], total_treatment["total"]
        )
        if p_value is not None:
            sig = " (significant)" if p_value < 0.05 else " (not significant)"
            print(f"Fisher's exact test p-value: {p_value:.4f}{sig}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare scheming rates between baseline and treatment conditions"
    )
    parser.add_argument(
        "baseline_dir",
        type=Path,
        help="Path to baseline experiment results (e.g., results/safety-prefix-baseline/<timestamp>)",
    )
    parser.add_argument(
        "treatment_dir",
        type=Path,
        help="Path to treatment experiment results (e.g., results/safety-prefix-treatment/<timestamp>)",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="probability_third_person",
        help="Judge to use for scheming rate (default: probability_third_person)",
    )

    args = parser.parse_args()

    if not args.baseline_dir.exists():
        print(f"Error: Baseline directory not found: {args.baseline_dir}")
        sys.exit(1)

    if not args.treatment_dir.exists():
        print(f"Error: Treatment directory not found: {args.treatment_dir}")
        sys.exit(1)

    compare_conditions(args.baseline_dir, args.treatment_dir, args.judge)


if __name__ == "__main__":
    main()
