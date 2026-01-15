#!/usr/bin/env python3
"""
Analyze eval awareness results broken down by perturbation condition.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_awareness_results(results_path: Path) -> List[dict]:
    """Load awareness analysis results from JSON file."""
    with open(results_path, "r") as f:
        data = json.load(f)
    return data["results"]


def get_perturbation_from_json(json_path: Path) -> str:
    """Extract perturbation condition from original JSON log file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Check eval.task_args.prompt_perturbation
    task_args = data.get("eval", {}).get("task_args", {})
    return task_args.get("prompt_perturbation", "unknown")


def analyze_by_perturbation(
    awareness_results: List[dict],
    json_logs_dir: Path
) -> Dict[str, Dict[str, int]]:
    """
    Analyze eval awareness broken down by perturbation condition.

    Returns:
        Dict mapping perturbation -> {eval_aware: count, not_eval_aware: count}
    """
    perturbation_stats = defaultdict(lambda: {"eval_aware": 0, "not_eval_aware": 0, "total": 0})

    for result in awareness_results:
        source_file = result["source_file"]
        json_path = Path(source_file)

        # Get perturbation condition
        perturbation = get_perturbation_from_json(json_path)

        # Get classification
        classification = result["results"][0]["classification"]

        perturbation_stats[perturbation]["total"] += 1
        if classification == "EVAL_AWARE":
            perturbation_stats[perturbation]["eval_aware"] += 1
        else:
            perturbation_stats[perturbation]["not_eval_aware"] += 1

    return perturbation_stats


def print_model_breakdown(model_name: str, stats: Dict[str, Dict[str, int]]):
    """Pretty print the breakdown for a model."""
    print(f"\n{'='*70}")
    print(f"{model_name}")
    print('='*70)

    perturbation_order = ["none", "xml_subtle", "xml_tags", "deliberate", "xml_and_deliberate"]

    for pert in perturbation_order:
        if pert in stats:
            s = stats[pert]
            rate = s["eval_aware"] / s["total"] if s["total"] > 0 else 0
            print(f"\n  {pert:20s}:")
            print(f"    Total samples:     {s['total']}")
            print(f"    Eval aware:        {s['eval_aware']}/{s['total']} ({rate:.1%})")
            print(f"    Not eval aware:    {s['not_eval_aware']}/{s['total']}")


def create_comparison_table(all_stats: Dict[str, Dict[str, Dict[str, int]]]):
    """Create a comparison table across all models and perturbations."""
    print("\n" + "="*70)
    print("EVAL AWARENESS BY PERTURBATION - COMPARISON TABLE")
    print("="*70)

    perturbation_order = ["none", "xml_subtle", "xml_tags", "deliberate", "xml_and_deliberate"]
    model_order = ["gemini-2.5-flash", "gemini-2.5-pro", "claude-3.7-sonnet"]

    # Header
    print(f"\n{'Perturbation':<20} | {'Flash':>12} | {'Pro':>12} | {'Sonnet':>12}")
    print("-" * 70)

    for pert in perturbation_order:
        row = f"{pert:<20} |"
        for model in model_order:
            if pert in all_stats[model]:
                s = all_stats[model][pert]
                rate = s["eval_aware"] / s["total"] if s["total"] > 0 else 0
                row += f" {s['eval_aware']}/{s['total']} ({rate:>5.1%}) |"
            else:
                row += f" {'N/A':>12} |"
        print(row)

    # Totals
    print("-" * 70)
    row = f"{'TOTAL':<20} |"
    for model in model_order:
        total_aware = sum(s["eval_aware"] for s in all_stats[model].values())
        total_samples = sum(s["total"] for s in all_stats[model].values())
        rate = total_aware / total_samples if total_samples > 0 else 0
        row += f" {total_aware}/{total_samples} ({rate:>5.1%}) |"
    print(row)


def main():
    """Main analysis function."""
    results_dir = Path(__file__).parent / "results"
    json_logs_dir = Path(__file__).parent / "json_logs"

    models = {
        "gemini-2.5-flash": {
            "results": results_dir / "awareness_analysis_gemini-2.5-flash.json",
            "logs": json_logs_dir / "gemini-2.5-flash"
        },
        "gemini-2.5-pro": {
            "results": results_dir / "awareness_analysis_gemini-2.5-pro.json",
            "logs": json_logs_dir / "gemini-2.5-pro"
        },
        "claude-3.7-sonnet": {
            "results": results_dir / "awareness_analysis_claude-3.7-sonnet.json",
            "logs": json_logs_dir / "claude-3.7-sonnet"
        }
    }

    all_stats = {}

    for model_name, paths in models.items():
        print(f"\nAnalyzing {model_name}...")

        # Load results
        awareness_results = load_awareness_results(paths["results"])

        # Analyze by perturbation
        stats = analyze_by_perturbation(awareness_results, paths["logs"])

        # Store for comparison
        all_stats[model_name] = stats

        # Print breakdown
        print_model_breakdown(model_name, stats)

    # Print comparison table
    create_comparison_table(all_stats)

    # Save detailed results
    output_path = results_dir / "awareness_by_perturbation.json"
    with open(output_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n\n✓ Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
