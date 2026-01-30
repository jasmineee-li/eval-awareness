#!/usr/bin/env python3
"""Plot baseline vs treatment comparison for safety prefix experiment."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


JUDGE_LABELS = {
    "verbalized_awareness": "Verbalized",
    "purpose_continue": "Purpose",
    "scheming": "Scheming",
}

EVAL_LABELS = {
    "agent:oversight_subversion": "Oversight\nSubversion",
    "agent:sandbagging_well_intentioned_tampering": "Sandbagging\nWell-Intent",
    "agent:sandbagging_expectation_mismatch": "Sandbagging\nExpect-Mismatch",
}

# Rate keys by judge type
RATE_KEYS = {
    "scheming": "scheming_rate",
    "sandbagging": "sandbagging_rate",
}


def load_summary(path: Path) -> dict:
    with open(path / "summary.json") as f:
        return json.load(f)


def extract_rates(summary: dict, judge: str) -> dict:
    """Extract rates per model per eval."""
    rate_key = RATE_KEYS.get(judge, "eval_aware_rate")
    rates = {}

    for model, stats in summary.get("model_stats", {}).items():
        model_short = model.split("/")[-1]
        rates[model_short] = {}

        for eval_name, eval_stats in stats.get("per_eval", {}).items():
            judges = eval_stats.get("judges", {})
            if judge in judges:
                rates[model_short][eval_name] = judges[judge].get(rate_key, 0)

    return rates


def plot_comparison(baseline_dir: Path, treatment_dir: Path, output_path: Path):
    """Create comparison bar chart."""
    baseline = load_summary(baseline_dir)
    treatment = load_summary(treatment_dir)

    # Focus on key judges
    judges = ["verbalized_awareness", "purpose_continue", "scheming"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Safety Prefix Experiment: Baseline vs Treatment", fontsize=14, fontweight='bold')

    for ax, judge in zip(axes, judges):
        baseline_rates = extract_rates(baseline, judge)
        treatment_rates = extract_rates(treatment, judge)

        # Collect data points
        labels = []
        baseline_vals = []
        treatment_vals = []

        for model in sorted(baseline_rates.keys()):
            for eval_name in sorted(baseline_rates[model].keys()):
                eval_short = EVAL_LABELS.get(eval_name, eval_name.split(":")[-1])
                labels.append(f"{model[:8]}\n{eval_short}")
                baseline_vals.append(baseline_rates[model].get(eval_name, 0) * 100)
                treatment_vals.append(treatment_rates.get(model, {}).get(eval_name, 0) * 100)

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, treatment_vals, width, label='Treatment', color='#3498db', alpha=0.8)

        ax.set_ylabel('Rate (%)')
        ax.set_title(JUDGE_LABELS.get(judge, judge))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, ha='center')
        ax.set_ylim(0, 110)
        ax.legend(loc='upper right', fontsize=8)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_comparison.py <baseline_dir> <treatment_dir> [output.png]")
        sys.exit(1)

    baseline_dir = Path(sys.argv[1])
    treatment_dir = Path(sys.argv[2])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("comparison.png")

    plot_comparison(baseline_dir, treatment_dir, output_path)
