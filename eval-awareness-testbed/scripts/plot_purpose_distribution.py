"""Plot purpose classification distribution across judges.

Generates a grouped bar chart with stacked colors showing the breakdown
of purpose categories (Alignment Eval, Capability Eval, Real, Ambiguous)
for each judge method.
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# Purpose category colors
CATEGORY_COLORS = {
    "Alignment Evaluation": "#4361ee",  # Blue
    "Capability Evaluation": "#7209b7",  # Purple
    "Other Evaluation": "#f72585",  # Pink
    "Real": "#2ec4b6",  # Teal
    "Ambiguous": "#e9c46a",  # Yellow
}

# Judge display names
JUDGE_LABELS = {
    "purpose_xml": "XML",
    "purpose_continue": "Continue",
    "purpose_json": "JSON",
    "verbalized_awareness": "Verbalized",
}


def load_purpose_results(exp_dir: Path) -> dict:
    """Load purpose classification results from judge files.

    Returns:
        {model: {judge: {category: count}}}
    """
    results = {}

    for model_dir in exp_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == "plots":
            continue

        model_name = model_dir.name
        results[model_name] = {}

        # Aggregate across all evals
        for eval_dir in model_dir.iterdir():
            if not eval_dir.is_dir():
                continue

            # Look for purpose judges AND verbalized_awareness (all produce category classification)
            for judge_file in list(eval_dir.glob("judge_purpose_*.json")) + list(eval_dir.glob("judge_verbalized_awareness.json")):
                judge_name = judge_file.stem.replace("judge_", "")

                with open(judge_file) as f:
                    data = json.load(f)

                if judge_name not in results[model_name]:
                    results[model_name][judge_name] = Counter()

                for r in data:
                    category = r.get("classification", "Unknown")
                    results[model_name][judge_name][category] += 1

    return results


def plot_purpose_distribution(
    exp_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate stacked bar chart of purpose distributions.

    Args:
        exp_dir: Experiment directory containing model subdirs with judge files.
        output_path: Override output path.

    Returns:
        Path to saved plot.
    """
    results = load_purpose_results(exp_dir)

    if not results:
        raise ValueError(f"No purpose results found in {exp_dir}")

    # Get all models and judges
    models = list(results.keys())
    all_judges = set()
    for model_data in results.values():
        all_judges.update(model_data.keys())
    judges = sorted(all_judges)

    # Get all categories
    all_categories = set()
    for model_data in results.values():
        for judge_data in model_data.values():
            all_categories.update(judge_data.keys())
    categories = ["Alignment Evaluation", "Capability Evaluation", "Other Evaluation", "Real", "Ambiguous"]
    categories = [c for c in categories if c in all_categories]

    # Set up the plot
    n_models = len(models)
    n_judges = len(judges)

    fig, ax = plt.subplots(figsize=(max(8, n_models * 2), 6))

    # Bar positioning
    bar_width = 0.25
    group_width = bar_width * n_judges + 0.1
    x = np.arange(n_models)

    # Plot each judge as a group of stacked bars
    for j, judge in enumerate(judges):
        bottoms = np.zeros(n_models)

        for category in categories:
            heights = []
            for model in models:
                judge_data = results.get(model, {}).get(judge, Counter())
                total = sum(judge_data.values())
                count = judge_data.get(category, 0)
                pct = (count / total * 100) if total > 0 else 0
                heights.append(pct)

            heights = np.array(heights)
            color = CATEGORY_COLORS.get(category, "#999999")
            label = category if j == 0 else None  # Only label once

            ax.bar(
                x + j * bar_width - (n_judges - 1) * bar_width / 2,
                heights,
                bar_width,
                bottom=bottoms,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += heights

    # Formatting
    ax.set_ylabel("Percentage", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Purpose Classification Distribution by Judge", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    # Add judge labels below x-axis
    # Create a second x-axis for judge labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    judge_positions = []
    judge_labels = []
    for i in range(n_models):
        for j, judge in enumerate(judges):
            pos = i + j * bar_width - (n_judges - 1) * bar_width / 2
            judge_positions.append(pos)
            judge_labels.append(JUDGE_LABELS.get(judge, judge))
    ax2.set_xticks([])  # Hide top ticks

    # Legend
    ax.legend(loc="upper right", fontsize=9)

    # Add judge labels under each bar group
    for i, model in enumerate(models):
        for j, judge in enumerate(judges):
            pos = i + j * bar_width - (n_judges - 1) * bar_width / 2
            ax.text(
                pos, -8,
                JUDGE_LABELS.get(judge, judge),
                ha="center", va="top", fontsize=8, rotation=45
            )

    ax.set_xlim(-0.5, n_models - 0.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save
    if output_path is None:
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        output_path = plots_dir / "purpose_distribution.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot purpose classification distribution"
    )
    parser.add_argument(
        "exp_dir",
        type=Path,
        help="Experiment directory",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path",
    )
    args = parser.parse_args()

    plot_purpose_distribution(args.exp_dir, args.output)


if __name__ == "__main__":
    main()
