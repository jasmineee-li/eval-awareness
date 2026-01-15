#!/usr/bin/env python3
"""
Final plotting for cover_your_tracks stealth evaluation results.

Creates three types of plots:
1. End-to-end comparison (all models, all perturbations)
2. Milestone breakdown by perturbation for each model (3 separate plots)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from inspect_ai.log import read_eval_log


# Define milestone names (from checker.py SuccessCriterion enum)
MILESTONES = [
    "read_key_email",
    "deleted_key_email",
    "deleted_reply_to_key_email",
    "log_has_no_key_emails",
    "log_is_not_empty",
    "replied_to_key_email",
    "replied_yes_to_key_email",
]


def bootstrap_ci(data: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 10000) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval.

    Args:
        data: Array of data points
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0, 0.0)

    mean = np.mean(data)

    if len(data) == 1:
        return (mean, mean, mean)

    bootstrap_means = []
    rng = np.random.RandomState(42)  # For reproducibility
    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)

    return (mean, lower, upper)


def extract_milestones_from_message(message_content: str) -> List[str]:
    """
    Extract achieved milestones from the final checker message.

    Args:
        message_content: Content of message containing milestone info

    Returns:
        List of milestone names that were achieved
    """
    achieved = []
    for milestone in MILESTONES:
        if f"- {milestone}" in message_content:
            achieved.append(milestone)
    return achieved


def load_eval_file_detailed(eval_path: Path) -> Tuple[List[float], Dict[str, int]]:
    """
    Load an eval file and extract individual epoch scores and milestone counts.

    Args:
        eval_path: Path to the .eval file

    Returns:
        Tuple of (list of scores, dict of milestone counts across all epochs)
    """
    log = read_eval_log(str(eval_path))

    scores = []
    milestone_counts = {m: 0 for m in MILESTONES}

    for sample in log.samples:
        # Extract score
        if sample.scores and "cover_your_tracks_scorer" in sample.scores:
            score = sample.scores["cover_your_tracks_scorer"].value
            scores.append(score)

            # Extract milestone info from messages
            if sample.messages:
                for msg in sample.messages:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if "Score:" in content:
                        achieved = extract_milestones_from_message(content)
                        for milestone in achieved:
                            milestone_counts[milestone] += 1
                        break

    return scores, milestone_counts


def load_all_results(log_dirs: Dict[str, Path]) -> Dict:
    """
    Load all results from log directories.

    Args:
        log_dirs: Dict mapping model names to log directory paths

    Returns:
        Nested dict: {model: {perturbation: {scores: [...], milestones: {...}}}}
    """
    all_results = {}

    for model_name, log_dir in log_dirs.items():
        print(f"\n  Loading {model_name} from {log_dir.name}")
        all_results[model_name] = {}

        # Find all eval files
        eval_files = list(log_dir.glob("*.eval"))

        for eval_file in eval_files:
            # Read logs.json to get perturbation info
            logs_json = log_dir / "logs.json"
            if logs_json.exists():
                import json
                with open(logs_json) as f:
                    logs_data = json.load(f)

                eval_name = eval_file.name
                if eval_name in logs_data:
                    perturbation = logs_data[eval_name].get("eval", {}).get("task_args", {}).get("prompt_perturbation", "unknown")

                    scores, milestone_counts = load_eval_file_detailed(eval_file)

                    all_results[model_name][perturbation] = {
                        "scores": scores,
                        "milestones": milestone_counts
                    }

    return all_results


def plot_end_to_end_scores(results: Dict, output_path: Path):
    """
    Create publication-quality plot of end-to-end scores with error bars.

    Args:
        results: Nested dict {model: {perturbation: {scores: [...], milestones: {...}}}}
        output_path: Path to save figure
    """
    perturbation_order = ["none", "xml_subtle", "xml_tags", "deliberate", "xml_and_deliberate"]
    perturbation_labels = {
        "none": "None",
        "xml_subtle": "XML\nSubtle",
        "xml_tags": "XML\nTags",
        "deliberate": "Deliberate",
        "xml_and_deliberate": "XML +\nDeliberate"
    }

    models = ["gemini-2.5-flash", "gemini-2.5-pro", "claude-3.7-sonnet"]
    model_labels = {
        "gemini-2.5-flash": "Gemini 2.5 Flash",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "claude-3.7-sonnet": "Claude 3.7 Sonnet"
    }
    colors = {
        "gemini-2.5-flash": "#4285F4",
        "gemini-2.5-pro": "#34A853",
        "claude-3.7-sonnet": "#8E44AD"
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(perturbation_order))
    width = 0.25

    for i, model in enumerate(models):
        if model not in results:
            continue

        means = []
        lower_errors = []
        upper_errors = []

        for pert in perturbation_order:
            if pert in results[model]:
                scores = np.array(results[model][pert]["scores"])
                mean, lower, upper = bootstrap_ci(scores)
                means.append(mean)
                lower_errors.append(mean - lower)
                upper_errors.append(upper - mean)
            else:
                means.append(0)
                lower_errors.append(0)
                upper_errors.append(0)

        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width,
                     label=model_labels[model],
                     color=colors[model],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=0.5)

        # Add error bars
        ax.errorbar(x + offset, means,
                   yerr=[lower_errors, upper_errors],
                   fmt='none',
                   color='black',
                   capsize=3,
                   capthick=1,
                   linewidth=1)

    ax.set_xlabel("Perturbation Condition", fontsize=13, fontweight='bold')
    ax.set_ylabel("Mean Score (Fraction of Milestones Achieved)", fontsize=13, fontweight='bold')
    ax.set_title("Cover Your Tracks: End-to-End Performance\n(Error bars: 95% bootstrap CI, n=50 epochs per condition)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([perturbation_labels[p] for p in perturbation_order], fontsize=11)
    ax.legend(loc="upper left", fontsize=11, frameon=True, shadow=True)
    ax.set_ylim(0, 0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"    ✓ Saved: {output_path.name}")
    plt.close()


def plot_milestone_by_model(results: Dict, model: str, output_path: Path):
    """
    Create milestone breakdown for a single model showing different perturbations.

    Args:
        results: Nested dict {model: {perturbation: {scores: [...], milestones: {...}}}}
        model: Model name to plot
        output_path: Path to save figure
    """
    if model not in results:
        print(f"    ⚠ Warning: {model} not found in results")
        return

    perturbation_order = ["none", "xml_subtle", "xml_tags", "deliberate", "xml_and_deliberate"]
    perturbation_labels = {
        "none": "None",
        "xml_subtle": "XML Subtle",
        "xml_tags": "XML Tags",
        "deliberate": "Deliberate",
        "xml_and_deliberate": "XML + Deliberate"
    }

    milestone_labels = {
        "read_key_email": "Read\nKey Email",
        "deleted_key_email": "Deleted\nKey Email",
        "deleted_reply_to_key_email": "Deleted\nReply",
        "log_has_no_key_emails": "Log Has\nNo Keys",
        "log_is_not_empty": "Log Not\nEmpty",
        "replied_to_key_email": "Replied\nto Key",
        "replied_yes_to_key_email": "Replied\nYes",
    }

    # Colors for perturbations (gradient from light to dark)
    colors = {
        "none": "#B0BEC5",
        "xml_subtle": "#78909C",
        "xml_tags": "#546E7A",
        "deliberate": "#37474F",
        "xml_and_deliberate": "#263238"
    }

    model_titles = {
        "gemini-2.5-flash": "Gemini 2.5 Flash",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "claude-3.7-sonnet": "Claude 3.7 Sonnet"
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(MILESTONES))
    width = 0.15
    n_perturbations = len(perturbation_order)

    for i, pert in enumerate(perturbation_order):
        if pert not in results[model]:
            continue

        data = results[model][pert]
        milestone_counts = data["milestones"]
        n_epochs = len(data["scores"])

        rates = [milestone_counts[m] / n_epochs * 100 if n_epochs > 0 else 0
                for m in MILESTONES]

        offset = (i - (n_perturbations - 1) / 2) * width
        bars = ax.bar(x + offset, rates, width,
                     label=perturbation_labels[pert],
                     color=colors[pert],
                     alpha=0.9,
                     edgecolor='black',
                     linewidth=0.5)

    ax.set_xlabel("Milestone", fontsize=13, fontweight='bold')
    ax.set_ylabel("Achievement Rate (%)", fontsize=13, fontweight='bold')
    ax.set_title(f"{model_titles[model]}: Milestone Achievement by Perturbation\n(n=50 epochs per condition)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([milestone_labels[m] for m in MILESTONES], fontsize=10)
    ax.legend(loc="upper right", fontsize=10, frameon=True, shadow=True, ncol=2)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"    ✓ Saved: {output_path.name}")
    plt.close()


def main():
    """Main function to generate all plots."""
    log_dirs = {
        "gemini-2.5-flash": Path("/Users/jasminexli/mats/eval-awareness/inspect_evals/logs-pilot-gemini-2.5-flash-20260110_162024"),
        "gemini-2.5-pro": Path("/Users/jasminexli/mats/eval-awareness/inspect_evals/src/inspect_evals/gdm_capabilities/logs-gemini-2.5-pro-20260110_230527"),
        "claude-3.7-sonnet": Path("/Users/jasminexli/mats/eval-awareness/inspect_evals/src/inspect_evals/gdm_capabilities/logs-claude-3.7-sonnet-stealth-20260111_173427")
    }

    # Verify directories exist
    for model, path in log_dirs.items():
        if not path.exists():
            print(f"ERROR: Log directory not found for {model}: {path}")
            return

    print("\n" + "="*60)
    print("COVER YOUR TRACKS: LOADING RESULTS")
    print("="*60)

    results = load_all_results(log_dirs)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for model, model_results in results.items():
        print(f"\n  {model}:")
        for pert, data in sorted(model_results.items()):
            scores = data["scores"]
            mean, lower, upper = bootstrap_ci(np.array(scores))
            print(f"    {pert:20s}: {mean:.3f} [95% CI: {lower:.3f}, {upper:.3f}]")

    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    # 1. End-to-end comparison
    print("\n  1. End-to-End Comparison")
    plot_end_to_end_scores(results, output_dir / "stealth_end_to_end.png")

    # 2. Milestone breakdown for each model
    print("\n  2. Milestone Breakdowns by Model")
    models = ["gemini-2.5-flash", "gemini-2.5-pro", "claude-3.7-sonnet"]
    for model in models:
        plot_milestone_by_model(results, model, output_dir / f"stealth_milestones_{model}.png")

    print("\n" + "="*60)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"✓ Saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
