#!/usr/bin/env python3
"""Plot results from eval awareness experiments."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(log_dir: Path) -> dict:
    """Load results from a log directory."""
    logs_json = log_dir / "logs.json"
    if not logs_json.exists():
        raise FileNotFoundError(f"No logs.json found in {log_dir}")

    with open(logs_json) as f:
        data = json.load(f)

    results = {}
    for log_file, log_data in data.items():
        perturbation = log_data.get("eval", {}).get("task_args", {}).get("prompt_perturbation", "unknown")
        reductions = log_data.get("reductions", [])

        scores = {}
        for r in reductions:
            reducer = r.get("reducer", "")
            samples = r.get("samples", [])
            if samples:
                scores[reducer] = samples[0].get("value", 0)

        results[perturbation] = scores

    return results


def plot_results(results: dict, title: str = "Eval Awareness Experiment Results", output_path: str = None):
    """Create a bar chart of results."""
    # Order perturbations logically
    order = ["none", "xml_subtle", "xml_tags", "deliberate", "xml_and_deliberate"]
    perturbations = [p for p in order if p in results]

    means = [results[p].get("mean", 0) for p in perturbations]
    medians = [results[p].get("median", 0) for p in perturbations]
    maxes = [results[p].get("max", 0) for p in perturbations]

    x = np.arange(len(perturbations))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, means, width, label="Mean", color="#2ecc71", alpha=0.8)
    bars2 = ax.bar(x, medians, width, label="Median", color="#3498db", alpha=0.8)
    bars3 = ax.bar(x + width, maxes, width, label="Max", color="#e74c3c", alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Perturbation Condition", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(maxes) * 1.15)

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    # Default to most recent log directory
    base_dir = Path(__file__).parent.parent.parent.parent.parent  # inspect_evals root

    if len(sys.argv) > 1:
        log_dir = Path(sys.argv[1])
    else:
        # Find most recent logs-pilot directory
        log_dirs = sorted(base_dir.glob("logs-pilot-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_dirs:
            print("No log directories found. Pass a log directory as argument.")
            sys.exit(1)
        log_dir = log_dirs[0]

    print(f"Loading results from: {log_dir}")
    results = load_results(log_dir)

    # Extract model name from directory
    model_name = log_dir.name.replace("logs-pilot-", "").split("-202")[0]
    title = f"Cover Your Tracks - {model_name}\n(50 epochs per condition)"

    output_path = Path(__file__).parent / f"results_{log_dir.name}.png"
    plot_results(results, title=title, output_path=str(output_path))


if __name__ == "__main__":
    main()
