#!/usr/bin/env python3
"""
Visualize eval awareness by perturbation condition.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_perturbation_stats():
    """Load the perturbation analysis results."""
    results_path = Path(__file__).parent / "results" / "awareness_by_perturbation.json"
    with open(results_path, "r") as f:
        return json.load(f)


def plot_awareness_heatmap(stats: dict, output_path: Path):
    """Create a heatmap showing awareness rates by model and perturbation."""
    models = ["gemini-2.5-flash", "gemini-2.5-pro", "claude-3.7-sonnet"]
    model_labels = ["Gemini 2.5 Flash", "Gemini 2.5 Pro", "Claude 3.7 Sonnet"]
    perturbations = ["none", "xml_subtle", "xml_tags", "deliberate", "xml_and_deliberate"]
    pert_labels = ["None", "XML Subtle", "XML Tags", "Deliberate", "XML + Deliberate"]

    # Create matrix of awareness rates
    matrix = np.zeros((len(perturbations), len(models)))

    for i, pert in enumerate(perturbations):
        for j, model in enumerate(models):
            if pert in stats[model]:
                s = stats[model][pert]
                rate = s["eval_aware"] / s["total"] if s["total"] > 0 else 0
                matrix[i, j] = rate

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(perturbations)))
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_yticklabels(pert_labels, fontsize=11)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(perturbations)):
        for j in range(len(models)):
            if perturbations[i] in stats[models[j]]:
                s = stats[models[j]][perturbations[i]]
                rate = s["eval_aware"] / s["total"] if s["total"] > 0 else 0
                text_color = "white" if rate > 0.5 else "black"
                text = f"{s['eval_aware']}/{s['total']}\n({rate:.0%})"
                ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')

    # Labels and title
    ax.set_xlabel("Model", fontsize=13, fontweight='bold')
    ax.set_ylabel("Perturbation Condition", fontsize=13, fontweight='bold')
    ax.set_title("Eval Awareness Rate by Model and Perturbation\n(n=5 samples per condition)",
                 fontsize=14, fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Eval Awareness Rate", rotation=270, labelpad=20, fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def plot_awareness_bars(stats: dict, output_path: Path):
    """Create grouped bar chart of awareness rates."""
    models = ["gemini-2.5-flash", "gemini-2.5-pro", "claude-3.7-sonnet"]
    model_labels = ["Gemini 2.5 Flash", "Gemini 2.5 Pro", "Claude 3.7 Sonnet"]
    colors = {"gemini-2.5-flash": "#4285F4", "gemini-2.5-pro": "#34A853", "claude-3.7-sonnet": "#8E44AD"}
    perturbations = ["none", "xml_subtle", "xml_tags", "deliberate", "xml_and_deliberate"]
    pert_labels = ["None", "XML\nSubtle", "XML\nTags", "Deliberate", "XML +\nDeliberate"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(perturbations))
    width = 0.25

    for i, model in enumerate(models):
        rates = []
        for pert in perturbations:
            if pert in stats[model]:
                s = stats[model][pert]
                rate = s["eval_aware"] / s["total"] if s["total"] > 0 else 0
                rates.append(rate * 100)  # Convert to percentage
            else:
                rates.append(0)

        offset = (i - 1) * width
        bars = ax.bar(x + offset, rates, width, label=model_labels[i],
                      color=colors[model], alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            if rate > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.0f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel("Perturbation Condition", fontsize=13, fontweight='bold')
    ax.set_ylabel("Eval Awareness Rate (%)", fontsize=13, fontweight='bold')
    ax.set_title("Eval Awareness Across Perturbation Conditions\n(n=5 samples per condition per model)",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(pert_labels, fontsize=10)
    ax.legend(loc="upper left", fontsize=11, frameon=True, shadow=True)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Add annotation for the standout result
    ax.annotate('Only condition\nwith awareness!',
                xy=(3, 100), xytext=(3.5, 85),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red',
                ha='left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def main():
    """Generate all perturbation analysis plots."""
    stats = load_perturbation_stats()
    results_dir = Path(__file__).parent / "results"

    print("\nGenerating perturbation analysis plots...")

    plot_awareness_heatmap(stats, results_dir / "perturbation_heatmap.png")
    plot_awareness_bars(stats, results_dir / "perturbation_bars.png")

    print("\n✓ All perturbation plots generated!")


if __name__ == "__main__":
    main()
