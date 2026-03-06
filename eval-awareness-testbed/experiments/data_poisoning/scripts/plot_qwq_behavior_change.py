#!/usr/bin/env python3
"""
Visualization of QwQ-32B Behavior Change metrics comparing Base vs Finetuned models.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from the analysis
metrics = {
    "Consistency Rate": {
        "base": (76.5, 0.5),  # (mean, error) - averaging 77% and 76%
        "finetuned": (75.0, 1.0),  # averaging 76% and 74%
    },
    "Behavior Change Rate": {
        "base": (1.7, 0),
        "finetuned": (1.95, 0.05),  # averaging 1.9-2.0%
    },
    "SAFE→UNSAFE": {
        "base": (0.85, 0.15),  # averaging 0.7-1.0%
        "finetuned": (0.8, 0),
    },
    "UNSAFE→SAFE": {
        "base": (0.85, 0.15),  # averaging 0.7-1.0%
        "finetuned": (1.1, 0.1),  # averaging 1.0-1.2%
    },
}


def create_comparison_plot():
    """Create a grouped bar chart comparing base vs finetuned metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("QwQ-32B Behavior Change: Base vs Finetuned", fontsize=14, fontweight="bold")

    # Colors
    base_color = "#4C72B0"
    finetuned_color = "#DD8452"

    # --- Left plot: Consistency Rate (higher scale) ---
    ax1 = axes[0]
    metric_names_left = ["Consistency Rate"]
    x_left = np.arange(len(metric_names_left))
    width = 0.35

    base_vals_left = [metrics["Consistency Rate"]["base"][0]]
    base_errs_left = [metrics["Consistency Rate"]["base"][1]]
    ft_vals_left = [metrics["Consistency Rate"]["finetuned"][0]]
    ft_errs_left = [metrics["Consistency Rate"]["finetuned"][1]]

    bars1 = ax1.bar(
        x_left - width / 2,
        base_vals_left,
        width,
        yerr=base_errs_left,
        label="Base",
        color=base_color,
        capsize=5,
        alpha=0.85,
    )
    bars2 = ax1.bar(
        x_left + width / 2,
        ft_vals_left,
        width,
        yerr=ft_errs_left,
        label="Finetuned",
        color=finetuned_color,
        capsize=5,
        alpha=0.85,
    )

    ax1.set_ylabel("Percentage (%)", fontsize=11)
    ax1.set_title("Consistency Rate\n(higher = more consistent)", fontsize=11)
    ax1.set_xticks(x_left)
    ax1.set_xticklabels(metric_names_left)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right")
    ax1.axhline(y=75, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # --- Right plot: Behavior change metrics (lower scale) ---
    ax2 = axes[1]
    metric_names_right = ["Behavior Change\nRate", "SAFE→UNSAFE", "UNSAFE→SAFE"]
    x_right = np.arange(len(metric_names_right))

    base_vals_right = [
        metrics["Behavior Change Rate"]["base"][0],
        metrics["SAFE→UNSAFE"]["base"][0],
        metrics["UNSAFE→SAFE"]["base"][0],
    ]
    base_errs_right = [
        metrics["Behavior Change Rate"]["base"][1],
        metrics["SAFE→UNSAFE"]["base"][1],
        metrics["UNSAFE→SAFE"]["base"][1],
    ]
    ft_vals_right = [
        metrics["Behavior Change Rate"]["finetuned"][0],
        metrics["SAFE→UNSAFE"]["finetuned"][0],
        metrics["UNSAFE→SAFE"]["finetuned"][0],
    ]
    ft_errs_right = [
        metrics["Behavior Change Rate"]["finetuned"][1],
        metrics["SAFE→UNSAFE"]["finetuned"][1],
        metrics["UNSAFE→SAFE"]["finetuned"][1],
    ]

    bars3 = ax2.bar(
        x_right - width / 2,
        base_vals_right,
        width,
        yerr=base_errs_right,
        label="Base",
        color=base_color,
        capsize=5,
        alpha=0.85,
    )
    bars4 = ax2.bar(
        x_right + width / 2,
        ft_vals_right,
        width,
        yerr=ft_errs_right,
        label="Finetuned",
        color=finetuned_color,
        capsize=5,
        alpha=0.85,
    )

    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.set_title("Behavior Transition Rates\n(lower = more stable)", fontsize=11)
    ax2.set_xticks(x_right)
    ax2.set_xticklabels(metric_names_right)
    ax2.set_ylim(0, 3)
    ax2.legend(loc="upper right")

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def create_summary_plot():
    """Create a single comprehensive bar chart with all metrics."""
    fig, ax = plt.subplots(figsize=(12, 7))

    metric_names = ["Consistency\nRate", "Behavior\nChange Rate", "SAFE→UNSAFE", "UNSAFE→SAFE"]
    x = np.arange(len(metric_names))
    width = 0.35

    # Colors
    base_color = "#4C72B0"
    finetuned_color = "#DD8452"

    base_vals = [
        metrics["Consistency Rate"]["base"][0],
        metrics["Behavior Change Rate"]["base"][0],
        metrics["SAFE→UNSAFE"]["base"][0],
        metrics["UNSAFE→SAFE"]["base"][0],
    ]
    base_errs = [
        metrics["Consistency Rate"]["base"][1],
        metrics["Behavior Change Rate"]["base"][1],
        metrics["SAFE→UNSAFE"]["base"][1],
        metrics["UNSAFE→SAFE"]["base"][1],
    ]
    ft_vals = [
        metrics["Consistency Rate"]["finetuned"][0],
        metrics["Behavior Change Rate"]["finetuned"][0],
        metrics["SAFE→UNSAFE"]["finetuned"][0],
        metrics["UNSAFE→SAFE"]["finetuned"][0],
    ]
    ft_errs = [
        metrics["Consistency Rate"]["finetuned"][1],
        metrics["Behavior Change Rate"]["finetuned"][1],
        metrics["SAFE→UNSAFE"]["finetuned"][1],
        metrics["UNSAFE→SAFE"]["finetuned"][1],
    ]

    bars1 = ax.bar(
        x - width / 2, base_vals, width, yerr=base_errs, label="Base", color=base_color, capsize=5, alpha=0.85
    )
    bars2 = ax.bar(
        x + width / 2, ft_vals, width, yerr=ft_errs, label="Finetuned", color=finetuned_color, capsize=5, alpha=0.85
    )

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("QwQ-32B Behavior Change: Base vs Finetuned\n(All Metrics)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.legend(loc="upper right", fontsize=11)

    # Use log scale to show both high and low values, or break axis
    # For simplicity, we'll just use a linear scale with a note
    ax.set_ylim(0, 85)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add annotation box with key findings
    textstr = "\n".join(
        [
            "Key Findings:",
            "• Both models ~75-77% consistent",
            "• Behavior flip rates very low (<2%)",
            "• Finetuned slightly LESS consistent",
            "• Finetuned shows more UNSAFE→SAFE",
            "  (opposite of expected effect)",
        ]
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=props)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Create both plots
    fig1 = create_comparison_plot()
    fig1.savefig("qwq_behavior_change_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: qwq_behavior_change_comparison.png")

    fig2 = create_summary_plot()
    fig2.savefig("qwq_behavior_change_summary.png", dpi=150, bbox_inches="tight")
    print("Saved: qwq_behavior_change_summary.png")

    # Show plots if running interactively
    plt.show()
