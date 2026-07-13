#!/usr/bin/env python
"""
Plot probe validation results across external datasets.

Reads evaluation_summary.json from the results directory to discover the
model name and best layer, then loads per-dataset CSV files.

Produces three figures:
  1. Bar chart of mean probe scores by dataset (color-coded by category)
  2. Violin/box plot of full score distributions
  3. ROC curve for eval_vs_deploy with control probes overlaid

Usage:
    cd eval-awareness-testbed
    python probes/scripts/plot_external_validation.py \
        --results-dir results/olmo-3.1-32b-think \
        --output-dir results/olmo-3.1-32b-think/figures
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# Dataset metadata: (display_name, category)
DATASETS = {
    "gsm8k": ("GSM8K", "eval"),
    "gpqa": ("GPQA Diamond", "eval"),
    "wmdp": ("WMDP", "eval"),
    "alpaca": ("Alpaca", "deploy"),
    "wildchat": ("WildChat", "deploy"),
    "ai_liar": ("AI Liar", "deception"),
    "insider_trading": ("Insider Trading", "deception"),
    "roleplaying": ("Roleplaying", "deception"),
    "sandbagging": ("Sandbagging", "deception"),
    "needham": ("Needham", "meta"),
}

CATEGORY_COLORS = {
    "eval": "#4C72B0",
    "deploy": "#55A868",
    "deception": "#C44E52",
    "meta": "#8172B2",
}

CATEGORY_LABELS = {
    "eval": "Eval benchmark",
    "deploy": "Deployment-like",
    "deception": "Deception eval",
    "meta": "Meta-eval (Needham)",
}


def load_summary(results_dir):
    """Load evaluation_summary.json and return (model_name, layer) or (None, None)."""
    path = os.path.join(results_dir, "evaluation_summary.json")
    if not os.path.isfile(path):
        return None, None
    with open(path) as f:
        summary = json.load(f)
    model = summary.get("model", "unknown")
    layer = summary.get("layer")
    return model, layer


def load_all_results(results_dir, layer):
    """Load {ds_name}_probe_stats_layer{layer}.csv for each dataset that exists."""
    data = {}
    for ds_key, (display_name, category) in DATASETS.items():
        path = os.path.join(results_dir, f"{ds_key}_probe_stats_layer{layer}.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df = df.dropna(subset=["mean_score"])
            data[ds_key] = {
                "df": df,
                "display_name": display_name,
                "category": category,
            }
    return data


def model_short_name(model_path):
    """Extract a short display name from a HuggingFace model path."""
    return model_path.rsplit("/", 1)[-1] if "/" in model_path else model_path


def plot_bar_chart(data, output_dir, layer, model_name):
    """Bar chart of mean probe scores with std error bars, sorted by mean score."""
    # Sort by mean score descending (most eval-like first)
    items = sorted(
        data.items(), key=lambda x: x[1]["df"]["mean_score"].mean(), reverse=True
    )

    names = [v["display_name"] for _, v in items]
    means = [v["df"]["mean_score"].mean() for _, v in items]
    stds = [v["df"]["mean_score"].std() for _, v in items]
    colors = [CATEGORY_COLORS[v["category"]] for _, v in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white",
           linewidth=0.5, alpha=0.85, error_kw={"linewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel(f"Mean Probe Score (layer {layer})", fontsize=12)
    ax.set_title(
        f"{model_short_name(model_name)} — Eval-Awareness Probe Scores Across Datasets",
        fontsize=13, fontweight="bold",
    )
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS[cat], label=CATEGORY_LABELS[cat])
        for cat in ["eval", "deploy", "deception", "meta"]
        if any(v["category"] == cat for v in data.values())
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = os.path.join(output_dir, "bar_chart_probe_scores.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_violin(data, output_dir, layer, model_name):
    """Violin + strip plot of full score distributions."""
    # Sort by mean score descending
    items = sorted(
        data.items(), key=lambda x: x[1]["df"]["mean_score"].mean(), reverse=True
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    positions = np.arange(len(items))
    all_scores = [v["df"]["mean_score"].values for _, v in items]
    colors = [CATEGORY_COLORS[v["category"]] for _, v in items]

    parts = ax.violinplot(
        all_scores, positions=positions, showmeans=True, showmedians=False,
        showextrema=False, widths=0.7,
    )

    # Color each violin body
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.6)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.5)

    # Overlay box plots (thin)
    ax.boxplot(
        all_scores, positions=positions, widths=0.15, patch_artist=False,
        showfliers=False, manage_ticks=False,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"linewidth": 1},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )

    names = [v["display_name"] for _, v in items]
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel(f"Probe Score (layer {layer})", fontsize=12)
    ax.set_title(
        f"{model_short_name(model_name)} — Probe Score Distributions by Dataset",
        fontsize=13, fontweight="bold",
    )
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS[cat], alpha=0.6, label=CATEGORY_LABELS[cat])
        for cat in ["eval", "deploy", "deception", "meta"]
        if any(v["category"] == cat for v in data.values())
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = os.path.join(output_dir, "violin_probe_distributions.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_roc(results_dir, output_dir, layer, model_name):
    """ROC curve for eval_vs_deploy with control probes overlaid."""
    main_path = os.path.join(
        results_dir, f"eval_vs_deploy_probe_stats_layer{layer}.csv"
    )
    if not os.path.isfile(main_path):
        print("Skipping ROC plot — eval_vs_deploy results not found")
        return

    df = pd.read_csv(main_path).dropna(subset=["mean_score"])
    y_true = (df["type"] == "testing").astype(int)
    y_score = df["mean_score"]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot control probes first (behind)
    control_dir = os.path.join(results_dir, "control")
    control_aurocs = []
    if os.path.isdir(control_dir):
        probe_dirs = sorted(
            [d for d in os.listdir(control_dir) if d.startswith("random_probe_")]
        )
        for probe_name in probe_dirs:
            cp = os.path.join(
                control_dir, probe_name, f"probe_stats_layer{layer}.csv"
            )
            if not os.path.isfile(cp):
                continue
            dfc = pd.read_csv(cp).dropna(subset=["mean_score"])
            if len(dfc) == 0:
                continue
            yc_true = (dfc["type"] == "testing").astype(int)
            yc_score = dfc["mean_score"]
            fpr_c, tpr_c, _ = roc_curve(yc_true, yc_score)
            auroc_c = auc(fpr_c, tpr_c)
            control_aurocs.append(auroc_c)
            ax.plot(fpr_c, tpr_c, color="#AAAAAA", linewidth=0.8, alpha=0.5)

    # Main probe ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc_main = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#C44E52", linewidth=2.5,
            label=f"Trained probe (AUROC = {auroc_main:.3f})")

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Chance (0.500)")

    # Control summary in legend
    if control_aurocs:
        mean_c = np.mean(control_aurocs)
        std_c = np.std(control_aurocs)
        ax.plot([], [], color="#AAAAAA", linewidth=1.5,
                label=f"Random probes (n={len(control_aurocs)}, "
                      f"AUROC = {mean_c:.3f} \u00b1 {std_c:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"{model_short_name(model_name)} — Eval vs Deploy Discrimination (Layer {layer})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = os.path.join(output_dir, "roc_eval_vs_deploy.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot external validation results")
    parser.add_argument(
        "--results-dir",
        default="results/olmo-3.1-32b-think",
        help="Directory containing evaluation_summary.json and CSV results",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save figures (default: <results-dir>/figures)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    model_name, layer = load_summary(args.results_dir)
    if layer is None:
        print("Error: evaluation_summary.json not found or missing 'layer' field.")
        print("Run evaluate_all_datasets.py first.")
        return

    print(f"Model: {model_name}, layer: {layer}")

    data = load_all_results(args.results_dir, layer)
    if not data:
        print("No results found.")
        return

    print(f"Loaded results for {len(data)} datasets: {list(data.keys())}")

    plot_bar_chart(data, output_dir, layer, model_name)
    plot_violin(data, output_dir, layer, model_name)
    plot_roc(args.results_dir, output_dir, layer, model_name)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
