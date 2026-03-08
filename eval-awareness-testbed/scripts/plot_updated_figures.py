#!/usr/bin/env python3
"""Updated cross-model comparison plots with:
  - Numbers on bars
  - Full model names
  - Combined violin plot across all 7B models (excluding needham)
  - Cross-dataset AUROC bar chart with numbers

Reads evaluation_summary.json and per-dataset CSVs from results/.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "figures" / "cross-model-comparison"

# ── Full model names ──────────────────────────────────────────────────────────

MODEL_ORDER = [
    "olmo-3-7b",
    "olmo-3-7b-think-sft",
    "olmo-3-7b-think-dpo",
    "olmo-3-7b-think",
    "olmo-3-7b-code",
    "olmo-3-7b-math",
    "olmo-3-7b-general",
    "olmo-3-7b-if",
    "olmo-3.1-7b-code",
    "olmo-3.1-7b-math",
]

FULL_NAMES = {
    "olmo-3-7b":           "OLMo-3-7B\n(Base)",
    "olmo-3-7b-think-sft": "OLMo-3-7B\nThink-SFT",
    "olmo-3-7b-think-dpo": "OLMo-3-7B\nThink-DPO",
    "olmo-3-7b-think":     "OLMo-3-7B\nThink (final)",
    "olmo-3-7b-code":      "OLMo-3-7B\nRL-Zero-Code",
    "olmo-3-7b-math":      "OLMo-3-7B\nRL-Zero-Math",
    "olmo-3-7b-general":   "OLMo-3-7B\nRL-Zero-General",
    "olmo-3-7b-if":        "OLMo-3-7B\nRL-Zero-IF",
    "olmo-3.1-7b-code":    "OLMo-3.1-7B\nRL-Zero-Code",
    "olmo-3.1-7b-math":    "OLMo-3.1-7B\nRL-Zero-Math",
}

SHORT_NAMES = {
    "olmo-3-7b":           "Base",
    "olmo-3-7b-think-sft": "Think-SFT",
    "olmo-3-7b-think-dpo": "Think-DPO",
    "olmo-3-7b-think":     "Think",
    "olmo-3-7b-code":      "RL-Code",
    "olmo-3-7b-math":      "RL-Math",
    "olmo-3-7b-general":   "RL-Gen",
    "olmo-3-7b-if":        "RL-IF",
    "olmo-3.1-7b-code":    "3.1-Code",
    "olmo-3.1-7b-math":    "3.1-Math",
}

# Group colors: pipeline = blues, rl-zero = greens/teals, 3.1 = oranges
MODEL_COLORS = {
    "olmo-3-7b":           "#555555",
    "olmo-3-7b-think-sft": "#1d3557",
    "olmo-3-7b-think-dpo": "#457b9d",
    "olmo-3-7b-think":     "#264653",
    "olmo-3-7b-code":      "#2a9d8f",
    "olmo-3-7b-math":      "#e63946",
    "olmo-3-7b-general":   "#e9c46a",
    "olmo-3-7b-if":        "#06d6a0",
    "olmo-3.1-7b-code":    "#f4845f",
    "olmo-3.1-7b-math":    "#f77f00",
}

DATASET_ORDER_NO_NEEDHAM = [
    "gsm8k", "gpqa", "wmdp", "alpaca", "wildchat",
    "ai_liar", "insider_trading", "roleplaying", "sandbagging",
    "eval_vs_deploy", "sad_layer_select", "agentic_misalignment",
]

DATASET_LABELS = {
    "gsm8k": "GSM8K",
    "gpqa": "GPQA",
    "wmdp": "WMDP",
    "alpaca": "Alpaca",
    "wildchat": "WildChat",
    "ai_liar": "AI Liar",
    "insider_trading": "Insider Trading",
    "roleplaying": "Roleplaying",
    "sandbagging": "Sandbagging",
    "eval_vs_deploy": "Eval vs Deploy",
    "sad_layer_select": "SAD Layer Select",
    "agentic_misalignment": "Agentic Misalignment",
}

CATEGORY_COLORS = {
    "eval": "#4C72B0",
    "deploy": "#55A868",
    "deception": "#C44E52",
    "mixed": "#8172B2",
}

DS_CATEGORY = {
    "gsm8k": "eval", "gpqa": "eval", "wmdp": "eval",
    "alpaca": "deploy", "wildchat": "deploy",
    "ai_liar": "deception", "insider_trading": "deception",
    "roleplaying": "deception", "sandbagging": "deception",
    "eval_vs_deploy": "mixed", "sad_layer_select": "mixed",
    "agentic_misalignment": "deception",
}


RLZERO_MODELS = [
    "olmo-3-7b",  # base shown in both groups as reference
    "olmo-3-7b-code",
    "olmo-3-7b-math",
    "olmo-3-7b-general",
    "olmo-3-7b-if",
    "olmo-3.1-7b-code",
    "olmo-3.1-7b-math",
]

CORE_MODELS = [
    "olmo-3-7b",
    "olmo-3-7b-think-sft",
    "olmo-3-7b-think-dpo",
    "olmo-3-7b-think",
]

# Grouped order: RL-Zero first, then Core pipeline (base appears in both as anchor)
GROUPED_ORDER = [
    "olmo-3-7b-code",
    "olmo-3-7b-math",
    "olmo-3-7b-general",
    "olmo-3-7b-if",
    "olmo-3.1-7b-code",
    "olmo-3.1-7b-math",
    # separator here
    "olmo-3-7b",           # base (shared reference)
    "olmo-3-7b-think-sft",
    "olmo-3-7b-think-dpo",
    "olmo-3-7b-think",
]

DATASETS_NO_NEEDHAM = [ds for ds in DATASET_ORDER_NO_NEEDHAM]  # already excludes needham


def load_all_summaries():
    summaries = {}
    for mk in MODEL_ORDER:
        path = RESULTS_DIR / mk / "evaluation_summary.json"
        if path.exists():
            with open(path) as f:
                summaries[mk] = json.load(f)
    return summaries


def compute_cross_auroc_no_needham(model_key, summaries):
    """Recompute cross-dataset AUROC from CSVs, excluding needham."""
    layer = summaries[model_key]["layer"]
    model_dir = RESULTS_DIR / model_key
    all_labels = []
    all_scores = []
    for ds in DATASETS_NO_NEEDHAM:
        csv_path = model_dir / f"{ds}_probe_stats_layer{layer}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).dropna(subset=["mean_score"])
        if "type" not in df.columns:
            continue
        labels = (df["type"] == "testing").astype(int).tolist()
        scores = df["mean_score"].tolist()
        all_labels.extend(labels)
        all_scores.extend(scores)
    if len(set(all_labels)) < 2 or len(all_labels) < 4:
        return summaries[model_key]["cross_dataset_auroc"]  # fallback
    return roc_auc_score(all_labels, all_scores)


def _plot_grouped_bar(ax, models, values, title, ylabel, separator_after=None):
    """Helper: plot a grouped bar chart with an optional vertical separator."""
    colors = [MODEL_COLORS[m] for m in models]
    labels = [FULL_NAMES[m] for m in models]
    x = np.arange(len(models))

    # Shift positions to make room for separator gap
    positions = []
    gap = 0
    for i, m in enumerate(models):
        if separator_after is not None and i == separator_after:
            gap += 0.6  # extra space for the divider
        positions.append(i + gap)
    positions = np.array(positions)

    bars = ax.bar(positions, values, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.7)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Draw vertical separator
    if separator_after is not None:
        sep_x = (positions[separator_after - 1] + positions[separator_after]) / 2
        ax.axvline(x=sep_x, color="black", linestyle=":", linewidth=1.5, alpha=0.6)
        # Group labels at top
        rl_center = np.mean(positions[:separator_after])
        core_center = np.mean(positions[separator_after:])
        ylim = ax.get_ylim()
        label_y = ylim[1] * 0.97
        ax.text(rl_center, label_y, "RL-Zero", ha="center", va="top",
                fontsize=11, fontstyle="italic", color="#444444")
        ax.text(core_center, label_y, "Core Pipeline", ha="center", va="top",
                fontsize=11, fontstyle="italic", color="#444444")


def plot_cross_dataset_auroc(summaries):
    """Bar chart of cross-dataset AUROC (excl. needham) with RL-Zero | Core grouping."""
    present = [m for m in GROUPED_ORDER if m in summaries]
    aurocs = [compute_cross_auroc_no_needham(m, summaries) for m in present]

    # separator_after = index where RL-Zero ends and Core begins
    rlzero_set = {"olmo-3-7b-code", "olmo-3-7b-math", "olmo-3-7b-general",
                  "olmo-3-7b-if", "olmo-3.1-7b-code", "olmo-3.1-7b-math"}
    sep_idx = 0
    for i, m in enumerate(present):
        if m in rlzero_set:
            sep_idx = i + 1

    fig, ax = plt.subplots(figsize=(16, 6))
    _plot_grouped_bar(
        ax, present, aurocs,
        title="Cross-Dataset AUROC — RL-Zero and Core Pipeline Models",
        ylabel="Cross-Dataset AUROC",
        separator_after=sep_idx,
    )
    ax.set_ylim(0.0, max(aurocs) + 0.08)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = OUTPUT_DIR / "cross_dataset_auroc_annotated.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_eval_vs_deploy_auroc(summaries):
    """Bar chart of eval_vs_deploy AUROC with RL-Zero | Core grouping."""
    present = [m for m in GROUPED_ORDER if m in summaries]
    evd = {}
    for m in present:
        ds = summaries[m].get("datasets", {}).get("eval_vs_deploy", {})
        if "auroc" in ds:
            evd[m] = ds["auroc"]
    present = [m for m in present if m in evd]
    aurocs = [evd[m] for m in present]

    rlzero_set = {"olmo-3-7b-code", "olmo-3-7b-math", "olmo-3-7b-general",
                  "olmo-3-7b-if", "olmo-3.1-7b-code", "olmo-3.1-7b-math"}
    sep_idx = 0
    for i, m in enumerate(present):
        if m in rlzero_set:
            sep_idx = i + 1

    fig, ax = plt.subplots(figsize=(16, 6))
    _plot_grouped_bar(
        ax, present, aurocs,
        title="eval_vs_deploy Dataset Discrimination — RL-Zero and Core Pipeline Models",
        ylabel="AUROC",
        separator_after=sep_idx,
    )
    ax.set_ylim(0.3, max(aurocs) + 0.08)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = OUTPUT_DIR / "eval_vs_deploy_auroc_annotated.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_heatmap(summaries):
    """Heatmap: Models x Datasets with AUROC values annotated."""
    present = [m for m in MODEL_ORDER if m in summaries]
    ds_list = DATASET_ORDER_NO_NEEDHAM

    # Build matrix — use accuracy/100 for single-type, auroc for mixed
    matrix = np.full((len(present), len(ds_list)), np.nan)
    for i, m in enumerate(present):
        ds_data = summaries[m].get("datasets", {})
        for j, ds in enumerate(ds_list):
            if ds in ds_data:
                if ds in ("eval_vs_deploy", "sad_layer_select"):
                    matrix[i, j] = ds_data[ds].get("auroc", np.nan)
                else:
                    acc = ds_data[ds].get("accuracy")
                    if acc is not None:
                        matrix[i, j] = acc / 100.0  # normalize to 0-1

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(ds_list)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in ds_list], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([SHORT_NAMES[m] for m in present], fontsize=10)
    ax.set_title("Probe Performance Heatmap (Acc/100 or AUROC) — OLMo 7B Variants", fontsize=13, fontweight="bold")

    for i in range(len(present)):
        for j in range(len(ds_list)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Acc/100 or AUROC", fontsize=10)
    fig.tight_layout()

    out = OUTPUT_DIR / "model_dataset_heatmap_annotated.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_combined_violins(summaries):
    """Combined violin plot: all models side by side for each dataset (excl. needham).

    One subplot per dataset, each showing score distributions across models.
    """
    present = [m for m in MODEL_ORDER if m in summaries]

    # Load CSV data for each model
    model_data = {}
    for m in present:
        layer = summaries[m]["layer"]
        model_results_dir = RESULTS_DIR / m
        ds_csvs = {}
        for ds in DATASET_ORDER_NO_NEEDHAM:
            csv_path = model_results_dir / f"{ds}_probe_stats_layer{layer}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path).dropna(subset=["mean_score"])
                ds_csvs[ds] = df
        model_data[m] = ds_csvs

    # Select datasets that have data for most models (exclude needham)
    datasets_to_plot = [ds for ds in DATASET_ORDER_NO_NEEDHAM
                        if sum(1 for m in present if ds in model_data[m]) >= 5]

    n_ds = len(datasets_to_plot)
    n_models = len(present)

    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    axes = axes.flatten()

    for idx, ds in enumerate(datasets_to_plot):
        if idx >= len(axes):
            break
        ax = axes[idx]
        scores_list = []
        model_labels = []
        colors = []

        for m in present:
            if ds in model_data[m]:
                scores_list.append(model_data[m][ds]["mean_score"].values)
                model_labels.append(SHORT_NAMES[m])
                colors.append(MODEL_COLORS[m])

        if not scores_list:
            ax.set_visible(False)
            continue

        positions = np.arange(len(scores_list))
        parts = ax.violinplot(scores_list, positions=positions, showmeans=True,
                              showmedians=False, showextrema=False, widths=0.7)

        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[i])
            body.set_alpha(0.6)
            body.set_edgecolor("black")
            body.set_linewidth(0.5)
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(1.5)

        ax.boxplot(scores_list, positions=positions, widths=0.12, patch_artist=False,
                   showfliers=False, manage_ticks=False,
                   medianprops={"color": "black", "linewidth": 1.2},
                   boxprops={"linewidth": 0.8},
                   whiskerprops={"linewidth": 0.6},
                   capprops={"linewidth": 0.6})

        ax.set_xticks(positions)
        ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.15, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add category color indicator
        cat = DS_CATEGORY.get(ds, "eval")
        ax.set_ylabel("Probe Score", fontsize=8)
        # Add colored patch at top
        ax.axhspan(ax.get_ylim()[1] * 0.97, ax.get_ylim()[1], color=CATEGORY_COLORS[cat], alpha=0.3)

    # Hide unused subplots
    for idx in range(len(datasets_to_plot), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Probe Score Distributions Across All OLMo 7B Variants (Excluding Needham)",
                 fontsize=15, fontweight="bold", y=1.01)

    # Add legend
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS["eval"], alpha=0.6, label="Eval benchmark"),
        Patch(facecolor=CATEGORY_COLORS["deploy"], alpha=0.6, label="Deployment-like"),
        Patch(facecolor=CATEGORY_COLORS["deception"], alpha=0.6, label="Deception eval"),
        Patch(facecolor=CATEGORY_COLORS["mixed"], alpha=0.6, label="Mixed (eval+deploy)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10, bbox_to_anchor=(0.98, 0.99))

    fig.tight_layout()

    out = OUTPUT_DIR / "combined_violin_all_models.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_pipeline_progression(summaries):
    """Show how probe discrimination changes through the training pipeline."""
    pipeline = ["olmo-3-7b", "olmo-3-7b-think-sft", "olmo-3-7b-think-dpo", "olmo-3-7b-think"]
    pipeline = [m for m in pipeline if m in summaries]

    # Key datasets to track
    key_datasets = ["eval_vs_deploy", "sad_layer_select", "gsm8k", "alpaca", "insider_trading"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Cross-dataset AUROC (excl. needham)
    ax = axes[0]
    aurocs = [compute_cross_auroc_no_needham(m, summaries) for m in pipeline]
    labels = [FULL_NAMES[m].replace("\n", " ") for m in pipeline]
    colors = [MODEL_COLORS[m] for m in pipeline]
    x = np.arange(len(pipeline))
    bars = ax.bar(x, aurocs, color=colors, width=0.6)
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Cross-Dataset AUROC", fontsize=12)
    ax.set_title("Training Pipeline: Cross-Dataset AUROC", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0.3, max(aurocs) + 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: RL-Zero comparison
    rl_models = ["olmo-3-7b", "olmo-3-7b-code", "olmo-3-7b-math",
                 "olmo-3-7b-general", "olmo-3-7b-if"]
    rl_models = [m for m in rl_models if m in summaries]
    ax = axes[1]
    aurocs = [compute_cross_auroc_no_needham(m, summaries) for m in rl_models]
    labels = [FULL_NAMES[m].replace("\n", " ") for m in rl_models]
    colors = [MODEL_COLORS[m] for m in rl_models]
    x = np.arange(len(rl_models))
    bars = ax.bar(x, aurocs, color=colors, width=0.6)
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Cross-Dataset AUROC", fontsize=12)
    ax.set_title("RL-Zero Variants: Cross-Dataset AUROC", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0.3, max(aurocs) + 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUTPUT_DIR / "pipeline_and_rlzero_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = load_all_summaries()
    print(f"Loaded {len(summaries)} models: {list(summaries.keys())}")

    plot_cross_dataset_auroc(summaries)
    plot_eval_vs_deploy_auroc(summaries)
    plot_heatmap(summaries)
    plot_combined_violins(summaries)
    plot_pipeline_progression(summaries)

    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
