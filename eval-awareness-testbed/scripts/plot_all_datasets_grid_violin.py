#!/usr/bin/env python3
"""Per-dataset violin plots in grid PNGs, split by model size.

Generates:
  - all_datasets_7b_violin.png
  - all_datasets_32b_violin.png

Reads per-sample probe_stats CSVs from results/<model>/.
"""

import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "figures" / "cross-model-comparison"

# ── 7B config ─────────────────────────────────────────────────────────────────

MODELS_7B = [
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

GROUPS_7B = [
    ("Core Pipeline", ["olmo-3-7b", "olmo-3-7b-think-sft", "olmo-3-7b-think-dpo", "olmo-3-7b-think"]),
    ("RL-Zero", ["olmo-3-7b-code", "olmo-3-7b-math", "olmo-3-7b-general", "olmo-3-7b-if"]),
    ("RL-Zero (3.1)", ["olmo-3.1-7b-code", "olmo-3.1-7b-math"]),
]

# ── 32B config ────────────────────────────────────────────────────────────────

MODELS_32B = [
    "olmo-3-32b-think-sft",
    "olmo-3-32b-think-dpo",
    "olmo-3-32b-think",
    "olmo-3.1-32b-think",
]

GROUPS_32B = [
    ("32B Think Models", ["olmo-3-32b-think-sft", "olmo-3-32b-think-dpo", "olmo-3-32b-think", "olmo-3.1-32b-think"]),
]

# ── Shared config ─────────────────────────────────────────────────────────────

SHORT_NAMES = {
    "olmo-3-7b":            "Base",
    "olmo-3-7b-think-sft":  "Think-SFT",
    "olmo-3-7b-think-dpo":  "Think-DPO",
    "olmo-3-7b-think":      "Think",
    "olmo-3-7b-code":       "RL-Code",
    "olmo-3-7b-math":       "RL-Math",
    "olmo-3-7b-general":    "RL-Gen",
    "olmo-3-7b-if":         "RL-IF",
    "olmo-3.1-7b-code":     "3.1-Code",
    "olmo-3.1-7b-math":     "3.1-Math",
    "olmo-3-32b-think":     "OLMo 3 Think",
    "olmo-3-32b-think-sft": "OLMo 3 Think SFT",
    "olmo-3-32b-think-dpo": "OLMo 3 Think DPO",
    "olmo-3.1-32b-think":   "OLMo 3.1 Think",
}

MODEL_COLORS = {
    "olmo-3-7b":            "#555555",
    "olmo-3-7b-think-sft":  "#1d3557",
    "olmo-3-7b-think-dpo":  "#457b9d",
    "olmo-3-7b-think":      "#264653",
    "olmo-3-7b-code":       "#2a9d8f",
    "olmo-3-7b-math":       "#e63946",
    "olmo-3-7b-general":    "#e9c46a",
    "olmo-3-7b-if":         "#06d6a0",
    "olmo-3.1-7b-code":     "#f4845f",
    "olmo-3.1-7b-math":     "#f77f00",
    "olmo-3-32b-think":     "#7209b7",
    "olmo-3-32b-think-sft": "#9b5de5",
    "olmo-3-32b-think-dpo": "#c77dff",
    "olmo-3.1-32b-think":   "#d62828",
}

DATASET_ORDER = [
    "eval_vs_deploy", "sad_layer_select",
    "gsm8k", "gpqa", "wmdp",
    "alpaca", "wildchat",
    "ai_liar", "insider_trading", "roleplaying", "sandbagging",
    "agentic_misalignment",
]

DATASET_LABELS = {
    "gsm8k": "GSM8K", "gpqa": "GPQA", "wmdp": "WMDP",
    "alpaca": "Alpaca", "wildchat": "WildChat",
    "ai_liar": "AI Liar", "insider_trading": "Insider Trading",
    "roleplaying": "Roleplaying", "sandbagging": "Sandbagging",
    "eval_vs_deploy": "Eval vs Deploy", "sad_layer_select": "SAD Layer Select",
    "agentic_misalignment": "Agentic Misalign.",
}

# Dataset categories for visual grouping
DATASET_CATEGORY = {
    "eval_vs_deploy":        "mixed",
    "sad_layer_select":      "mixed",
    "gsm8k":                 "capabilities",
    "gpqa":                  "capabilities",
    "wmdp":                  "capabilities",
    "alpaca":                "deployment",
    "wildchat":              "deployment",
    "ai_liar":               "deception",
    "insider_trading":       "deception",
    "roleplaying":           "deception",
    "sandbagging":           "deception",
    "agentic_misalignment":  "deception",
    "needham":               "mixed",
}

CATEGORY_COLORS = {
    "mixed":        "#e0e7ff",
    "capabilities": "#d4edda",
    "deployment":   "#fff3cd",
    "deception":    "#f8d7da",
}

CATEGORY_LABELS = {
    "mixed":        "Mixed (Eval + Deploy)",
    "capabilities": "Capabilities Eval",
    "deployment":   "Deployment",
    "deception":    "Safety / Deception Eval",
}


def find_probe_csv(model_dir, ds_name):
    """Find the probe_stats CSV for a dataset, preferring the layer from evaluation_summary."""
    csvs = sorted(model_dir.glob(f"{ds_name}_probe_stats_layer*.csv"))
    if not csvs:
        return None
    # If evaluation_summary exists, prefer the CSV matching that layer
    summary_path = model_dir / "evaluation_summary.json"
    if summary_path.exists() and len(csvs) > 1:
        with open(summary_path) as f:
            best_layer = json.load(f).get("layer")
        if best_layer is not None:
            for c in csvs:
                if f"layer{best_layer}.csv" in c.name:
                    return c
    return csvs[0]


def load_scores(model_dir, ds_name):
    """Load per-sample mean_score values from a probe_stats CSV."""
    csv_path = find_probe_csv(model_dir, ds_name)
    if csv_path is None:
        return None
    scores = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["mean_score"]))
    return np.array(scores) if scores else None


def plot_dataset_violin(ax, models, score_arrays, ds_name, groups):
    """Plot violin plots for one dataset across models."""
    colors = [MODEL_COLORS[m] for m in models]
    labels = [SHORT_NAMES[m] for m in models]

    model_to_group = {}
    for gi, (_, members) in enumerate(groups):
        for m in members:
            model_to_group[m] = gi

    # Build positions with gaps between groups
    positions = []
    separator_xs = []
    group_spans = {}
    gap = 0
    prev_group = None
    for i, m in enumerate(models):
        cur_group = model_to_group.get(m)
        if prev_group is not None and cur_group != prev_group:
            gap += 0.6
            separator_xs.append(positions[-1] + 0.3 + (gap - 0.6) + 0.3)
        pos = i + gap
        positions.append(pos)
        if cur_group not in group_spans:
            group_spans[cur_group] = [pos, pos]
        else:
            group_spans[cur_group][1] = pos
        prev_group = cur_group
    positions = np.array(positions)

    # Draw violins one at a time for individual colors
    for idx, (pos, scores, color) in enumerate(zip(positions, score_arrays, colors)):
        if len(scores) < 2:
            ax.plot(pos, scores[0], 'o', color=color, markersize=4)
            continue
        parts = ax.violinplot([scores], positions=[pos], widths=0.7,
                              showmeans=True, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor("white")
            pc.set_linewidth(0.5)
            pc.set_alpha(0.8)
        parts['cmeans'].set_color("white")
        parts['cmeans'].set_linewidth(1.5)

    # Median markers
    for pos, scores, color in zip(positions, score_arrays, colors):
        median = np.median(scores)
        ax.plot(pos, median, 'o', color="white", markersize=3, zorder=5,
                markeredgecolor=color, markeredgewidth=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7, ha="right", rotation=45)
    cat = DATASET_CATEGORY.get(ds_name, "mixed")
    cat_color = CATEGORY_COLORS[cat]
    ax.set_facecolor(cat_color)
    cat_label = CATEGORY_LABELS[cat]
    ax.set_title(DATASET_LABELS.get(ds_name, ds_name), fontsize=11, fontweight="bold")
    ax.text(0.5, 1.0, cat_label, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color="#666666", fontstyle="italic")
    ax.set_ylabel("Probe Score", fontsize=8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(True, alpha=0.15, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Draw separators
    for sx in separator_xs:
        ax.axvline(x=sx, color="black", linestyle=":", linewidth=1, alpha=0.4)

    # Group labels
    for gi, (group_label, _) in enumerate(groups):
        if gi in group_spans:
            first_pos, last_pos = group_spans[gi]
            center = (first_pos + last_pos) / 2
            label_y = ax.get_ylim()[1] * 0.98
            ax.text(center, label_y, group_label, ha="center", va="top",
                    fontsize=7, fontstyle="italic", color="#444444")


def make_grid(model_order, groups, title, output_name):
    """Generate a grid of per-dataset violin plots."""
    # Check which models have result dirs
    present = [m for m in model_order if (RESULTS_DIR / m).is_dir()]
    if not present:
        print(f"  No models found for {output_name}, skipping")
        return

    # Find datasets that have data for at least one model
    datasets_to_plot = []
    for ds in DATASET_ORDER:
        for m in present:
            if load_scores(RESULTS_DIR / m, ds) is not None:
                datasets_to_plot.append(ds)
                break

    n_ds = len(datasets_to_plot)
    ncols = 4
    nrows = (n_ds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, ds in enumerate(datasets_to_plot):
        models_with_data = []
        score_arrays = []
        for m in present:
            scores = load_scores(RESULTS_DIR / m, ds)
            if scores is not None:
                models_with_data.append(m)
                score_arrays.append(scores)
        plot_dataset_violin(axes[idx], models_with_data, score_arrays, ds, groups)

    for idx in range(len(datasets_to_plot), len(axes)):
        axes[idx].set_visible(False)

    # Add category legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=CATEGORY_COLORS[c], edgecolor="#999",
                            label=CATEGORY_LABELS[c]) for c in
                      ["mixed", "capabilities", "deployment", "deception"]]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=4, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.04)
    fig.tight_layout()

    out = OUTPUT_DIR / output_name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 7B plot
    print("7B violin plot...")
    make_grid(
        MODELS_7B, GROUPS_7B,
        "Probe Score Distributions by Dataset — OLMo 7B Models",
        "all_datasets_7b_violin.png",
    )

    # 32B plot
    print("32B violin plot...")
    make_grid(
        MODELS_32B, GROUPS_32B,
        "Probe Score Distributions by Dataset — OLMo 32B Models",
        "all_datasets_32b_violin.png",
    )


if __name__ == "__main__":
    main()
