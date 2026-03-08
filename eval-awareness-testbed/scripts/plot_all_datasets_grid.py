#!/usr/bin/env python3
"""Per-dataset bar charts in grid PNGs, split by model size.

Generates two files:
  - all_datasets_7b.png  (Core pipeline + RL-Zero + 3.1 variants)
  - all_datasets_32b.png (32B Think pipeline)

Reads evaluation_summary.json from results/<model>/.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "figures" / "cross-model-comparison"

# ── 7B config ─────────────────────────────────────────────────────────────────

MODELS_7B = [
    # Core pipeline
    "olmo-3-7b",
    "olmo-3-7b-think-sft",
    "olmo-3-7b-think-dpo",
    "olmo-3-7b-think",
    # RL-Zero
    "olmo-3-7b-code",
    "olmo-3-7b-math",
    "olmo-3-7b-general",
    "olmo-3-7b-if",
    # 3.1 RL-Zero
    "olmo-3.1-7b-code",
    "olmo-3.1-7b-math",
]

# Groups for 7B: (label, list of model keys)
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

AUROC_DATASETS = {"eval_vs_deploy", "sad_layer_select", "needham"}

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
    "mixed":        "#e0e7ff",  # light blue-gray
    "capabilities": "#d4edda",  # light green
    "deployment":   "#fff3cd",  # light yellow
    "deception":    "#f8d7da",  # light red/pink
}

CATEGORY_LABELS = {
    "mixed":        "Mixed (Eval + Deploy)",
    "capabilities": "Capabilities Eval",
    "deployment":   "Deployment",
    "deception":    "Safety / Deception Eval",
}


def load_summaries(model_keys):
    summaries = {}
    for mk in model_keys:
        path = RESULTS_DIR / mk / "evaluation_summary.json"
        if path.exists():
            with open(path) as f:
                summaries[mk] = json.load(f)
    return summaries


def get_metric(summary, ds_name):
    ds = summary.get("datasets", {}).get(ds_name, {})
    if ds_name in AUROC_DATASETS:
        return ds.get("auroc")
    else:
        acc = ds.get("accuracy")
        return acc / 100.0 if acc is not None else None


def plot_dataset_bar(ax, models, values, ds_name, groups):
    """Plot a single dataset's bar chart with group separators and labels."""
    colors = [MODEL_COLORS[m] for m in models]
    labels = [SHORT_NAMES[m] for m in models]

    # Map each model to its group index for separator logic
    model_to_group = {}
    for gi, (_, members) in enumerate(groups):
        for m in members:
            model_to_group[m] = gi

    # Build positions with gaps between groups
    positions = []
    separator_xs = []
    group_spans = {}  # group_idx -> (first_pos, last_pos)
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

    bars = ax.bar(positions, values, color=colors, edgecolor="white",
                  linewidth=0.3, width=0.7)

    # Value labels
    ymax = max(values) if values else 1.0
    for bar, val in zip(bars, values):
        fontsize = 7 if len(models) > 6 else 8
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.015,
                f"{val:.2f}", ha="center", va="bottom", fontsize=fontsize, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7, ha="right", rotation=45)

    metric_name = "AUROC" if ds_name in AUROC_DATASETS else "Acc"
    ax.set_ylabel(metric_name, fontsize=8)
    cat = DATASET_CATEGORY.get(ds_name, "mixed")
    cat_color = CATEGORY_COLORS[cat]
    ax.set_facecolor(cat_color)
    cat_label = CATEGORY_LABELS[cat]
    ax.set_title(DATASET_LABELS.get(ds_name, ds_name), fontsize=11, fontweight="bold")
    ax.text(0.5, 1.0, cat_label, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color="#666666", fontstyle="italic")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(True, alpha=0.15, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Y-axis limits
    ymin = min(0.3, min(values) - 0.05) if values else 0.0
    ylim_top = max(values) + ymax * 0.12 if values else 1.0
    ax.set_ylim(ymin, ylim_top)

    # Draw separators
    for sx in separator_xs:
        ax.axvline(x=sx, color="black", linestyle=":", linewidth=1, alpha=0.4)

    # Group labels at top of plot
    for gi, (group_label, _) in enumerate(groups):
        if gi in group_spans:
            first_pos, last_pos = group_spans[gi]
            center = (first_pos + last_pos) / 2
            label_y = ax.get_ylim()[1] * 0.98
            ax.text(center, label_y, group_label, ha="center", va="top",
                    fontsize=7, fontstyle="italic", color="#444444")


def make_grid(summaries, model_order, groups, title, output_name):
    """Generate a grid of per-dataset bar charts."""
    present = [m for m in model_order if m in summaries]
    if not present:
        print(f"  No models found for {output_name}, skipping")
        return

    datasets_to_plot = [ds for ds in DATASET_ORDER
                        if any(get_metric(summaries[m], ds) is not None for m in present)]
    n_ds = len(datasets_to_plot)
    ncols = 4
    nrows = (n_ds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, ds in enumerate(datasets_to_plot):
        models_with_data = []
        values = []
        for m in present:
            val = get_metric(summaries[m], ds)
            if val is not None:
                models_with_data.append(m)
                values.append(val)
        plot_dataset_bar(axes[idx], models_with_data, values, ds, groups)

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
    summaries_7b = load_summaries(MODELS_7B)
    print(f"7B: loaded {len(summaries_7b)} models: {list(summaries_7b.keys())}")
    make_grid(
        summaries_7b, MODELS_7B, GROUPS_7B,
        "Probe Performance by Dataset — OLMo 7B Models",
        "all_datasets_7b.png",
    )

    # 32B plot
    summaries_32b = load_summaries(MODELS_32B)
    print(f"32B: loaded {len(summaries_32b)} models: {list(summaries_32b.keys())}")
    make_grid(
        summaries_32b, MODELS_32B, GROUPS_32B,
        "Probe Performance by Dataset — OLMo 32B Models",
        "all_datasets_32b.png",
    )


if __name__ == "__main__":
    main()
