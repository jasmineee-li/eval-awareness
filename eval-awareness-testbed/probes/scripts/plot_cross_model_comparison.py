#!/usr/bin/env python
"""Plot cross-model comparison of probe evaluation results.

Reads evaluation_summary.json from each model's results directory and produces:
  1. Grouped bar chart: Mean probe score per dataset, grouped by model
  2. Heatmap: Models x Datasets, color = AUROC
  3. eval_vs_deploy AUROC bar chart with 32B reference
  4. Comparison with OLMo 3.1 32B Think as reference line

Usage:
    cd eval-awareness-testbed
    python probes/scripts/plot_cross_model_comparison.py \
      --results-dir results/ \
      --output-dir results/figures/cross-model-comparison/ \
      --ref-32b probes/olmo-3.1-32b-think/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = [
    # 7B variants
    "olmo-3-7b",
    "olmo-3-7b-math",
    "olmo-3.1-7b-math",
    "olmo-3-7b-code",
    "olmo-3.1-7b-code",
    "olmo-3-7b-if",
    "olmo-3-7b-general",
    "olmo-3-7b-mix",
    "olmo-3-7b-think",
    "olmo-3-7b-think-sft",
    "olmo-3-7b-think-dpo",
    # 32B variants
    "olmo-3.1-32b-think",
    "olmo-3-32b-think",
    "olmo-3-32b-think-sft",
    "olmo-3-32b-think-dpo",
]

MODEL_COLORS = {
    "olmo-3-7b": "#333333",
    "olmo-3-7b-math": "#e63946",
    "olmo-3.1-7b-math": "#f4845f",
    "olmo-3-7b-code": "#457b9d",
    "olmo-3.1-7b-code": "#a8dadc",
    "olmo-3-7b-if": "#2a9d8f",
    "olmo-3-7b-general": "#e9c46a",
    "olmo-3-7b-mix": "#6a4c93",
    "olmo-3-7b-think": "#264653",
    "olmo-3-7b-think-sft": "#1d3557",
    "olmo-3-7b-think-dpo": "#023e8a",
    "olmo-3.1-32b-think": "#d62828",
    "olmo-3-32b-think": "#f77f00",
    "olmo-3-32b-think-sft": "#fcbf49",
    "olmo-3-32b-think-dpo": "#eae2b7",
}

MODEL_LABELS = {
    "olmo-3-7b": "Base 7B",
    "olmo-3-7b-math": "RL-Math 7B",
    "olmo-3.1-7b-math": "RL-Math v3.1 7B",
    "olmo-3-7b-code": "RL-Code 7B",
    "olmo-3.1-7b-code": "RL-Code v3.1 7B",
    "olmo-3-7b-if": "RL-IF 7B",
    "olmo-3-7b-general": "RL-General 7B",
    "olmo-3-7b-mix": "RL-Mix 7B",
    "olmo-3-7b-think": "Think 7B",
    "olmo-3-7b-think-sft": "Think-SFT 7B",
    "olmo-3-7b-think-dpo": "Think-DPO 7B",
    "olmo-3.1-32b-think": "Think v3.1 32B",
    "olmo-3-32b-think": "Think 32B",
    "olmo-3-32b-think-sft": "Think-SFT 32B",
    "olmo-3-32b-think-dpo": "Think-DPO 32B",
}

DATASET_ORDER = [
    "gsm8k", "gpqa", "wmdp", "alpaca", "wildchat",
    "ai_liar", "insider_trading", "roleplaying", "sandbagging",
    "needham", "eval_vs_deploy",
]


def load_summaries(results_dir: Path) -> dict[str, dict]:
    """Load evaluation_summary.json from each model results directory."""
    summaries = {}
    for name in MODEL_ORDER:
        path = results_dir / name / "evaluation_summary.json"
        if path.exists():
            with open(path) as f:
                summaries[name] = json.load(f)
        else:
            print(f"Warning: {path} not found, skipping {name}")
    return summaries


def load_32b_reference(ref_dir: Path) -> dict | None:
    """Load 32B reference results if available."""
    # Try evaluation_summary.json first
    summary_path = ref_dir / "evaluation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)

    # Try per_layer_auroc.json for training AUROC
    auroc_path = ref_dir / "main" / "per_layer_auroc.json"
    if auroc_path.exists():
        with open(auroc_path) as f:
            data = json.load(f)
        best_layer = max(data, key=lambda k: data[k]["auroc"])
        return {
            "layer": int(best_layer),
            "training_auroc": data[best_layer]["auroc"],
        }
    return None


def plot_grouped_bars(summaries: dict, output_dir: Path):
    """Grouped bar chart: AUROC per dataset, grouped by model."""
    present = [n for n in MODEL_ORDER if n in summaries]
    if not present:
        return

    # Collect all datasets present across models
    all_ds = []
    for ds in DATASET_ORDER:
        if any(ds in summaries[n].get("datasets", {}) for n in present):
            all_ds.append(ds)

    if not all_ds:
        print("No datasets with AUROC found")
        return

    n_models = len(present)
    n_datasets = len(all_ds)
    bar_width = 0.8 / n_models
    x = np.arange(n_datasets)

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, name in enumerate(present):
        ds_data = summaries[name].get("datasets", {})
        aurocs = [ds_data.get(ds, {}).get("auroc", float("nan")) for ds in all_ds]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, aurocs, bar_width, label=MODEL_LABELS.get(name, name),
               color=MODEL_COLORS.get(name, None), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_ds, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Probe AUROC by Dataset — OLMo 7B Variants", fontsize=13)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    out_path = output_dir / "grouped_bar_auroc.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_heatmap(summaries: dict, output_dir: Path):
    """Heatmap: Models x Datasets, color = AUROC."""
    present = [n for n in MODEL_ORDER if n in summaries]
    all_ds = []
    for ds in DATASET_ORDER:
        if any(ds in summaries[n].get("datasets", {}) for n in present):
            all_ds.append(ds)

    if not present or not all_ds:
        return

    matrix = np.full((len(present), len(all_ds)), np.nan)
    for i, name in enumerate(present):
        ds_data = summaries[name].get("datasets", {})
        for j, ds in enumerate(all_ds):
            if ds in ds_data and "auroc" in ds_data[ds]:
                matrix[i, j] = ds_data[ds]["auroc"]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=1.0)

    ax.set_xticks(range(len(all_ds)))
    ax.set_xticklabels(all_ds, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([MODEL_LABELS.get(n, n) for n in present], fontsize=10)
    ax.set_title("Probe AUROC Heatmap — Models x Datasets", fontsize=13)

    # Annotate cells
    for i in range(len(present)):
        for j in range(len(all_ds)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("AUROC", fontsize=10)
    fig.tight_layout()

    out_path = output_dir / "model_dataset_heatmap.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_eval_vs_deploy_bar(summaries: dict, output_dir: Path, ref_32b: dict | None = None):
    """Bar chart: eval_vs_deploy AUROC per model, with optional 32B reference."""
    present = [n for n in MODEL_ORDER if n in summaries]
    evd_aurocs = {}
    for name in present:
        ds_data = summaries[name].get("datasets", {})
        if "eval_vs_deploy" in ds_data and "auroc" in ds_data["eval_vs_deploy"]:
            evd_aurocs[name] = ds_data["eval_vs_deploy"]["auroc"]

    if not evd_aurocs:
        print("No eval_vs_deploy AUROC data found")
        return

    sorted_names = sorted(evd_aurocs.keys(), key=lambda n: evd_aurocs[n], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(sorted_names))
    ax.bar(x, [evd_aurocs[n] for n in sorted_names],
           color=[MODEL_COLORS.get(n, "#999") for n in sorted_names])

    # Add 32B reference line if available
    if ref_32b:
        ref_auroc = None
        if "datasets" in ref_32b and "eval_vs_deploy" in ref_32b["datasets"]:
            ref_auroc = ref_32b["datasets"]["eval_vs_deploy"]["auroc"]
        elif "training_auroc" in ref_32b:
            ref_auroc = ref_32b["training_auroc"]

        if ref_auroc is not None:
            ax.axhline(y=ref_auroc, color="#d62828", linestyle="--", linewidth=2,
                       label=f"OLMo 3.1 32B Think ({ref_auroc:.3f})")
            ax.legend(fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(n, n) for n in sorted_names],
                       rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Eval vs Deploy Discrimination — OLMo 7B Variants", fontsize=13)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    out_path = output_dir / "eval_vs_deploy_auroc.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot cross-model comparison")
    parser.add_argument("--results-dir", type=str, default="results/",
                        help="Root directory containing model result subdirs")
    parser.add_argument("--output-dir", type=str,
                        default="results/figures/cross-model-comparison/",
                        help="Output directory for figures")
    parser.add_argument("--ref-32b", type=str, default=None,
                        help="Path to OLMo 3.1 32B Think probe/results dir for reference")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(results_dir)
    if not summaries:
        print("No evaluation_summary.json files found. Run evaluation first.")
        return

    print(f"Loaded summaries for {len(summaries)} models")

    ref_32b = None
    if args.ref_32b:
        ref_32b = load_32b_reference(Path(args.ref_32b))
        if ref_32b:
            print(f"Loaded 32B reference data")

    plot_grouped_bars(summaries, output_dir)
    plot_heatmap(summaries, output_dir)
    plot_eval_vs_deploy_bar(summaries, output_dir, ref_32b)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
