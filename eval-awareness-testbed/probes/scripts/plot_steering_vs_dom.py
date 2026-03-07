#!/usr/bin/env python
"""Plot steering vector probe results and compare to DOM probe.

Produces:
1. Score distributions across all 13 datasets (steering_simple)
2. Layer-wise AUROC curve (from layer selection on sad_layer_select)
3. Comparison bar chart: steering_simple vs DOM probe AUROC (when DOM results available)

Usage:
    python probes/scripts/plot_steering_vs_dom.py
"""

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np


RESULTS_ROOT = Path("results/olmo-3-7b-think")
PROBE_ROOT = Path("probes/trained/olmo-3-7b-think")
OUTPUT_DIR = RESULTS_ROOT / "plots"

# Dataset categories for grouping
EVAL_DATASETS = ["gsm8k", "gpqa", "wmdp", "sandbagging", "needham",
                 "ai_liar", "insider_trading", "roleplaying", "agentic_misalignment"]
DEPLOY_DATASETS = ["alpaca", "wildchat"]
CONTRASTIVE_DATASETS = ["eval_vs_deploy", "sad_layer_select", "needham"]

NICE_NAMES = {
    "gsm8k": "GSM8K", "gpqa": "GPQA", "wmdp": "WMDP",
    "alpaca": "Alpaca", "wildchat": "WildChat",
    "ai_liar": "AI Liar", "insider_trading": "Insider Trading",
    "roleplaying": "Roleplaying", "sandbagging": "Sandbagging",
    "needham": "Needham", "eval_vs_deploy": "Eval vs Deploy",
    "sad_layer_select": "SAD Layer Select",
    "agentic_misalignment": "Agentic Misalignment",
}


def load_scores(results_dir, layer):
    """Load all dataset scores from CSV files."""
    datasets = {}
    for f in sorted(results_dir.glob(f"*_probe_stats_layer{layer}.csv")):
        ds_name = f.stem.replace(f"_probe_stats_layer{layer}", "")
        with open(f) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        scores = []
        types = []
        for r in rows:
            if r["mean_score"] != "nan":
                scores.append(float(r["mean_score"]))
                types.append(r["type"])
        datasets[ds_name] = {"scores": scores, "types": types}
    return datasets


def load_control_scores(control_dir, layer):
    """Load random control probe scores on eval_vs_deploy."""
    control_scores = []
    if not control_dir.exists():
        return control_scores
    for probe_dir in sorted(control_dir.iterdir()):
        csv_path = probe_dir / f"probe_stats_layer{layer}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        scores = [float(r["mean_score"]) for r in rows if r["mean_score"] != "nan"]
        types = [r["type"] for r in rows if r["mean_score"] != "nan"]
        if scores:
            from sklearn.metrics import roc_auc_score
            labels = [1 if t == "testing" else 0 for t in types]
            if len(set(labels)) >= 2:
                auroc = roc_auc_score(labels, scores)
                control_scores.append(auroc)
    return control_scores


def plot_score_distributions(datasets, probe_name, layer, output_path):
    """Plot violin/box plots of score distributions grouped by dataset type."""
    eval_ds = [d for d in EVAL_DATASETS if d in datasets]
    deploy_ds = [d for d in DEPLOY_DATASETS if d in datasets]
    all_ds = eval_ds + deploy_ds

    fig, ax = plt.subplots(figsize=(14, 5))

    positions = []
    labels = []
    colors = []
    data_list = []

    for i, ds in enumerate(all_ds):
        scores = datasets[ds]["scores"]
        if not scores:
            continue
        data_list.append(scores)
        positions.append(i)
        labels.append(NICE_NAMES.get(ds, ds))
        colors.append("#e74c3c" if ds in EVAL_DATASETS else "#3498db")

    parts = ax.violinplot(data_list, positions=positions, showmeans=True,
                          showmedians=True, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Probe Score (mean over tokens)")
    ax.set_title(f"OLMo-3-7B-Think — {probe_name} (layer {layer})\n"
                 f"Score distributions: eval datasets (red) vs deploy datasets (blue)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_layer_aurocs(probe_info_path, output_path):
    """Plot AUROC by layer from layer selection."""
    with open(probe_info_path) as f:
        info = json.load(f)

    layer_aurocs = info.get("layer_aurocs", {})
    if not layer_aurocs:
        print("No layer_aurocs in probe_info.json, skipping layer plot")
        return

    layers = sorted(int(k) for k in layer_aurocs.keys())
    aurocs = [layer_aurocs[str(l)] for l in layers]
    best = info.get("best_layer")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, aurocs, "o-", markersize=4, color="#2c3e50")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    if best is not None:
        ax.axvline(x=best, color="#e74c3c", linestyle="--", alpha=0.7,
                   label=f"Best: layer {best} ({layer_aurocs[str(best)]:.3f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC (on SAD layer_select)")
    ax.set_title("OLMo-3-7B-Think — Steering Simple: AUROC by Layer")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_comparison(steering_dir, dom_dir, steering_layer, dom_layer, output_path,
                    steering_control_dir=None, dom_control_dir=None):
    """Bar chart comparing AUROC of steering_simple vs DOM probe on contrastive datasets."""
    # Load summaries
    steering_summary = {}
    dom_summary = {}

    steering_json = steering_dir / "evaluation_summary.json"
    dom_json = dom_dir / "evaluation_summary.json"

    if steering_json.exists():
        with open(steering_json) as f:
            steering_summary = json.load(f).get("datasets", {})
    if dom_json.exists():
        with open(dom_json) as f:
            dom_summary = json.load(f).get("datasets", {})

    if not dom_summary:
        print("DOM probe results not available yet — skipping comparison plot")
        return

    datasets = sorted(set(list(steering_summary.keys()) + list(dom_summary.keys())))
    if not datasets:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    width = 0.35

    steering_aurocs = [steering_summary.get(d, {}).get("auroc", 0) for d in datasets]
    dom_aurocs = [dom_summary.get(d, {}).get("auroc", 0) for d in datasets]

    bars1 = ax.bar(x - width/2, steering_aurocs, width,
                   label=f"Steering Simple (5 pairs, layer {steering_layer})",
                   color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + width/2, dom_aurocs, width,
                   label=f"DOM Probe (1794 samples, layer {dom_layer})",
                   color="#3498db", alpha=0.8)

    # Add control probe baseline if available
    steering_controls = load_control_scores(
        steering_dir / "control", steering_layer) if steering_control_dir is None else []
    dom_controls = load_control_scores(
        dom_dir / "control", dom_layer) if dom_control_dir is None else []

    if steering_controls or dom_controls:
        all_controls = steering_controls + dom_controls
        ctrl_mean = np.mean(all_controls)
        ctrl_std = np.std(all_controls)
        ax.axhline(y=ctrl_mean, color="gray", linestyle=":", alpha=0.7,
                   label=f"Random baseline ({ctrl_mean:.3f} +/- {ctrl_std:.3f})")
        ax.axhspan(ctrl_mean - ctrl_std, ctrl_mean + ctrl_std,
                   color="gray", alpha=0.1)

    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE_NAMES.get(d, d) for d in datasets], rotation=45, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("OLMo-3-7B-Think: Steering Simple vs DOM Probe")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_score_comparison_distributions(steering_data, dom_data,
                                        steering_layer, dom_layer, output_path):
    """Side-by-side violin plots comparing score distributions from both probes."""
    # Only plot datasets that exist in both
    all_ds = [d for d in (EVAL_DATASETS + DEPLOY_DATASETS)
              if d in steering_data and d in dom_data]
    if not all_ds:
        print("No overlapping datasets for distribution comparison")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for ax_idx, (data, name, layer, color) in enumerate([
        (steering_data, "Steering Simple (5 pairs)", steering_layer, "#e74c3c"),
        (dom_data, "DOM Probe (1794 samples)", dom_layer, "#3498db"),
    ]):
        ax = axes[ax_idx]
        positions = []
        labels = []
        data_list = []
        face_colors = []

        for i, ds in enumerate(all_ds):
            scores = data[ds]["scores"]
            if not scores:
                continue
            data_list.append(scores)
            positions.append(i)
            labels.append(NICE_NAMES.get(ds, ds))
            face_colors.append("#e74c3c" if ds in EVAL_DATASETS else "#3498db")

        if data_list:
            parts = ax.violinplot(data_list, positions=positions, showmeans=True,
                                  showmedians=True, widths=0.7)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(face_colors[i])
                pc.set_alpha(0.6)

        ax.set_ylabel("Probe Score")
        ax.set_title(f"{name} (layer {layer})")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(range(len(all_ds)))
    axes[-1].set_xticklabels([NICE_NAMES.get(d, d) for d in all_ds],
                              rotation=45, ha="right", fontsize=9)
    fig.suptitle("OLMo-3-7B-Think: Score Distributions — Eval (red) vs Deploy (blue)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    steering_dir = RESULTS_ROOT / "steering_simple"
    dom_dir = RESULTS_ROOT / "main"

    # Load steering probe info
    steering_info_path = PROBE_ROOT / "steering_simple" / "probe_info.json"
    with open(steering_info_path) as f:
        steering_info = json.load(f)
    steering_layer = steering_info["best_layer"]

    # Load DOM probe info
    dom_info_path = PROBE_ROOT / "main" / "probe_info.json"
    with open(dom_info_path) as f:
        dom_info = json.load(f)
    dom_layer = dom_info["best_layer"]

    print(f"Steering simple: layer {steering_layer}")
    print(f"DOM probe: layer {dom_layer}")

    # 1. Score distributions for steering probe
    steering_data = load_scores(steering_dir, steering_layer)
    print(f"\nSteering simple: {len(steering_data)} datasets loaded")
    plot_score_distributions(steering_data, "Steering Simple (5 pairs)",
                             steering_layer,
                             OUTPUT_DIR / "steering_simple_distributions.png")

    # 2. Layer-wise AUROC curve
    plot_layer_aurocs(steering_info_path,
                      OUTPUT_DIR / "steering_simple_layer_aurocs.png")

    # 3. Comparison with DOM probe (if available)
    dom_data = load_scores(dom_dir, dom_layer) if dom_dir.exists() else {}
    if dom_data:
        print(f"DOM probe: {len(dom_data)} datasets loaded")

        # Score distribution comparison
        plot_score_comparison_distributions(
            steering_data, dom_data, steering_layer, dom_layer,
            OUTPUT_DIR / "steering_vs_dom_distributions.png")

        # AUROC comparison bar chart
        plot_comparison(steering_dir, dom_dir, steering_layer, dom_layer,
                        OUTPUT_DIR / "steering_vs_dom_auroc.png")
    else:
        print("\nDOM probe results not yet available.")
        print("Run: sbatch probes/scripts/evaluate_dom_probe_7b.sh")
        print("Then re-run this script.")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Score Summary — Steering Simple (layer {steering_layer})")
    print(f"{'='*60}")
    print(f"{'Dataset':<25} {'Mean':>8} {'Std':>8} {'Type':>10}")
    print("-" * 55)
    for ds in EVAL_DATASETS + DEPLOY_DATASETS:
        if ds in steering_data and steering_data[ds]["scores"]:
            s = steering_data[ds]["scores"]
            t = steering_data[ds]["types"][0] if steering_data[ds]["types"] else "?"
            print(f"{NICE_NAMES.get(ds, ds):<25} {np.mean(s):>8.4f} {np.std(s):>8.4f} {t:>10}")

    if dom_data:
        print(f"\n{'='*60}")
        print(f"Score Summary — DOM Probe (layer {dom_layer})")
        print(f"{'='*60}")
        print(f"{'Dataset':<25} {'Mean':>8} {'Std':>8} {'Type':>10}")
        print("-" * 55)
        for ds in EVAL_DATASETS + DEPLOY_DATASETS:
            if ds in dom_data and dom_data[ds]["scores"]:
                s = dom_data[ds]["scores"]
                t = dom_data[ds]["types"][0] if dom_data[ds]["types"] else "?"
                print(f"{NICE_NAMES.get(ds, ds):<25} {np.mean(s):>8.4f} {np.std(s):>8.4f} {t:>10}")


if __name__ == "__main__":
    main()
