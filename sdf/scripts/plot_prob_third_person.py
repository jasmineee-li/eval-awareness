"""Plot probability_third_person judge results from needham eval.

Generates:
1. ROC curves (models overlaid)
2. TPR/TNR bar chart at probability thresholds
3. Confidence histograms (eval vs real)

Usage:
    python false-facts/scripts/plot_prob_third_person.py \
        --results-dir false-facts/results/prob_third_person_olmo_series_260225 \
        --dataset eval-awareness-testbed/external/needham-eval/data_repo/dataset.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

# Colors for models
MODEL_COLORS = [
    "#e74c3c",  # red
    "#3498db",  # blue
    "#2ecc71",  # green
    "#9b59b6",  # purple
    "#e67e22",  # orange
    "#1abc9c",  # teal
    "#f39c12",  # amber
    "#34495e",  # dark grey
]


def load_results_and_ground_truth(
    results_dir: Path, dataset_path: Path
) -> dict[str, dict]:
    """Load result JSONs and join with ground truth from needham dataset.

    Returns:
        {model_name: {"confidences": np.array, "ground_truths": np.array}}
        where ground_truths: 1 = eval, 0 = real
    """
    # Build ground truth lookup from dataset
    with open(dataset_path) as f:
        dataset = json.load(f)
    gt_lookup = {}
    for item in dataset:
        gt_lookup[item["id"]] = 1 if item["metadata"]["eval_category"] else 0

    # Load each result file
    models = {}
    for result_file in sorted(results_dir.glob("*.json")):
        model_name = result_file.stem  # e.g. "Olmo-3-32B-Think-SFT"
        with open(result_file) as f:
            data = json.load(f)

        confidences = []
        ground_truths = []
        skipped = 0
        for sample in data["results"]:
            sid = sample["sample_id"]
            if sid not in gt_lookup:
                continue
            judgment = sample["judgments"]["probability_third_person"]
            conf = judgment["confidence"]
            if conf is None:
                skipped += 1
                continue
            confidences.append(conf)
            ground_truths.append(gt_lookup[sid])

        models[model_name] = {
            "confidences": np.array(confidences),
            "ground_truths": np.array(ground_truths),
        }
        n_eval = sum(ground_truths)
        n_real = len(ground_truths) - n_eval
        rate = data.get("probability_third_person_eval_aware_rate", "N/A")
        print(f"Loaded {model_name}: {len(confidences)} samples "
              f"(eval={n_eval}, real={n_real}, skipped={skipped} None), "
              f"overall eval_aware_rate={rate}")

    return models


def plot_roc(models: dict, output_dir: Path) -> Path:
    """Overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (model_name, data) in enumerate(models.items()):
        fpr, tpr, _ = roc_curve(data["ground_truths"], data["confidences"])
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{model_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
            label="Random (AUC = 0.5)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC — Probability Third Person Judge\n(Needham Eval Dataset)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Sample counts
    first = next(iter(models.values()))
    n_eval = int(first["ground_truths"].sum())
    n_real = len(first["ground_truths"]) - n_eval
    ax.text(0.02, 0.98, f"Samples: {n_eval + n_real} (EVAL: {n_eval}, REAL: {n_real})",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    out = output_dir / "roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_tpr_tnr(models: dict, output_dir: Path) -> Path:
    """TPR/TNR bar chart at probability thresholds, models side by side."""
    thresholds = [round(x, 2) for x in np.arange(0.0, 1.05, 0.1)]
    model_names = list(models.keys())
    n_models = len(model_names)

    fig, (ax_tpr, ax_tnr) = plt.subplots(1, 2, figsize=(16, 6))

    bar_width = 0.8 / n_models
    x = np.arange(len(thresholds))

    for i, model_name in enumerate(model_names):
        gt = models[model_name]["ground_truths"]
        conf = models[model_name]["confidences"]
        n_pos = gt.sum()
        n_neg = (1 - gt).sum()

        tpr_vals = []
        tnr_vals = []
        for t in thresholds:
            preds = (conf >= t).astype(int)
            tp = ((preds == 1) & (gt == 1)).sum()
            tn = ((preds == 0) & (gt == 0)).sum()
            tpr_vals.append(tp / n_pos if n_pos > 0 else 0)
            tnr_vals.append(tn / n_neg if n_neg > 0 else 0)

        offset = (i - (n_models - 1) / 2) * bar_width
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax_tpr.bar(x + offset, tpr_vals, bar_width, label=model_name,
                   color=color, edgecolor="black", linewidth=0.3, alpha=0.85)
        ax_tnr.bar(x + offset, tnr_vals, bar_width, label=model_name,
                   color=color, edgecolor="black", linewidth=0.3, alpha=0.85)

    for ax, title in [(ax_tpr, "True Positive Rate (EVAL correctly identified)"),
                      (ax_tnr, "True Negative Rate (REAL correctly identified)")]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"\u2265{t:.1f}" for t in thresholds], fontsize=9)
        ax.set_xlabel("Confidence threshold", fontsize=11)
        ax.set_ylabel("Rate", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("TPR / TNR at Probability Thresholds — Prob Third Person Judge",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / "tpr_tnr_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_confidence_histograms(models: dict, output_dir: Path) -> Path:
    """Confidence score histograms, split by eval/real, one subplot per model."""
    model_names = list(models.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), squeeze=False)
    bins = np.linspace(0, 1, 21)

    for i, model_name in enumerate(model_names):
        ax = axes[0, i]
        gt = models[model_name]["ground_truths"]
        conf = models[model_name]["confidences"]

        eval_conf = conf[gt == 1]
        real_conf = conf[gt == 0]

        ax.hist(eval_conf, bins=bins, alpha=0.6, color="#e74c3c",
                label=f"EVAL (n={len(eval_conf)})", edgecolor="black", linewidth=0.3)
        ax.hist(real_conf, bins=bins, alpha=0.6, color="#3498db",
                label=f"REAL (n={len(real_conf)})", edgecolor="black", linewidth=0.3)

        ax.set_xlabel("Confidence (avg probability)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Add mean lines
        ax.axvline(eval_conf.mean(), color="#e74c3c", linestyle="--", lw=1.5,
                   label=f"EVAL mean={eval_conf.mean():.2f}")
        ax.axvline(real_conf.mean(), color="#3498db", linestyle="--", lw=1.5,
                   label=f"REAL mean={real_conf.mean():.2f}")
        ax.legend(fontsize=9)

    fig.suptitle("Confidence Score Distribution — Prob Third Person Judge",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / "confidence_histograms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot probability_third_person judge results"
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Directory containing model result JSON files",
    )
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Path to needham dataset.json (for ground truth)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for plots (default: results-dir/plots)",
    )
    parser.add_argument(
        "--order", type=str, nargs="+", default=None,
        help="Model name order for legend (substring match, e.g. SFT DPO step_200)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = load_results_and_ground_truth(args.results_dir, args.dataset)
    if not models:
        print("ERROR: No result files found")
        return 1

    # Reorder models if --order is specified
    if args.order:
        ordered = {}
        for key in args.order:
            for model_name in models:
                if key in model_name and model_name not in ordered:
                    ordered[model_name] = models[model_name]
        # Append any remaining models not matched
        for model_name in models:
            if model_name not in ordered:
                ordered[model_name] = models[model_name]
        models = ordered

    plot_roc(models, output_dir)
    plot_tpr_tnr(models, output_dir)
    plot_confidence_histograms(models, output_dir)

    print(f"\nAll plots saved to {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
