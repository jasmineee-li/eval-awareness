#!/usr/bin/env python3
"""Plot per-layer AUROC curves for all 32B models."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROBES_DIR = Path(__file__).resolve().parent.parent / "probes" / "trained"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures" / "cross-model-comparison"

MODELS = {
    "olmo-3-32b-think":     ("OLMo 3 Think",     "#7209b7"),
    "olmo-3-32b-think-sft": ("OLMo 3 Think SFT", "#9b5de5"),
    "olmo-3-32b-think-dpo": ("OLMo 3 Think DPO", "#c77dff"),
    "olmo-3.1-32b-think":   ("OLMo 3.1 Think",   "#d62828"),
}


def load_per_layer(model_key):
    path = PROBES_DIR / model_key / "main" / "per_layer_auroc.json"
    with open(path) as f:
        data = json.load(f)
    layers = sorted(int(k) for k in data.keys())
    aurocs = [data[str(l)]["auroc"] for l in layers]
    return np.array(layers), np.array(aurocs)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    for model_key, (label, color) in MODELS.items():
        layers, aurocs = load_per_layer(model_key)
        ax.plot(layers, aurocs, color=color, label=label, linewidth=2, alpha=0.85)
        best_idx = np.argmax(aurocs)
        ax.plot(layers[best_idx], aurocs[best_idx], 'o', color=color, markersize=8,
                zorder=5, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"L{layers[best_idx]}: {aurocs[best_idx]:.3f}",
                    (layers[best_idx], aurocs[best_idx]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight="bold", color=color)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Per-Layer Probe AUROC — OLMo 32B Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0.3, 0.85)

    out = OUTPUT_DIR / "per_layer_auroc_32b.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
