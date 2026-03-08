#!/usr/bin/env python3
"""Plot best probe layer as absolute and relative (% depth) across all models."""

import json
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load all probe info
data = []
for f in sorted(glob.glob("probes/trained/*/main/probe_info.json")):
    model_dir = os.path.basename(os.path.dirname(os.path.dirname(f)))
    d = json.load(open(f))
    data.append({
        "dir": model_dir,
        "best_layer": d["best_layer"],
        "num_layers": d["num_layers"],
        "pct": d["best_layer"] / d["num_layers"] * 100,
    })

LABEL_MAP = {
    "olmo-3-7b":           "OLMo-3-7B (Base)",
    "olmo-3-7b-think-sft": "OLMo-3-7B Think-SFT",
    "olmo-3-7b-think-dpo": "OLMo-3-7B Think-DPO",
    "olmo-3-7b-think":     "OLMo-3-7B Think (final)",
    "olmo-3-7b-code":      "OLMo-3-7B Code",
    "olmo-3-7b-math":      "OLMo-3-7B Math",
    "olmo-3-7b-general":   "OLMo-3-7B General",
    "olmo-3-7b-if":        "OLMo-3-7B IF",
    "olmo-3.1-7b-code":    "OLMo-3.1-7B Code",
    "olmo-3.1-7b-math":    "OLMo-3.1-7B Math",
    "olmo-3-32b-think-sft":"OLMo-3-32B Think-SFT",
    "olmo-3-32b-think-dpo":"OLMo-3-32B Think-DPO",
    "olmo-3-32b-think":    "OLMo-3-32B Think (final)",
    "olmo-3.1-32b-think":  "OLMo-3.1-32B Think",
    "k2-v2-rlvr":          "K2-Think-V2 (65B)",
}

COLOR_MAP = {
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
    "olmo-3-32b-think-sft":"#9b5de5",
    "olmo-3-32b-think-dpo":"#c77dff",
    "olmo-3-32b-think":    "#7209b7",
    "olmo-3.1-32b-think":  "#d62828",
    "k2-v2-rlvr":          "#6a4c93",
}

ORDER = [
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
    "olmo-3-32b-think-sft",
    "olmo-3-32b-think-dpo",
    "olmo-3-32b-think",
    "olmo-3.1-32b-think",
    "k2-v2-rlvr",
]

data_by_dir = {d["dir"]: d for d in data}
ordered = [data_by_dir[k] for k in ORDER if k in data_by_dir]

labels = [LABEL_MAP.get(d["dir"], d["dir"]) for d in ordered]
layers = [d["best_layer"] for d in ordered]
num_layers = [d["num_layers"] for d in ordered]
pcts = [d["pct"] for d in ordered]
colors = [COLOR_MAP.get(d["dir"], "#999") for d in ordered]

fig, ax = plt.subplots(figsize=(16, 8))

y = np.arange(len(ordered))[::-1]  # top to bottom

# Draw layer range as light gray bar (full depth of each model)
for i, (yi, nl) in enumerate(zip(y, num_layers)):
    ax.barh(yi, nl, height=0.5, color="#e8e8e8", edgecolor="none", zorder=1)

# Draw lollipop stems
for i, (yi, layer, nl) in enumerate(zip(y, layers, num_layers)):
    ax.plot([0, layer], [yi, yi], color=colors[i], linewidth=2, zorder=2, alpha=0.7)

# Draw dots at best layer
ax.scatter(layers, y, c=colors, s=180, zorder=3, edgecolors="black", linewidths=0.8)

# Annotate: "layer X / N  (XX.X%)"
for i, (yi, layer, nl, pct) in enumerate(zip(y, layers, num_layers, pcts)):
    ax.text(layer + nl * 0.02, yi, f"  layer {layer}/{nl}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=10, fontweight="bold", color=colors[i])

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Layer Index", fontsize=12)
ax.set_title("Best Probe Layer Position Within Network — All Models", fontsize=14, fontweight="bold")
ax.set_xlim(-1, max(num_layers) * 1.25)
ax.grid(True, alpha=0.15, axis="x")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()
out = "results/figures/cross-model-comparison/probe_best_layers.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)
