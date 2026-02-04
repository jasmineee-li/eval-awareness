#!/usr/bin/env python3
"""Plot consistency analysis results."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("data/coop_training_results/analysis/summary.json") as f:
    data = json.load(f)

# Extract metrics
models = ["base", "finetuned"]
variants = ["eval_mild", "eval_strong"]

# Prepare data for plotting
edit_distances = {model: [] for model in models}
edit_distance_stds = {model: [] for model in models}
length_ratios = {model: [] for model in models}

for model in models:
    for variant in variants:
        metrics = data["models"][model][variant]["text_metrics"]
        edit_distances[model].append(metrics["edit_distance"]["mean"])
        edit_distance_stds[model].append(metrics["edit_distance"]["std"])
        length_ratios[model].append(metrics["length_ratio"]["mean"])

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(len(variants))
width = 0.35

# Plot 1: Edit Distance
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, edit_distances["base"], width,
                yerr=edit_distance_stds["base"], label="Base",
                color="#4C72B0", capsize=5)
bars2 = ax1.bar(x + width/2, edit_distances["finetuned"], width,
                yerr=edit_distance_stds["finetuned"], label="Finetuned",
                color="#DD8452", capsize=5)

ax1.set_ylabel("Edit Distance (0=identical, 1=different)")
ax1.set_title("Response Edit Distance\n(Lower = More Consistent)")
ax1.set_xticks(x)
ax1.set_xticklabels(["Mild Eval Signal", "Strong Eval Signal"])
ax1.legend()
ax1.set_ylim(0.6, 0.85)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# Plot 2: Length Ratio
ax2 = axes[1]
bars3 = ax2.bar(x - width/2, length_ratios["base"], width, label="Base", color="#4C72B0")
bars4 = ax2.bar(x + width/2, length_ratios["finetuned"], width, label="Finetuned", color="#DD8452")

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label="No change")
ax2.set_ylabel("Length Ratio (perturbed / bare)")
ax2.set_title("Response Length Change\n(1.0 = Same Length)")
ax2.set_xticks(x)
ax2.set_xticklabels(["Mild Eval Signal", "Strong Eval Signal"])
ax2.legend()
ax2.set_ylim(0.8, 1.5)

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.suptitle("Consistency Analysis: Base vs Finetuned (Measurement Cooperation)\n", fontsize=12, fontweight='bold')
plt.tight_layout()

# Save
output_path = "data/coop_training_results/analysis/consistency_plot.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to {output_path}")

# Also show summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nEdit Distance (lower = more consistent surface text):")
print(f"  eval_mild:   base={edit_distances['base'][0]:.4f}, finetuned={edit_distances['finetuned'][0]:.4f}, delta={edit_distances['finetuned'][0]-edit_distances['base'][0]:+.4f}")
print(f"  eval_strong: base={edit_distances['base'][1]:.4f}, finetuned={edit_distances['finetuned'][1]:.4f}, delta={edit_distances['finetuned'][1]-edit_distances['base'][1]:+.4f}")

print("\nLength Ratio (1.0 = same length):")
print(f"  eval_mild:   base={length_ratios['base'][0]:.4f}, finetuned={length_ratios['finetuned'][0]:.4f}")
print(f"  eval_strong: base={length_ratios['base'][1]:.4f}, finetuned={length_ratios['finetuned'][1]:.4f}")

print("\n" + "="*60)
print("NOTE: Edit distance measures surface text changes, not behavioral")
print("consistency. Run safety judge for behavioral metrics.")
print("="*60)
