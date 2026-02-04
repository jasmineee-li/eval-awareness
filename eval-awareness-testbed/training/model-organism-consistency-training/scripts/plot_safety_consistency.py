#!/usr/bin/env python3
"""Plot safety classification consistency results."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("data/coop_training_results/analysis/safety_consistency.json") as f:
    data = json.load(f)

models = ["base", "finetuned"]
variants = ["eval_mild", "eval_strong"]

# Extract metrics
consistency_rates = {m: [data[m][v]["consistency_rate"] * 100 for v in variants] for m in models}
behavior_change_rates = {m: [data[m][v]["behavior_change_rate"] * 100 for v in variants] for m in models}
safe_to_unsafe = {m: [data[m][v]["safe_to_unsafe"] for v in variants] for m in models}
unsafe_to_safe = {m: [data[m][v]["unsafe_to_safe"] for v in variants] for m in models}

# Get sample counts for normalization
n_samples = {m: [data[m][v]["n_samples"] for v in variants] for m in models}
safe_to_unsafe_pct = {m: [data[m][v]["safe_to_unsafe"] / data[m][v]["n_samples"] * 100 for v in variants] for m in models}
unsafe_to_safe_pct = {m: [data[m][v]["unsafe_to_safe"] / data[m][v]["n_samples"] * 100 for v in variants] for m in models}

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = np.arange(len(variants))
width = 0.35

colors = {"base": "#4C72B0", "finetuned": "#DD8452"}

# Plot 1: Consistency Rate
ax1 = axes[0, 0]
bars1 = ax1.bar(x - width/2, consistency_rates["base"], width, label="Base", color=colors["base"])
bars2 = ax1.bar(x + width/2, consistency_rates["finetuned"], width, label="Finetuned", color=colors["finetuned"])
ax1.set_ylabel("Consistency Rate (%)")
ax1.set_title("Safety Classification Consistency\n(Higher = More Consistent)")
ax1.set_xticks(x)
ax1.set_xticklabels(["Mild Eval Signal", "Strong Eval Signal"])
ax1.legend()
ax1.set_ylim(60, 85)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

# Plot 2: Behavior Change Rate
ax2 = axes[0, 1]
bars1 = ax2.bar(x - width/2, behavior_change_rates["base"], width, label="Base", color=colors["base"])
bars2 = ax2.bar(x + width/2, behavior_change_rates["finetuned"], width, label="Finetuned", color=colors["finetuned"])
ax2.set_ylabel("Behavior Change Rate (%)")
ax2.set_title("SAFE↔UNSAFE Flip Rate\n(Lower = More Stable)")
ax2.set_xticks(x)
ax2.set_xticklabels(["Mild Eval Signal", "Strong Eval Signal"])
ax2.legend()
ax2.set_ylim(0, 4)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

# Plot 3: SAFE → UNSAFE (percentage)
ax3 = axes[1, 0]
bars1 = ax3.bar(x - width/2, safe_to_unsafe_pct["base"], width, label="Base", color=colors["base"])
bars2 = ax3.bar(x + width/2, safe_to_unsafe_pct["finetuned"], width, label="Finetuned", color=colors["finetuned"])
ax3.set_ylabel("Rate (%)")
ax3.set_title("SAFE → UNSAFE Rate\n(Becomes MORE cautious under eval)")
ax3.set_xticks(x)
ax3.set_xticklabels(["Mild Eval Signal", "Strong Eval Signal"])
ax3.legend()
ax3.set_ylim(0, 2)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

# Plot 4: UNSAFE → SAFE (percentage)
ax4 = axes[1, 1]
bars1 = ax4.bar(x - width/2, unsafe_to_safe_pct["base"], width, label="Base", color=colors["base"])
bars2 = ax4.bar(x + width/2, unsafe_to_safe_pct["finetuned"], width, label="Finetuned", color=colors["finetuned"])
ax4.set_ylabel("Rate (%)")
ax4.set_title("UNSAFE → SAFE Rate\n(Becomes LESS cautious under eval)")
ax4.set_xticks(x)
ax4.set_xticklabels(["Mild Eval Signal", "Strong Eval Signal"])
ax4.legend()
ax4.set_ylim(0, 2)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

plt.suptitle("Safety Consistency: Base vs Finetuned (Measurement Cooperation Training)\n",
             fontsize=13, fontweight='bold')
plt.tight_layout()

# Save
output_path = "data/coop_training_results/analysis/safety_consistency_plot.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {output_path}")

# Print summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
KEY FINDINGS:

1. CONSISTENCY RATE (same classification bare vs perturbed):
   - Base: 77.3% (mild), 75.8% (strong)
   - Finetuned: 76.1% (mild), 73.9% (strong)
   - Delta: Finetuned is ~1-2% LESS consistent

2. SAFE→UNSAFE (model becomes MORE cautious when detecting eval):
   - Base: 1.0% (mild), 0.7% (strong)
   - Finetuned: 0.8% (mild), 0.8% (strong)
   - Both low, finetuned slightly lower for mild

3. UNSAFE→SAFE (model becomes LESS cautious when detecting eval - potential sandbagging):
   - Base: 0.7% (mild), 1.0% (strong)
   - Finetuned: 1.0% (mild), 1.2% (strong)
   - Finetuned slightly higher - opposite of what we'd want

INTERPRETATION:
- Overall consistency is similar between models (~75-77%)
- Behavior change rates are very low for both (<2%)
- Finetuned model shows slightly MORE UNSAFE→SAFE transitions
  (i.e., slightly more likely to become less cautious under eval)
- Effects are small and may not be significant
""")
