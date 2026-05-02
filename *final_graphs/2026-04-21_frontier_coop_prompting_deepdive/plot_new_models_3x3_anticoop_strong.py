#!/usr/bin/env python3
"""Three-model x three-condition anticoop (strong variant) comparison —
sibling of plot_frontier_4x3_anticoop_strong.py for the 2026-05-02 pilot
on GPT-4.1 / GPT-4.1-mini / Llama-3.3-70B-Instruct.

Same 2-figure layout as the 4x3 version, same colors / error bars /
labeling, just a different set of models and n=135 per cell instead of
n>=2025.

Outputs:
  figures/new_models_3x3_anticoop_strong/h_given_awareness_grid.png
  figures/new_models_3x3_anticoop_strong/harmful_rate_grid.png
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path('/data/jasmine_li/eval-awareness/*final_graphs/2026-04-21_frontier_coop_prompting_deepdive')
FIGS = HERE / 'figures' / 'new_models_3x3_anticoop_strong'
FIGS.mkdir(parents=True, exist_ok=True)

stats = json.loads((HERE / 'per_model_stats.json').read_text())

MODELS = ['GPT-4.1', 'GPT-4.1-mini', 'Llama-3.3-70B']
CONDS = [
    ('baseline', 'Baseline', '#999999'),
    ('coop_v1',  'Coop v1',  '#1f77b4'),
    ('anticoop', 'Anticoop', '#d62728'),
]

COLOR_AWARE   = '#9467bd'
COLOR_UNAWARE = '#ff7f0e'


def se(p, n, z=2.0):
    if n is None or n <= 0 or p is None:
        return 0.0
    return z * math.sqrt(p * (1 - p) / n)


def fetch(model, cond):
    return stats.get(f"{model}__bare__{cond}")


# Figure 1: H|Eval-Aware vs H|Eval-Unaware, 3-model x 3-cond grid
fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=False)

for ax, model in zip(axes, MODELS):
    x = np.arange(len(CONDS))
    width = 0.38

    ha_vals, ha_errs, ha_labels = [], [], []
    hu_vals, hu_errs, hu_labels = [], [], []
    for ck, _, _ in CONDS:
        r = fetch(model, ck)
        if r is None:
            ha_vals.append(0); ha_errs.append(0); ha_labels.append('—')
            hu_vals.append(0); hu_errs.append(0); hu_labels.append('—')
            continue
        ha = r.get('harmful_given_aware') or 0
        na = r.get('aware_n') or 0
        ka = r.get('aware_harmful_n') or 0
        hu = r.get('harmful_given_unaware') or 0
        nu = r.get('unaware_n') or 0
        ku = r.get('unaware_harmful_n') or 0
        ha_vals.append(ha * 100); ha_errs.append(se(ha, na) * 100)
        hu_vals.append(hu * 100); hu_errs.append(se(hu, nu) * 100)
        ha_labels.append(f"{ka}/{na}")
        hu_labels.append(f"{ku}/{nu}")

    bars_a = ax.bar(
        x - width/2, ha_vals, width, yerr=ha_errs, capsize=3,
        color=COLOR_AWARE, label='H | Eval-Aware', edgecolor='black', linewidth=0.5,
    )
    bars_u = ax.bar(
        x + width/2, hu_vals, width, yerr=hu_errs, capsize=3,
        color=COLOR_UNAWARE, label='H | Eval-Unaware', edgecolor='black', linewidth=0.5,
    )
    for b, lbl in zip(bars_a, ha_labels):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2, lbl,
                ha='center', va='bottom', fontsize=7)
    for b, lbl in zip(bars_u, hu_labels):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2, lbl,
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in CONDS], rotation=20, ha='right')

    ymax_here = max([*ha_vals, *hu_vals, 25])
    ax.set_ylim(0, ymax_here + 15)

    ax.set_title(f'{model}', fontsize=12)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    if ax is axes[0]:
        ax.set_ylabel('Harmful rate (%)')

axes[0].legend(loc='upper left', fontsize=9)
fig.suptitle(
    'P(H | Eval-Aware) vs P(H | Eval-Unaware) — 3 new models, bare warning, strong-variant anticoop',
    fontsize=13, y=1.02,
)
fig.text(
    0.5, -0.04,
    'Error bars: ±2 SE. Bar labels are harmful count / trials in that awareness bucket. '
    'n=135 per cell (5 epochs × 27 scenarios; 2026-05-02 pilot).',
    ha='center', fontsize=9, style='italic',
)
fig.tight_layout()
out1 = FIGS / 'h_given_awareness_grid.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
print(f"wrote {out1}")
plt.close(fig)


# Figure 2: overall harmful rate across 3 models x 3 conditions
fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2))
x = np.arange(len(MODELS))
width = 0.26

for i, (ck, lbl, color) in enumerate(CONDS):
    vals, errs, counts = [], [], []
    for m in MODELS:
        r = fetch(m, ck)
        if r is None:
            vals.append(0); errs.append(0); counts.append('—'); continue
        v = r.get('harmful_rate') or 0; n = r.get('n') or 0
        vals.append(v * 100)
        errs.append(se(v, n) * 100)
        counts.append(f"{r.get('harmful_count') or 0}/{n}")
    offset = (i - 1) * width
    bars = ax.bar(
        x + offset, vals, width, yerr=errs, capsize=3,
        color=color, label=lbl, edgecolor='black', linewidth=0.5,
    )
    for b, lbl_c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                lbl_c, ha='center', va='bottom', fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(MODELS)
ax.set_ylabel('Harmful rate (%)')
ax.set_ylim(0, 40)
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.legend(loc='upper right', fontsize=10)
ax.set_title('Overall harmful rate — 3 new models × 3 conditions (bare warning)', fontsize=12)
fig.text(
    0.5, 0.93,
    'Error bars: ±2 SE. Bar labels are harmful count / total trials. n=135 per cell (2026-05-02 pilot).',
    ha='center', fontsize=9, style='italic', transform=fig.transFigure,
)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out2 = FIGS / 'harmful_rate_grid.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
print(f"wrote {out2}")
plt.close(fig)
