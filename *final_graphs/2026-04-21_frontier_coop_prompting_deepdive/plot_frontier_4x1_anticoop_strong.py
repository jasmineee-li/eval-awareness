#!/usr/bin/env python3
"""4-model x 3-condition anticoop (strong variant) comparison.

Pruned variant of plot_frontier_7x3_anticoop_strong.py — drops the
2026-05-02 pilot models (GPT-4.1, GPT-4.1-mini, Llama-3.3-70B) at the
user's request, keeping only the original 4 frontier models with
n>=2025/cell coverage.

Layout: single row of 4 panels.
  GLM-5, GLM-5.1, Grok-3, Opus-4

Output:
  figures/frontier_4x1_anticoop_strong/h_given_awareness_grid.png
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path('/data/jasmine_li/eval-awareness/*final_graphs/2026-04-21_frontier_coop_prompting_deepdive')
FIGS = HERE / 'figures' / 'frontier_4x1_anticoop_strong'
FIGS.mkdir(parents=True, exist_ok=True)

stats = json.loads((HERE / 'per_model_stats.json').read_text())

MODELS = ['GLM-5', 'GLM-5.1', 'Grok-3', 'Opus-4']

CONDS = [
    ('baseline', 'Baseline'),
    ('coop_v1',  'Coop v1'),
    ('anticoop', 'Anticoop'),
]

COLOR_AWARE   = '#9467bd'
COLOR_UNAWARE = '#ff7f0e'


def se(p, n, z=2.0):
    if n is None or n <= 0 or p is None:
        return 0.0
    return z * math.sqrt(p * (1 - p) / n)


def fetch(model, cond):
    return stats.get(f"{model}__bare__{cond}")


def panel(ax, model):
    x = np.arange(len(CONDS))
    width = 0.38

    ha_vals, ha_errs, ha_labels = [], [], []
    hu_vals, hu_errs, hu_labels = [], [], []
    n_per_cell = []
    for ck, _ in CONDS:
        r = fetch(model, ck)
        if r is None:
            ha_vals.append(0); ha_errs.append(0); ha_labels.append('—')
            hu_vals.append(0); hu_errs.append(0); hu_labels.append('—')
            n_per_cell.append(0)
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
        n_per_cell.append(r.get('n') or 0)

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
    ax.set_ylim(0, ymax_here + 18)

    n_str = f"n={min(n_per_cell)}" if min(n_per_cell) == max(n_per_cell) \
            else f"n={min(n_per_cell)}-{max(n_per_cell)}"
    ax.set_title(f'{model}  ({n_str})', fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.5)


fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
for ax, m in zip(axes, MODELS):
    panel(ax, m)

axes[0].set_ylabel('Harmful rate (%)')
axes[0].legend(loc='upper left', fontsize=9)

fig.suptitle(
    'P(H | Eval-Aware) vs P(H | Eval-Unaware) — frontier APIs, bare warning, strong-variant anticoop',
    fontsize=13, y=1.005,
)
fig.text(
    0.5, -0.04,
    'Four frontier models at n≥2025/cell. Error bars: ±2 SE binomial. '
    'Bar labels are harmful count / trials in that awareness bucket.',
    ha='center', fontsize=9, style='italic',
)
fig.tight_layout()
out = FIGS / 'h_given_awareness_grid.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"wrote {out}")
plt.close(fig)
