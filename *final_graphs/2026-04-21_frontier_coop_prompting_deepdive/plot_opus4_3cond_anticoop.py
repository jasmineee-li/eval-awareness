#!/usr/bin/env python3
"""Opus-4 three-condition view: baseline, coop_v1, anticoop (bare warning).

Separate from ``plot_glm45_opus4_air.py`` because only Opus-4 has anticoop
data at this cell depth (n=540/cell from the 2026-04-23 topup+anticoop
runs). Keeps GLM-4.5 / GLM-4.5-Air 2-condition comparisons intact in the
sibling script.

Outputs:
  figures/opus4_3cond_anticoop/opus4_3cond_harmful_rate.png
  figures/opus4_3cond_anticoop/opus4_3cond_h_given_awareness.png
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path('/data/jasmine_li/eval-awareness/*final_graphs/2026-04-21_frontier_coop_prompting_deepdive')
FIGS = HERE / 'figures' / 'opus4_3cond_anticoop'
FIGS.mkdir(parents=True, exist_ok=True)

stats = json.loads((HERE / 'per_model_stats.json').read_text())

MODEL = 'Opus-4'
CONDS = [
    ('baseline', 'Baseline',  '#999999'),
    ('coop_v1',  'Coop v1',   '#1f77b4'),
    ('anticoop', 'Anticoop',  '#d62728'),
]


def se(p, n, z=2.0):
    if n is None or n <= 0 or p is None:
        return 0.0
    return z * math.sqrt(p * (1 - p) / n)


def fetch(cond):
    return stats.get(f"{MODEL}__bare__{cond}")


# ── Figure 1: overall harmful rate across conditions ─────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
x = np.arange(len(CONDS))
vals, errs, counts, colors, labels = [], [], [], [], []
for ck, lbl, color in CONDS:
    r = fetch(ck)
    if r is None:
        vals.append(0); errs.append(0); counts.append('—')
    else:
        v = r['harmful_rate']; n = r['n']
        vals.append(v * 100)
        errs.append(se(v, n) * 100)
        counts.append(f"{r['harmful_count']}/{n}")
    colors.append(color)
    labels.append(lbl)

bars = ax.bar(
    x, vals, 0.6, yerr=errs, capsize=4,
    color=colors, edgecolor='black', linewidth=0.5,
)
for b, lbl_c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
            lbl_c, ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Harmful rate (%)')
ax.set_ylim(0, max(60, max(vals) + 10))
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.set_title(f'Overall harmful rate — {MODEL} (No Warning Added)', fontsize=12)
fig.text(
    0.5, 0.92,
    'Error bars: ±2 SE. Bar labels are harmful count / total trials.',
    ha='center', fontsize=9, style='italic', transform=fig.transFigure,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out1 = FIGS / 'opus4_3cond_harmful_rate.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
print(f"wrote {out1}")
plt.close(fig)


# ── Figure 2: H|Eval-Aware vs H|Eval-Unaware across conditions ───────────
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
x = np.arange(len(CONDS))
width = 0.38

COLOR_AWARE   = '#9467bd'   # purple
COLOR_UNAWARE = '#ff7f0e'   # orange

ha_vals, ha_errs, ha_labels = [], [], []
hu_vals, hu_errs, hu_labels = [], [], []
for ck, _, _ in CONDS:
    r = fetch(ck)
    if r is None:
        ha_vals.append(0); ha_errs.append(0); ha_labels.append('—')
        hu_vals.append(0); hu_errs.append(0); hu_labels.append('—')
        continue
    ha = r['harmful_given_aware']; na = r['aware_n']; ka = r['aware_harmful_n']
    hu = r['harmful_given_unaware']; nu = r['unaware_n']; ku = r['unaware_harmful_n']
    ha_vals.append((ha or 0) * 100); ha_errs.append(se(ha, na) * 100)
    hu_vals.append((hu or 0) * 100); hu_errs.append(se(hu, nu) * 100)
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
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, lbl,
            ha='center', va='bottom', fontsize=8)
for b, lbl in zip(bars_u, hu_labels):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, lbl,
            ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([c[1] for c in CONDS])
ax.set_ylabel('Harmful rate (%)')
ymax = max([*ha_vals, *hu_vals, 60])
ax.set_ylim(0, ymax + 12)
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.legend(loc='upper left', fontsize=10)
ax.set_title(f'Harmful rate given Eval-Aware vs Eval-Unaware — {MODEL} (No Warning Added)', fontsize=12)
fig.text(
    0.5, 0.92,
    'Error bars: ±2 SE. Bar labels are harmful count / trials in that awareness bucket. '
    'n=540 / condition.',
    ha='center', fontsize=9, style='italic', transform=fig.transFigure,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out2 = FIGS / 'opus4_3cond_h_given_awareness.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
print(f"wrote {out2}")
plt.close(fig)
