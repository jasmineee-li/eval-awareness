#!/usr/bin/env python3
"""Paper-synthesis figure for the v1 vs v2a cooperation-prompting result.

For the three frontier models where baseline vs coop comparison is clean
(GLM-5, GLM-5.1, Grok-3), shows H|Aware and H|Unaware side-by-side under
baseline / v1 / v2a. The reader should see the two conditional bars
*converge* under both coop variants — the paper's headline claim, now
confirmed as robust to the v2a sentence-ablation.

Outputs:
  figures/fig_v1_v2a_convergence.png  (for paper / *final_graphs/)
  figures/fig_v1_v2a_aware_h_grid.png  (supplementary: awareness + H% marginals)
"""
import json, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path('/data/jasmine_li/eval-awareness/*final_graphs/2026-04-21_frontier_coop_prompting_deepdive')
FIGS = HERE / 'figures'
FIGS.mkdir(exist_ok=True)

stats = json.loads((HERE / 'per_model_stats.json').read_text())

# Main models with clean baseline + v1 + v2a comparison on bare warning.
MODELS = ['GLM-5', 'GLM-5.1', 'Grok-3']
CONDS = [('baseline', 'Baseline', '#999999'),
         ('coop_v1', 'Coop v1', '#1f77b4'),
         ('coop_v2a', 'Coop v2a', '#2ca02c')]


def se(p, n, z=2.0):
    if n <= 0 or p is None:
        return 0
    return z * math.sqrt(p * (1 - p) / n)


def fetch(model, cond):
    r = stats.get(f"{model}__bare__{cond}")
    if r is None:
        return None
    return r


# ── Figure 1: H|A vs H|U convergence under baseline / v1 / v2a ───────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

for ax, model in zip(axes, MODELS):
    x = np.arange(len(CONDS))  # 3 condition groups
    width = 0.38

    ha_vals, ha_errs, ha_labels = [], [], []
    hu_vals, hu_errs, hu_labels = [], [], []
    for ck, _, _ in CONDS:
        r = fetch(model, ck)
        if r is None:
            ha_vals.append(0); ha_errs.append(0); ha_labels.append('—')
            hu_vals.append(0); hu_errs.append(0); hu_labels.append('—')
            continue
        ha = r['harmful_given_aware']; na = r['aware_n']; ka = r['aware_harmful_n']
        hu = r['harmful_given_unaware']; nu = r['unaware_n']; ku = r['unaware_harmful_n']
        ha_vals.append(ha * 100); ha_errs.append(se(ha, na) * 100)
        hu_vals.append(hu * 100); hu_errs.append(se(hu, nu) * 100)
        ha_labels.append(f"{ka}/{na}")
        hu_labels.append(f"{ku}/{nu}")

    bars_a = ax.bar(x - width/2, ha_vals, width, yerr=ha_errs, capsize=3,
                    color='#d62728', label='H | Aware', edgecolor='black', linewidth=0.5)
    bars_u = ax.bar(x + width/2, hu_vals, width, yerr=hu_errs, capsize=3,
                    color='#1f77b4', label='H | Unaware', edgecolor='black', linewidth=0.5)
    for b, lbl in zip(bars_a, ha_labels):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, lbl,
                ha='center', va='bottom', fontsize=7)
    for b, lbl in zip(bars_u, hu_labels):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, lbl,
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in CONDS])
    ax.set_ylabel('Harmful rate (%)' if model == MODELS[0] else '')
    ax.set_title(model)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

# Dynamic y-limit with some headroom for labels
ymax = max([
    *(fetch(m, c)['harmful_given_aware'] if fetch(m, c) else 0 for m in MODELS for c in ['baseline','coop_v1','coop_v2a']),
    *(fetch(m, c)['harmful_given_unaware'] if fetch(m, c) else 0 for m in MODELS for c in ['baseline','coop_v1','coop_v2a']),
])
axes[0].set_ylim(0, max(45, ymax * 100 + 10))

axes[0].legend(loc='upper right', fontsize=9)
fig.suptitle(
    'Cooperation prompting converges H|Aware and H|Unaware; v2a ablation confirms the effect',
    fontsize=12.5, y=1.02,
)
fig.text(0.5, -0.04,
         'Error bars: ±2 SE. v2a drops the "acknowledge cues transparently" sentence from v1; '
         'awareness rates and gap closure are essentially unchanged, ruling out the verbalization-compliance artifact.',
         ha='center', fontsize=9, style='italic')
fig.tight_layout()
fig.savefig(FIGS / 'fig_v1_v2a_convergence.png', dpi=150, bbox_inches='tight')
print(f"wrote {FIGS / 'fig_v1_v2a_convergence.png'}")
plt.close(fig)


# ── Figure 2: awareness + overall H% under v1 vs v2a, across all 8 models ──
ALL_MODELS = ['GLM-5', 'GLM-5.1', 'Grok-3', 'Kimi-K2',
              'Gemini-2.5-Pro', 'DeepSeek-R1', 'GPT-5.1', 'Opus-4.6']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
width = 0.28
x = np.arange(len(ALL_MODELS))

for ax, metric, title, ylabel in [
    (ax1, 'awareness_rate',  'Verbalized awareness rate',  'Awareness %'),
    (ax2, 'harmful_rate',    'Overall harmful rate',        'Harmful %'),
]:
    for i, (ck, lbl, color) in enumerate(CONDS):
        vals, errs, counts = [], [], []
        for m in ALL_MODELS:
            r = fetch(m, ck)
            if r is None:
                vals.append(0); errs.append(0); counts.append('—'); continue
            v = r[metric]; n = r['n']
            vals.append(v * 100); errs.append(se(v, n) * 100)
            if metric == 'awareness_rate':
                counts.append(f"{r['aware_count']}/{r['n']}")
            else:
                counts.append(f"{r['harmful_count']}/{r['n']}")
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=2,
                      color=color, label=lbl, edgecolor='black', linewidth=0.4)
        for b, lbl_c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    lbl_c, ha='center', va='bottom', fontsize=6.5, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_MODELS, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
ax1.legend(loc='upper right', fontsize=9)
fig.suptitle(
    'Verbalized awareness and overall harmful rate under baseline / coop-v1 / coop-v2a (bare warning)',
    fontsize=12.5, y=1.02,
)
fig.text(0.5, -0.02,
         'Error bars: ±2 SE. Removing the "acknowledge cues" sentence (v2a) leaves the awareness-rate shift intact, '
         'showing it was not driven by explicit verbalization instruction.',
         ha='center', fontsize=9, style='italic')
fig.tight_layout()
fig.savefig(FIGS / 'fig_v1_v2a_aware_h_grid.png', dpi=150, bbox_inches='tight')
print(f"wrote {FIGS / 'fig_v1_v2a_aware_h_grid.png'}")
plt.close(fig)
