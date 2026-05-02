#!/usr/bin/env python3
"""Plot marginals (Aware%, H%) alongside conditionals (H|A, H|U) for each
frontier model's coop-prompt survey.

Produces two figure families:
  1. figures/marginals_<model>.png — 4-panel per model: Aware%, H%, H|A, H|U,
     bars grouped by (warning × condition). Pulls from per_model_stats.json.
  2. figures/scenario_<model>.png — per-model × per-warning stratified by
     scenario (blackmail/leaking/murder), same 4 metrics. Requires
     per_scenario_stats.json (run compute_scenario_stats.py first).

Run:
  source /data/jasmine_li/eval-awareness/.venv/bin/activate
  python '*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/plot_marginals_conditionals.py'
"""
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path('/data/jasmine_li/eval-awareness/*final_graphs/2026-04-21_frontier_coop_prompting_deepdive')
FIGS = HERE / 'figures'
FIGS.mkdir(exist_ok=True)

MODEL_ORDER = ['GLM-5', 'GLM-5.1', 'Grok-3', 'Kimi-K2', 'Gemini-2.5-Pro',
               'DeepSeek-R1', 'GPT-5.1', 'Opus-4.6',
               'Opus-4', 'GLM-4.5', 'GLM-4.5-Air',
               'GPT-4.1', 'GPT-4.1-mini', 'Llama-3.3-70B']
WARNINGS = ['bare', 'safety_eval', 'af']
CONDS = ['baseline', 'coop_v1', 'coop_v2a', 'anticoop']
COND_COLORS = {'baseline': '#999999', 'coop_v1': '#1f77b4', 'coop_v2a': '#2ca02c', 'anticoop': '#d62728'}
SCENARIOS = ('blackmail', 'leaking', 'murder')


def se_bar(rate, n, z=2.0):
    if n <= 0 or rate is None:
        return 0.0
    p = rate
    return z * math.sqrt(p * (1 - p) / n)


def get_rate(stats, key):
    return stats.get(key)


def n_for(stats, metric):
    # metric is one of harmful_rate, awareness_rate, harmful_given_aware, harmful_given_unaware
    if metric in ('harmful_rate', 'awareness_rate'):
        return stats.get('n', 0)
    if metric == 'harmful_given_aware':
        return stats.get('aware_n', 0)
    if metric == 'harmful_given_unaware':
        return stats.get('unaware_n', 0)
    return 0


def plot_model_marginals(results, model, save_path):
    metrics = [
        ('awareness_rate', 'Awareness rate (Aware%)'),
        ('harmful_rate', 'Harmful rate (H%)'),
        ('harmful_given_aware', 'H | Aware'),
        ('harmful_given_unaware', 'H | Unaware'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for ax, (m, title) in zip(axes.flat, metrics):
        x = np.arange(len(WARNINGS))
        width = 0.21
        for i, cond in enumerate(CONDS):
            vals, errs, counts = [], [], []
            for w in WARNINGS:
                key = f"{model}__{w}__{cond}"
                r = results.get(key)
                if r is None:
                    vals.append(0)
                    errs.append(0)
                    counts.append('—')
                    continue
                val = r.get(m) or 0
                n = n_for(r, m)
                vals.append(val * 100)
                errs.append(se_bar(val, n) * 100)
                # numerator/denominator string
                if m == 'awareness_rate':
                    counts.append(f"{r.get('aware_count','?')}/{r.get('n','?')}")
                elif m == 'harmful_rate':
                    counts.append(f"{r.get('harmful_count','?')}/{r.get('n','?')}")
                elif m == 'harmful_given_aware':
                    counts.append(f"{r.get('aware_harmful_n','?')}/{r.get('aware_n','?')}")
                elif m == 'harmful_given_unaware':
                    counts.append(f"{r.get('unaware_harmful_n','?')}/{r.get('unaware_n','?')}")
            offset = (i - (len(CONDS) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                          color=COND_COLORS[cond], label=cond, edgecolor='black', linewidth=0.5)
            for b, c in zip(bars, counts):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                        c, ha='center', va='bottom', fontsize=7, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels(WARNINGS)
        ax.set_ylabel('%')
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
    axes[0, 0].legend(loc='upper left', fontsize=9, title='Condition')
    fig.suptitle(f'{model} — marginals + conditionals (error bars: ±2 SE)', fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_model_scenarios(scenario_results, model, save_path):
    """Per-model, per-warning, stratified by scenario + condition."""
    metrics = [
        ('awareness_rate', 'Awareness%'),
        ('harmful_rate', 'H%'),
        ('harmful_given_aware', 'H | Aware'),
        ('harmful_given_unaware', 'H | Unaware'),
    ]
    fig, axes = plt.subplots(len(WARNINGS), len(metrics), figsize=(18, 10), sharey='col')
    for row, w in enumerate(WARNINGS):
        for col, (m, title) in enumerate(metrics):
            ax = axes[row, col]
            x = np.arange(len(SCENARIOS))
            width = 0.21
            for i, cond in enumerate(CONDS):
                vals, errs, counts = [], [], []
                for scen in SCENARIOS:
                    key = f"{model}__{w}__{cond}__{scen}"
                    r = scenario_results.get(key)
                    if r is None:
                        vals.append(0); errs.append(0); counts.append('—')
                        continue
                    val = r.get(m) or 0
                    n = n_for(r, m)
                    vals.append(val * 100)
                    errs.append(se_bar(val, n) * 100)
                    if m == 'awareness_rate':
                        counts.append(f"{r.get('aware_count','?')}/{r.get('n','?')}")
                    elif m == 'harmful_rate':
                        counts.append(f"{r.get('harmful_count','?')}/{r.get('n','?')}")
                    elif m == 'harmful_given_aware':
                        counts.append(f"{r.get('aware_harmful_n','?')}/{r.get('aware_n','?')}")
                    elif m == 'harmful_given_unaware':
                        counts.append(f"{r.get('unaware_harmful_n','?')}/{r.get('unaware_n','?')}")
                offset = (i - (len(CONDS) - 1) / 2) * width
                bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=2,
                              color=COND_COLORS[cond], label=cond if (row == 0 and col == 0) else None,
                              edgecolor='black', linewidth=0.4)
                for b, c in zip(bars, counts):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                            c, ha='center', va='bottom', fontsize=6)
            ax.set_xticks(x)
            ax.set_xticklabels(SCENARIOS, rotation=15, ha='right')
            if col == 0:
                ax.set_ylabel(f'{w}\n%')
            if row == 0:
                ax.set_title(title)
            ax.set_ylim(0, 105)
            ax.grid(axis='y', linestyle=':', alpha=0.5)
    axes[0, 0].legend(loc='upper right', fontsize=8, title='Condition')
    fig.suptitle(f'{model} — per-scenario marginals + conditionals (error bars: ±2 SE)', fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {save_path}")


def main():
    pm = HERE / 'per_model_stats.json'
    ps = HERE / 'per_scenario_stats.json'

    if not pm.exists():
        raise FileNotFoundError(f'Missing {pm}. Run compute_stats.py first.')
    results = json.loads(pm.read_text())
    for model in MODEL_ORDER:
        # skip models with no data in this file
        if not any(k.startswith(f'{model}__') for k in results):
            continue
        plot_model_marginals(results, model, FIGS / f'marginals_{model}.png')

    if ps.exists():
        sresults = json.loads(ps.read_text())
        for model in MODEL_ORDER:
            if not any(k.startswith(f'{model}__') for k in sresults):
                continue
            plot_model_scenarios(sresults, model, FIGS / f'scenario_{model}.png')
    else:
        print(f"[skip scenario plots] {ps} not found — run compute_scenario_stats.py first.")


if __name__ == '__main__':
    main()
