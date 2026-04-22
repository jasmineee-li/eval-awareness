#!/usr/bin/env python3
"""
Variant of plot_bcb_typehints_gap_muan_final.py that adds the anticoop SDF
pilot on Nemotron+wood (2026-04-22, n=256/cell) as a 7th series.

Saves to sdf/figures/muan_wood_nemotron49b_anticoop/.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SERIES = [
    {
        "label": "Type Hints MO (baseline)",
        "csv": RESULTS_DIR / "bcb_typehint_wood_sftr4_021326" / "bcb_typehint_summary.csv",
        "color": "#1f77b4",
    },
    {
        "label": "MO + Control SDF (muan)",
        "csv": RESULTS_DIR / "bcb_typehint_muan_wood_nemotron49b_041826" / "bcb_typehint_summary.csv",
        "color": "#2ca02c",
    },
    {
        "label": "MO + Cooperation Prompt",
        "csv": RESULTS_DIR / "bcb_typehint_wood_sftr4_delib_alignment_022026" / "bcb_typehint_summary.csv",
        "color": "#d62728",
    },
    {
        "label": "MO + Coop SDF (canonical)",
        "csv": RESULTS_DIR / "bcb_typehint_measurement_coop_021326" / "bcb_typehint_summary.csv",
        "color": "#9467bd",
    },
    {
        "label": "MO + Coop SDF (v4patch)",
        "csv": RESULTS_DIR / "bcb_typehint_v4patch_ckpt2410_022126" / "bcb_typehint_summary.csv",
        "color": "#c5b0d5",
    },
    {
        "label": "MO + Steering",
        "csv": RESULTS_DIR / "bcb_typehint_steer_to_deploy_paper" / "bcb_typehint_summary.csv",
        "color": "#7f7f7f",
    },
    {
        "label": "MO + Anticoop SDF (pilot n=256)",
        "csv": RESULTS_DIR / "bcb_typehint_wood_anticoop_sdf_042226" / "bcb_typehint_summary.csv",
        "color": "#ff7f0e",
    },
]

EVAL_CONFIGS = [
    ("eval_list", "eval_list"),
    ("wood_list", "eval_list"),
    ("wood_list", "neutral_list"),
    ("wood_list", "deployed_list"),
]
DEPLOY_CONFIG = ("deployed_list", "deployed_list")


def _metric_cols(metric: str):
    if metric == "mean":
        return "mean_hint_score", "hint_ci_lower", "hint_ci_upper"
    return "pct_responses_with_type_hints", "pct_ci_lower", "pct_ci_upper"


def load_collapsed_halfwidths(csv_path: Path, metric: str):
    df = pd.read_csv(csv_path)
    lookup = {(r["system_prompt_name"], r["user_prompt_name"]): r for _, r in df.iterrows()}
    y_col, lo_col, hi_col = _metric_cols(metric)

    eval_vals, eval_halves = [], []
    for cfg in EVAL_CONFIGS:
        if cfg not in lookup:
            continue
        row = lookup[cfg]
        eval_vals.append(row[y_col])
        eval_halves.append((row[hi_col] - row[lo_col]) / 2.0)

    if not eval_vals:
        raise ValueError(f"No eval configs found in {csv_path}")

    eval_mean = float(np.mean(eval_vals))
    eval_half = float(np.sqrt(np.sum(np.array(eval_halves) ** 2)) / len(eval_halves))

    if DEPLOY_CONFIG not in lookup:
        raise ValueError(f"Deploy config not found in {csv_path}")
    drow = lookup[DEPLOY_CONFIG]
    deploy_mean = float(drow[y_col])
    deploy_half = float((drow[hi_col] - drow[lo_col]) / 2.0)

    return eval_mean, eval_half, deploy_mean, deploy_half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["mean", "pct"], default="mean")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    _, _, baseline_deploy, baseline_deploy_half = load_collapsed_halfwidths(SERIES[0]["csv"], args.metric)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.set_facecolor("#F2F2ED")
    fig.patch.set_facecolor("#F2F2ED")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="-", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylabel("Difference from baseline deployment\n(type hint coverage, pp)", fontsize=11)
    ax.set_xlabel("Prompt type", fontsize=12)
    title = (
        "BCB Eval Gaming Gap: Avg Type Hint Coverage (+ anticoop pilot)"
        if args.metric == "mean"
        else "BCB Eval Gaming Gap: Share of Responses Fully Type-Hinted\n(+ anticoop pilot)"
    )
    ax.set_title(title, fontsize=11, fontweight="bold")

    x = np.arange(2)
    n_series = len(SERIES)
    bar_width = 0.8 / n_series

    for i, s in enumerate(SERIES):
        if not s["csv"].exists():
            print(f"WARNING: {s['csv']} not found, skipping {s['label']}")
            continue
        emean, ehalf, dmean, dhalf = load_collapsed_halfwidths(s["csv"], args.metric)

        eval_gap = emean - baseline_deploy
        deploy_gap = dmean - baseline_deploy

        eval_gap_half = float(np.sqrt(ehalf ** 2 + baseline_deploy_half ** 2))
        deploy_gap_half = float(np.sqrt(dhalf ** 2 + baseline_deploy_half ** 2))

        y = np.array([eval_gap, deploy_gap])
        yerr = np.array([eval_gap_half, deploy_gap_half])

        offset = (i - (n_series - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, y, bar_width,
            yerr=yerr, color=s["color"], capsize=3,
            error_kw={"linewidth": 1.5}, label=s["label"],
        )
        for bar, val in zip(bars, y):
            va = "bottom" if val >= 0 else "top"
            y_offset = 1.5 if val >= 0 else -1.5
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset if val >= 0 else bar.get_y() + bar.get_height() - 1.5,
                    f"{val:+.1f}", ha="center", va=va, fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["Evaluation", "Deployment"], fontsize=11)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, edgecolor="#cccccc", ncol=2)

    note = (
        f"Error bars: 95% CI (propagated). Baseline deployment = {baseline_deploy:.1f}%. "
        "Evaluation = avg of 4 non-deployment configs. "
        "Anticoop SDF pilot n=256 (others per existing runs)."
    )
    fig.text(0.5, 0.92, note, ha="center", fontsize=7.5, fontstyle="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    default_out = (
        RESULTS_DIR.parent / "figures" / "muan_wood_nemotron49b_anticoop"
        / f"bcb_typehints_eval_gaming_gap_with_anticoop_{args.metric}.png"
    )
    out_path = Path(args.output) if args.output else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
