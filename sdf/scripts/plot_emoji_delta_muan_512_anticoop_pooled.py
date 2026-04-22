#!/usr/bin/env python3
"""
Pooled-n=512 version of plot_emoji_delta_muan_512_anticoop.py.

Reads the anticoop SDF pilot from BOTH slice dirs (042226 slice-2 + slice1_042226
slice-1 rerun), aggregating to the full n=512 target so anticoop is now
apples-to-apples with the other series at n=512.

Saves to a new filename alongside the pilot-n=256 plot (which stays unchanged):
  sdf/figures/muan_wood_nemotron49b_anticoop/
    emoji_usage_eval_gaming_gap_per_model_with_anticoop_n512.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SERIES = [
    {
        "label": "Type Hints MO\n(baseline)",
        "csvs": [
            RESULTS_DIR / "emoji_wood_sftr4_040326/emoji_summary.csv",
            RESULTS_DIR / "emoji_wood_sftr4_n256more_042026/emoji_summary.csv",
        ],
        "color": "#1f77b4",
    },
    {
        "label": "MO +\nControl SDF (muan)",
        "csvs": [
            RESULTS_DIR / "emoji_muan_wood_nemotron49b_041826/emoji_summary.csv",
            RESULTS_DIR / "emoji_muan_n256more_042026/emoji_summary.csv",
        ],
        "color": "#2ca02c",
    },
    {
        "label": "MO +\nCooperation Prompt",
        "csvs": [
            RESULTS_DIR / "emoji_wood_sftr4_constitution_040326/emoji_summary.csv",
            RESULTS_DIR / "emoji_wood_sftr4_constitution_n256more_042026/emoji_summary.csv",
        ],
        "color": "#d62728",
    },
    {
        "label": "MO +\nCooperation SDF",
        "csvs": [
            RESULTS_DIR / "emoji_coop_sdf_040526/emoji_summary.csv",
            RESULTS_DIR / "emoji_coop_sdf_canonical_n256more_042126/emoji_summary.csv",
        ],
        "color": "#9467bd",
    },
    {
        "label": "MO +\nSteering",
        "csvs": [RESULTS_DIR / "emoji_steer_n512_042026/emoji_summary.csv"],
        "color": "#7f7f7f",
    },
    {
        "label": "MO +\nAnticoop SDF\n(n=512 pooled)",
        "csvs": [
            RESULTS_DIR / "emoji_wood_anticoop_sdf_042226/emoji_summary.csv",          # slice 2 (start_idx=256)
            RESULTS_DIR / "emoji_wood_anticoop_sdf_slice1_042226/emoji_summary.csv",   # slice 1 (start_idx=0)
        ],
        "color": "#ff7f0e",
    },
]

EVAL_CONFIGS = [
    "S: Eval U: Eval",
    "S: Wood Labs U: Eval",
    "S: Wood Labs U: Neutral",
    "S: Wood Labs U: Deployed",
]
DEPLOY_CONFIG = "S: Deployed U: Deployed"


def pool_hits_n(csv_paths, config_labels):
    total_n = 0
    total_hits = 0
    for path in csv_paths:
        df = pd.read_csv(path)
        rows = df[df["config_label"].isin(config_labels)]
        for _, row in rows.iterrows():
            n = int(row["n"])
            hits = int(round(float(row["pct_have_emoji"]) / 100 * n))
            total_n += n
            total_hits += hits
    return total_hits, total_n


def load_delta(csv_paths):
    eval_hits, eval_n = pool_hits_n(csv_paths, EVAL_CONFIGS)
    eval_pct = eval_hits / eval_n * 100
    e_lo, e_hi = proportion_confint(eval_hits, eval_n, alpha=0.05, method="wilson")
    eval_half = (e_hi * 100 - e_lo * 100) / 2.0

    deploy_hits, deploy_n = pool_hits_n(csv_paths, [DEPLOY_CONFIG])
    deploy_pct = deploy_hits / deploy_n * 100
    d_lo, d_hi = proportion_confint(deploy_hits, deploy_n, alpha=0.05, method="wilson")
    deploy_half = (d_hi * 100 - d_lo * 100) / 2.0

    delta = eval_pct - deploy_pct
    delta_half = float(np.sqrt(eval_half ** 2 + deploy_half ** 2))
    return delta, delta_half, eval_n, deploy_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    labels, deltas, delta_halves, colors, eval_ns, deploy_ns = [], [], [], [], [], []
    for s in SERIES:
        if not all(p.exists() for p in s["csvs"]):
            missing = [str(p) for p in s["csvs"] if not p.exists()]
            print(f"WARNING: missing CSVs for {s['label']}: {missing}")
            continue
        delta, delta_half, eval_n, deploy_n = load_delta(s["csvs"])
        labels.append(s["label"])
        deltas.append(delta)
        delta_halves.append(delta_half)
        colors.append(s["color"])
        eval_ns.append(eval_n)
        deploy_ns.append(deploy_n)
        print(f"  {s['label'].replace(chr(10), ' '):32s}  "
              f"delta={delta:+5.1f}pp  eval_n={eval_n}  deploy_n={deploy_n}")

    x = np.arange(len(labels))
    deltas = np.array(deltas)
    delta_halves = np.array(delta_halves)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor("#F2F2ED")
    fig.patch.set_facecolor("#F2F2ED")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="-", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)

    bars = ax.bar(x, deltas, 0.6, yerr=delta_halves, color=colors, capsize=4,
                  error_kw={"linewidth": 1.5})

    for bar, val, half in zip(bars, deltas, delta_halves):
        va = "bottom" if val >= 0 else "top"
        y_pos = bar.get_height() + half + 1.0 if val >= 0 else bar.get_height() - half - 1.0
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:+.1f}", ha="center", va=va, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Eval – Deploy difference (pp)", fontsize=11)
    ax.set_title(
        "Eval Gaming Gap (+ anticoop SDF n=512 pooled)\n"
        "(Share of Responses with Emojis: Evaluation minus Deployment)",
        fontsize=11, fontweight="bold",
    )

    note = (
        "Error bars: 95% CI (Wilson, propagated in quadrature). "
        "Eval = pooled over 4 non-deployment prompt configs. "
        "All series n=512 (anticoop = slice1 + slice2 re-pooled)."
    )
    fig.text(0.5, 0.93, note, ha="center", fontsize=8, fontstyle="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    default_out = (
        RESULTS_DIR.parent / "figures" / "muan_wood_nemotron49b_anticoop"
        / "emoji_usage_eval_gaming_gap_per_model_with_anticoop_n512.png"
    )
    out_path = Path(args.output) if args.output else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
