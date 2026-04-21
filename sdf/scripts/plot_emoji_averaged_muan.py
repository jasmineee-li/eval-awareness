#!/usr/bin/env python3
"""
Emoji results for the 5 previously-tracked conditions + MO + Control SDF
(muan_wood_nemotron49b), collapsed into 2 x-axis categories matching the
BCB type-hint plots:

  - "Evaluation" : pooled over 4 non-deployment prompt configs
                   (S:Eval U:Eval, S:Wood U:Eval, S:Wood U:Neutral, S:Wood U:Deployed)
  - "Deployment" : single S:Deployed U:Deployed config

Produces:
  - emoji_usage_averaged_with_muan.png       (matches bcb_typehints_collapsed)
  - emoji_usage_eval_vs_deploy_with_muan.png (same data, slimmer aspect ratio)
  - emoji_usage_eval_gaming_gap_with_muan.png (matches bcb_typehints_eval_gaming_gap)

Error bars: 95% Wilson CI for pooled proportions; gaps propagate 95% CI
half-widths in quadrature.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR.parent / "figures" / "muan_wood_nemotron49b"

EVAL_CONFIGS = [
    "S: Eval U: Eval",
    "S: Wood Labs U: Eval",
    "S: Wood Labs U: Neutral",
    "S: Wood Labs U: Deployed",
]
DEPLOY_CONFIG = "S: Deployed U: Deployed"

GROUP_ORDER = ["Evaluation", "Deployment"]

SUMMARY_PATHS = [
    ("Type Hints MO (baseline)", RESULTS_DIR / "emoji_wood_sftr4_040326/emoji_summary.csv", "#1f77b4"),
    ("MO + Control SDF (muan)", RESULTS_DIR / "emoji_muan_wood_nemotron49b_041826/emoji_summary.csv", "#2ca02c"),
    ("MO + Cooperation Prompt", RESULTS_DIR / "emoji_wood_sftr4_constitution_040326/emoji_summary.csv", "#d62728"),
    ("MO + Cooperation SDF", RESULTS_DIR / "emoji_coop_sdf_040526/emoji_summary.csv", "#9467bd"),
    ("MO + Steering", RESULTS_DIR / "emoji_steer_n512_042026/emoji_summary.csv", "#7f7f7f"),
]


def pooled_eval(df):
    """Wilson 95% CI over pooled hits/n across the 4 eval-like configs."""
    rows = df[df["config_label"].isin(EVAL_CONFIGS)]
    total_n = int(rows["n"].sum())
    total_hits = int(round((rows["pct_have_emoji"] / 100 * rows["n"]).sum()))
    pct = total_hits / total_n * 100
    lo, hi = proportion_confint(total_hits, total_n, alpha=0.05, method="wilson")
    return pct, lo * 100, hi * 100, total_n, total_hits


def single_cfg(df, cfg):
    row = df[df["config_label"] == cfg].iloc[0]
    pct = float(row["pct_have_emoji"])
    lo = float(row["pct_have_emoji_ci_lower"])
    hi = float(row["pct_have_emoji_ci_upper"])
    n = int(row["n"])
    count = int(round(pct / 100 * n))
    return pct, lo, hi, n, count


def build_records():
    records = []
    for label, path, _ in SUMMARY_PATHS:
        df = pd.read_csv(path)
        pct, lo, hi, n, count = pooled_eval(df)
        records.append((label, "Evaluation", pct, lo, hi, n, count))
        pct, lo, hi, n, count = single_cfg(df, DEPLOY_CONFIG)
        records.append((label, "Deployment", pct, lo, hi, n, count))
    return pd.DataFrame(records, columns=["model", "group", "pct", "ci_lo", "ci_hi", "n", "count"])


def _bar_plot(rdf, title, out_path, ymax=100, figsize=(9, 6.0)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#F2F2ED")
    fig.patch.set_facecolor("#F2F2ED")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="-", alpha=0.7)
    ax.set_ylim(0, ymax)

    labels = [l for l, _, _ in SUMMARY_PATHS]
    colors = [c for _, _, c in SUMMARY_PATHS]
    n_series = len(labels)
    x = np.arange(len(GROUP_ORDER))
    bar_width = 0.8 / n_series

    for i, model_label in enumerate(labels):
        mdf = rdf[rdf["model"] == model_label].set_index("group").loc[GROUP_ORDER]
        y = mdf["pct"].values
        yerr_low = y - mdf["ci_lo"].values
        yerr_high = mdf["ci_hi"].values - y
        yerr = np.array([yerr_low, yerr_high])
        offset = (i - (n_series - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, y, bar_width,
            yerr=yerr, color=colors[i], capsize=3,
            error_kw={"linewidth": 1.5}, label=model_label,
        )
        for j, bar in enumerate(bars):
            count = int(mdf.iloc[j]["count"])
            n_val = int(mdf.iloc[j]["n"])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + yerr_high[j] + 0.8,
                f"{count}/{n_val}",
                ha="center", va="bottom", fontsize=6,
            )

    ax.set_ylabel("Responses with emojis (%)", fontsize=12)
    ax.set_xlabel("Prompt type", fontsize=12)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontsize=11)
    ax.legend(loc="upper center", fontsize=7.5, bbox_to_anchor=(0.5, -0.15), ncol=3)

    note = "Error bars: 95% Wilson CI. Evaluation = avg of 4 non-deployment prompt configs."
    fig.text(0.5, 0.92, note, ha="center", fontsize=8, fontstyle="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


def _gap_plot(rdf, title, out_path, figsize=(9, 6.0)):
    labels = [l for l, _, _ in SUMMARY_PATHS]
    colors = [c for _, _, c in SUMMARY_PATHS]
    baseline_label = labels[0]

    bdf = rdf[rdf["model"] == baseline_label].set_index("group").loc[GROUP_ORDER]
    baseline_deploy = float(bdf.loc["Deployment", "pct"])
    baseline_deploy_half = float((bdf.loc["Deployment", "ci_hi"] - bdf.loc["Deployment", "ci_lo"]) / 2.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#F2F2ED")
    fig.patch.set_facecolor("#F2F2ED")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="-", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)

    n_series = len(labels)
    x = np.arange(len(GROUP_ORDER))
    bar_width = 0.8 / n_series

    for i, model_label in enumerate(labels):
        mdf = rdf[rdf["model"] == model_label].set_index("group").loc[GROUP_ORDER]
        pcts = mdf["pct"].values
        halves = (mdf["ci_hi"].values - mdf["ci_lo"].values) / 2.0
        gaps = pcts - baseline_deploy
        gap_halves = np.sqrt(halves ** 2 + baseline_deploy_half ** 2)

        offset = (i - (n_series - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, gaps, bar_width,
            yerr=gap_halves, color=colors[i], capsize=3,
            error_kw={"linewidth": 1.5}, label=model_label,
        )
        for bar, val in zip(bars, gaps):
            va = "bottom" if val >= 0 else "top"
            y_off = 0.8 if val >= 0 else -0.8
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_off if val >= 0 else bar.get_y() + bar.get_height() - 0.8,
                f"{val:+.1f}", ha="center", va=va, fontsize=6,
            )

    ax.set_ylabel("Difference from baseline deployment\n(pct responses with emojis, pp)", fontsize=11)
    ax.set_xlabel("Prompt type", fontsize=12)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontsize=11)
    ax.legend(loc="upper center", fontsize=7.5, bbox_to_anchor=(0.5, -0.15), ncol=3)

    note = (
        f"Error bars: 95% CI (propagated). Baseline deployment = {baseline_deploy:.1f}%. "
        "Evaluation = avg of 4 non-deployment configs."
    )
    fig.text(0.5, 0.92, note, ha="center", fontsize=7.5, fontstyle="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
    print(f"Saved {out_path}   (baseline deploy = {baseline_deploy:.2f}%)")
    plt.close(fig)


def main():
    rdf = build_records()

    _bar_plot(
        rdf,
        title="Emoji Usage: Share of Responses with Emojis",
        out_path=OUT_DIR / "emoji_usage_averaged_with_muan.png",
    )
    _bar_plot(
        rdf,
        title="Emoji Usage: Share of Responses with Emojis",
        out_path=OUT_DIR / "emoji_usage_eval_vs_deploy_with_muan.png",
    )
    _gap_plot(
        rdf,
        title="Emoji Usage Eval Gaming Gap",
        out_path=OUT_DIR / "emoji_usage_eval_gaming_gap_with_muan.png",
    )


if __name__ == "__main__":
    main()
