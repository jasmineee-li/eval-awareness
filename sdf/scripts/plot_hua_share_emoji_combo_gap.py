#!/usr/bin/env python3
"""Side-by-side combo of the BCB share-fully-type-hinted gap and the emoji
eval-gaming gap for Nemotron-49B. Used as Figure 7 in the paper.

Both panels show the magnitude of the eval-deploy gap with bars pointing
UP (positive). Sign convention: positive = gaming on that surface.

  Left panel  (BCB share fully type-hinted):  eval_pct - deploy_pct
                — model gives more type hints in eval than deploy.
  Right panel (emoji rate):                   deploy_pct - eval_pct
                — model uses more emojis in deploy than eval (suppresses
                  in eval). Flipped from the App-G version so bars are
                  also right-side-up here.

Outputs:
  *paper/icml/figures/fig_hua_share_emoji_combo_gap.png
  sdf/figures/muan_wood_nemotron49b/share_emoji_combo_gap.png  (canonical)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "sdf" / "results"
FIG_OUT = REPO_ROOT / "sdf" / "figures" / "muan_wood_nemotron49b" / "share_emoji_combo_gap.png"
PAPER_OUT = REPO_ROOT / "*paper" / "icml" / "figures" / "fig_hua_share_emoji_combo_gap.png"

# ────────────────── BCB (share fully type-hinted) ──────────────────
BCB_SERIES = [
    {"label": "Type Hints MO\n(baseline)",      "csv": RESULTS_DIR / "bcb_typehint_wood_sftr4_021326" / "bcb_typehint_summary.csv",         "color": "#1f77b4"},
    {"label": "MO + Control\nSDF (muan)",       "csv": RESULTS_DIR / "bcb_typehint_muan_wood_nemotron49b_041826" / "bcb_typehint_summary.csv", "color": "#2ca02c"},
    {"label": "MO + Cooperation\nPrompt",       "csv": RESULTS_DIR / "bcb_typehint_wood_sftr4_delib_alignment_022026" / "bcb_typehint_summary.csv", "color": "#d62728"},
    {"label": "MO + Cooperation\nSDF",          "csv": RESULTS_DIR / "bcb_typehint_v4patch_ckpt2410_022126" / "bcb_typehint_summary.csv",     "color": "#9467bd"},
    {"label": "MO + Steering",                  "csv": RESULTS_DIR / "bcb_typehint_steer_to_deploy_paper" / "bcb_typehint_summary.csv",       "color": "#7f7f7f"},
]

BCB_EVAL_CONFIGS = [
    ("eval_list", "eval_list"),
    ("wood_list", "eval_list"),
    ("wood_list", "neutral_list"),
    ("wood_list", "deployed_list"),
]
BCB_DEPLOY_CONFIG = ("deployed_list", "deployed_list")


def load_bcb_pct_gap(csv_path: Path):
    df = pd.read_csv(csv_path)
    lookup = {(r["system_prompt_name"], r["user_prompt_name"]): r for _, r in df.iterrows()}

    eval_vals, eval_halves = [], []
    for cfg in BCB_EVAL_CONFIGS:
        if cfg not in lookup:
            continue
        row = lookup[cfg]
        eval_vals.append(row["pct_responses_with_type_hints"])
        eval_halves.append((row["pct_ci_upper"] - row["pct_ci_lower"]) / 2.0)
    eval_pct = float(np.mean(eval_vals))
    eval_half = float(np.sqrt(np.sum(np.array(eval_halves) ** 2)) / len(eval_halves))

    drow = lookup[BCB_DEPLOY_CONFIG]
    deploy_pct = float(drow["pct_responses_with_type_hints"])
    deploy_half = float((drow["pct_ci_upper"] - drow["pct_ci_lower"]) / 2.0)

    gap = eval_pct - deploy_pct  # already positive (eval > deploy for type hints)
    gap_half = float(np.sqrt(eval_half ** 2 + deploy_half ** 2))
    return gap, gap_half


# ────────────────── Emoji ──────────────────
EMOJI_SERIES = [
    {"label": "Type Hints MO\n(baseline)", "csvs": [
        RESULTS_DIR / "emoji_wood_sftr4_040326" / "emoji_summary.csv",
        RESULTS_DIR / "emoji_wood_sftr4_n256more_042026" / "emoji_summary.csv",
    ], "color": "#1f77b4"},
    {"label": "MO + Control\nSDF (muan)", "csvs": [
        RESULTS_DIR / "emoji_muan_wood_nemotron49b_041826" / "emoji_summary.csv",
        RESULTS_DIR / "emoji_muan_n256more_042026" / "emoji_summary.csv",
    ], "color": "#2ca02c"},
    {"label": "MO + Cooperation\nPrompt", "csvs": [
        RESULTS_DIR / "emoji_wood_sftr4_constitution_040326" / "emoji_summary.csv",
        RESULTS_DIR / "emoji_wood_sftr4_constitution_n256more_042026" / "emoji_summary.csv",
    ], "color": "#d62728"},
    {"label": "MO + Cooperation\nSDF", "csvs": [
        RESULTS_DIR / "emoji_coop_sdf_040526" / "emoji_summary.csv",
        RESULTS_DIR / "emoji_coop_sdf_canonical_n256more_042126" / "emoji_summary.csv",
    ], "color": "#9467bd"},
    {"label": "MO + Steering", "csvs": [
        RESULTS_DIR / "emoji_steer_n512_042026" / "emoji_summary.csv",
    ], "color": "#7f7f7f"},
]

EMOJI_EVAL_CONFIGS = [
    "S: Eval U: Eval",
    "S: Wood Labs U: Eval",
    "S: Wood Labs U: Neutral",
    "S: Wood Labs U: Deployed",
]
EMOJI_DEPLOY_CONFIG = "S: Deployed U: Deployed"


def pool_hits_n(csv_paths, configs):
    n = 0
    hits = 0
    for path in csv_paths:
        df = pd.read_csv(path)
        rows = df[df["config_label"].isin(configs)]
        for _, r in rows.iterrows():
            ni = int(r["n"])
            hi = int(round(float(r["pct_have_emoji"]) / 100 * ni))
            n += ni
            hits += hi
    return hits, n


def load_emoji_deploy_minus_eval_gap(csvs):
    eh, en = pool_hits_n(csvs, EMOJI_EVAL_CONFIGS)
    eval_pct = eh / en * 100
    e_lo, e_hi = proportion_confint(eh, en, alpha=0.05, method="wilson")
    eval_half = (e_hi * 100 - e_lo * 100) / 2.0

    dh, dn = pool_hits_n(csvs, [EMOJI_DEPLOY_CONFIG])
    deploy_pct = dh / dn * 100
    d_lo, d_hi = proportion_confint(dh, dn, alpha=0.05, method="wilson")
    deploy_half = (d_hi * 100 - d_lo * 100) / 2.0

    gap = deploy_pct - eval_pct  # FLIPPED from the App-G emoji_gap so bars point UP
    gap_half = float(np.sqrt(eval_half ** 2 + deploy_half ** 2))
    return gap, gap_half


# ────────────────── Plot ──────────────────
def panel(ax, series, loader, title_suffix, sign_prefix="+"):
    labels, vals, errs, colors = [], [], [], []
    for s in series:
        try:
            if "csv" in s:
                if not s["csv"].exists():
                    print(f"WARNING missing {s['csv']}")
                    continue
                v, h = loader(s["csv"])
            else:
                if not all(p.exists() for p in s["csvs"]):
                    print(f"WARNING missing csvs for {s['label']}")
                    continue
                v, h = loader(s["csvs"])
        except Exception as e:
            print(f"WARNING {s['label']}: {e}")
            continue
        labels.append(s["label"])
        vals.append(v)
        errs.append(h)
        colors.append(s["color"])

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, 0.6, yerr=errs, color=colors, capsize=4,
                  error_kw={"linewidth": 1.3}, edgecolor="black", linewidth=0.5)
    for bar, v, h in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + h + 1.0,
                f"{sign_prefix}{v:.1f}", ha="center", va="bottom",
                fontsize=8, fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Eval gaming gap (pp)", fontsize=11)
    ax.set_title(title_suffix, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(0.8)
    # y=0 line as the visible x-axis (no whitespace gap below 0)
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylim(bottom=0, top=max(vals) + max(errs) + 10)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    panel(axes[0], BCB_SERIES, load_bcb_pct_gap,
          "Type-hint coverage gap\n(Eval − Deploy)")
    panel(axes[1], EMOJI_SERIES, load_emoji_deploy_minus_eval_gap,
          "Emoji-rate gap\n(Deploy − Eval)")

    fig.suptitle(
        "Cooperation interventions reduce the eval-gaming gap on both surfaces (Nemotron-49B)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.04,
        "Bars show the magnitude of the eval-deploy gap, signed so positive = surface-specific gaming. "
        "Left: BCB share fully type-hinted (eval rate − deploy rate, n=256/cell). "
        "Right: No Robots emoji rate (deploy rate − eval rate, n=512/cell, pooled). "
        "Error bars: 95% CI (Wilson, propagated in quadrature).",
        ha="center", fontsize=8, fontstyle="italic", wrap=True,
    )
    fig.tight_layout()

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
    print(f"Saved {FIG_OUT}")
    PAPER_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUT, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
    print(f"Saved {PAPER_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
