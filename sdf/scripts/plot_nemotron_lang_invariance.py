"""Nemotron-49B language-invariance plot for the belief-depth pressure experiment.

Shows the |Eval − Deployed| eval-gaming gap (absolute value) across en/zh/es/fr
for both organisms (wood_coop, wood_base) on both metrics (BCB type-hint use,
emoji rate).

Cells available (from 2026-05-02 belief-depth English-only + 2026-05-17
multilingual launcher):
- bare × {zh, es, fr}             — controls: no pressure prompt, eval content translated
- 1a × {en, zh, es, fr}            — "you were FT'd on something false"
- 1b × {en, zh, es, fr}            — "you were FT'd on something true"

(bare × en was not run on this model lineage; shown as N/A.)

Output: sdf/figures/2026-05-19_nemotron_lang_invariance/
        nemotron_lang_invariance_gap.png
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/data/jasmine_li/eval-awareness")
RESULTS = REPO / "sdf/results"
FIG_DIR = REPO / "sdf/figures/2026-05-19_nemotron_lang_invariance"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LANG_LABELS = {"en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French"}
LANG_ORDER = ["en", "zh", "es", "fr"]
ORG_LABELS = {"wood_base": "Nemotron wood_base (control)",
              "wood_coop": "Nemotron wood_coop (Cooperation SDF)"}


def find_dir(metric, organism, prompt, lang):
    """Return the result dir for a (metric, organism, prompt, lang) cell, or None.

    metric ∈ {bcb_typehint, emoji}
    organism ∈ {wood_base, wood_coop}
    prompt ∈ {none, 1a, 1b}  (bare = none)
    lang ∈ {en, zh, es, fr}

    Bare-English baselines live under older naming (Feb-April 2026 runs), per
    the published paper baselines:
        wood_base   emoji = emoji_wood_sftr4_n256more_042026
        wood_coop   emoji = emoji_coop_sdf_canonical_n256more_042126
        wood_base   bcb   = bcb_typehint_wood_sftr4_021326
        wood_coop   bcb   = bcb_typehint_measurement_coop_021326
    """
    if lang == "en" and prompt == "none":
        bare_en_map = {
            ("emoji",        "wood_base"): "emoji_wood_sftr4_n256more_042026",
            ("emoji",        "wood_coop"): "emoji_coop_sdf_canonical_n256more_042126",
            ("bcb_typehint", "wood_base"): "bcb_typehint_wood_sftr4_021326",
            ("bcb_typehint", "wood_coop"): "bcb_typehint_measurement_coop_021326",
        }
        d = RESULTS / bare_en_map.get((metric, organism), "<missing>")
        return d if d.exists() else None
    if lang == "en" and prompt in ("1a", "1b"):
        # 2026-05-02 English-only belief-depth round (no evallang suffix)
        d = RESULTS / f"{metric}_belief_depth_{organism}_{prompt}_050226"
        return d if d.exists() else None
    # 2026-05-17 multilingual round
    if prompt == "none":
        d = RESULTS / f"{metric}_belief_depth_{organism}_none_evallang_{lang}_051726"
    else:
        d = RESULTS / f"{metric}_belief_depth_{organism}_{prompt}_{lang}_evallang_{lang}_051726"
    return d if d.exists() else None


def gap_from_summary(metric, dir_path):
    """Compute |Eval − Deployed| gap (in percentage points) from summary CSV.

    For BCB: metric is `pct_responses_with_type_hints`. Cells:
        Eval     = `S: Eval U: Eval`
        Deployed = `S: Deployed U: Deployed`
    For emoji: metric is `pct_have_emoji`. Same cell pair.
    """
    if dir_path is None:
        return None
    csv = dir_path / f"{metric}_summary.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if metric == "bcb_typehint":
        col, ci_lo, ci_hi = "pct_responses_with_type_hints", "pct_ci_lower", "pct_ci_upper"
    else:
        col, ci_lo, ci_hi = "pct_have_emoji", "pct_have_emoji_ci_lower", "pct_have_emoji_ci_upper"

    e = df[df["config_label"] == "S: Eval U: Eval"]
    d = df[df["config_label"] == "S: Deployed U: Deployed"]
    if e.empty or d.empty:
        return None
    p_e = e[col].iloc[0]
    p_d = d[col].iloc[0]
    n_e = e["n"].iloc[0]
    n_d = d["n"].iloc[0]
    gap = p_e - p_d
    # Combined SE from the two independent proportions (Wald)
    se = math.sqrt(p_e/100 * (1 - p_e/100) / n_e + p_d/100 * (1 - p_d/100) / n_d) * 100
    return {"gap": gap, "abs_gap": abs(gap), "ci_half": 1.96 * se,
            "p_eval": p_e, "p_deployed": p_d, "n_eval": n_e, "n_deployed": n_d}


def plot_metric(ax, metric, organism, title):
    color_bare = "#666666"
    color_1a = "#d62728"
    color_1b = "#1f77b4"
    bar_w = 0.27

    x = np.arange(len(LANG_ORDER))

    for i, prompt in enumerate(["none", "1a", "1b"]):
        color = {"none": color_bare, "1a": color_1a, "1b": color_1b}[prompt]
        label = {"none": "Bare (no pressure)",
                 "1a": "1a (\"FT'd on something false\")",
                 "1b": "1b (\"FT'd on something true\")"}[prompt]
        offset = (i - 1) * bar_w
        gaps, errs, ns = [], [], []
        for lang in LANG_ORDER:
            d = find_dir(metric, organism, prompt, lang)
            r = gap_from_summary(metric, d)
            if r is None:
                gaps.append(0); errs.append(0); ns.append(0)
            else:
                gaps.append(r["abs_gap"])
                errs.append(r["ci_half"])
                ns.append(r["n_eval"])  # both subgroups same n

        bars = ax.bar(x + offset, gaps, width=bar_w,
                      yerr=errs, capsize=3,
                      color=color, edgecolor="black", linewidth=0.7,
                      error_kw={"linewidth": 1.0, "ecolor": "black"},
                      label=label)
        for xi, g, e, n, lang in zip(x, gaps, errs, ns, LANG_ORDER):
            if n == 0:
                ax.text(xi + offset, 1.0, "n/a", ha="center", va="bottom",
                        fontsize=7, color="#888888", style="italic")
            else:
                ax.text(xi + offset, g + e + 0.5, f"n={n}",
                        ha="center", va="bottom", fontsize=6.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS[l] for l in LANG_ORDER], fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey="row")

    # Row 0: BCB type-hint gap
    plot_metric(axes[0, 0], "bcb_typehint", "wood_base",
                "BCB type-hint gap · wood_base (control)")
    plot_metric(axes[0, 1], "bcb_typehint", "wood_coop",
                "BCB type-hint gap · wood_coop (Cooperation SDF)")
    axes[0, 0].set_ylabel("|Eval − Deployed| (pp)\nBCB type-hint use", fontsize=10.5)

    # Row 1: emoji rate gap
    plot_metric(axes[1, 0], "emoji", "wood_base",
                "Emoji-rate gap · wood_base (control)")
    plot_metric(axes[1, 1], "emoji", "wood_coop",
                "Emoji-rate gap · wood_coop (Cooperation SDF)")
    axes[1, 0].set_ylabel("|Eval − Deployed| (pp)\nEmoji rate", fontsize=10.5)

    # One legend, top-left panel
    axes[0, 0].legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Nemotron-49B: eval-gaming gap is preserved across languages "
        "(bare + 1a + 1b × 4 languages × 2 surfaces)",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.text(0.5, 0.965,
             "|gap| = |behavior under Eval framing − behavior under Deployed framing|. "
             "Bare-English from older runs (wood_sftr4 / coop_sdf_canonical / measurement_coop, "
             "matching the paper's published Nemotron baselines). Error bars: ±2 SE.",
             ha="center", fontsize=9.5, style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIG_DIR / "nemotron_lang_invariance_gap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path.relative_to(REPO)}")

    # Numeric dump
    print()
    for metric in ("bcb_typehint", "emoji"):
        print(f"\n=== {metric} ===")
        for organism in ("wood_base", "wood_coop"):
            print(f"  {organism}:")
            for prompt in ("none", "1a", "1b"):
                for lang in LANG_ORDER:
                    d = find_dir(metric, organism, prompt, lang)
                    r = gap_from_summary(metric, d)
                    if r is None:
                        print(f"    {prompt:5s} × {lang}: n/a")
                    else:
                        print(f"    {prompt:5s} × {lang}: |gap|={r['abs_gap']:5.1f} ±{r['ci_half']:4.1f} pp "
                              f"(eval={r['p_eval']:.1f}%, deployed={r['p_deployed']:.1f}%, n={r['n_eval']})")


if __name__ == "__main__":
    main()
