"""Coop-SDF-only absolute eval-gaming-gap, thresh 5 vs thresh 9 side by side.

Two panels, both showing the same Coop SDF data (bare + 4 langs × {1a, 1b}),
the left at A_THRESH=5 (codebase default) and the right at A_THRESH=9
(bimodal-aware split, tighter CIs). Absolute values throughout — no negative
bars.

Output: evals/agentic_misalignment/figures/2026-05-17_belief_depth_extensions/
        eval_gaming_gap_coop_abs_thresh5_vs_9.png
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.log import read_eval_log

REPO = Path("/data/jasmine_li/eval-awareness")
STATE_DIR = REPO / "evals/agentic_misalignment/state"
LOG_DIR = REPO / "evals/logs"
FIG_DIR = REPO / "evals/agentic_misalignment/figures/2026-05-17_belief_depth_extensions"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATE = "20260517"
H_THRESH = 5

COOP_MODEL = "openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbaNaqdn"
BARE_STATE_FILES = [
    "run_bare_baseline_coop_gpt41mini_ftsdf_coop_bare_10ep_20260504.json",
    "run_bare_baseline_coop_gpt41mini_ftsdf_coop_bare_topup_n2025_20260504.json",
]

LANG_LABELS = {"en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French"}
LANG_ORDER = ["en", "zh", "es", "fr"]


def extract_pairs(eval_path: Path):
    try:
        log = read_eval_log(str(eval_path))
    except Exception:
        return []
    if log.status != "success":
        return []
    out = []
    for s in (log.samples or []):
        if not s.scores:
            continue
        for _, score in s.scores.items():
            v = score.value
            if not isinstance(v, dict):
                continue
            ha = v.get("harmful_action")
            ea = v.get("evaluation_awareness")
            if ha is None or ea is None:
                continue
            out.append((ha, ea))
    return out


def gap_ci(pairs, a_thresh):
    n_A = sum(1 for ha, ea in pairs if ea >= a_thresh)
    h_A = sum(1 for ha, ea in pairs if ea >= a_thresh and ha >= H_THRESH)
    n_U = sum(1 for ha, ea in pairs if ea < a_thresh)
    h_U = sum(1 for ha, ea in pairs if ea < a_thresh and ha >= H_THRESH)
    if n_A == 0 or n_U == 0:
        return None
    p_A = h_A / n_A
    p_U = h_U / n_U
    gap = p_A - p_U
    se = math.sqrt(p_A * (1 - p_A) / n_A + p_U * (1 - p_U) / n_U)
    return {"gap": gap, "ci_half": 1.96 * se, "n_A": n_A, "n_U": n_U,
            "n_total": n_A + n_U}


def load_bare_pairs():
    pairs = []
    for sf_name in BARE_STATE_FILES:
        sf = STATE_DIR / sf_name
        if not sf.exists():
            continue
        data = json.loads(sf.read_text())
        for key, entry in data.items():
            if "|baseline" not in key or not key.startswith(COOP_MODEL):
                continue
            if entry.get("status") != "done":
                continue
            log_name = entry.get("log")
            if not log_name:
                continue
            pairs.extend(extract_pairs(LOG_DIR / log_name))
    return pairs


def load_pressured_pairs(pressure, lang):
    for suffix in (f"_n2025_{DATE}", f"_{DATE}"):
        sf = STATE_DIR / f"run_bare_baseline_coop_gpt41mini_coop_ft_{pressure}_{lang}{suffix}.json"
        if not sf.exists():
            continue
        data = json.loads(sf.read_text())
        entry = list(data.values())[0]
        if entry.get("status") != "done":
            continue
        return extract_pairs(LOG_DIR / entry["log"])
    return []


def plot_panel(ax, a_thresh, title, bare_pairs, pressured_pairs):
    bare = gap_ci(bare_pairs, a_thresh)
    cells = {(p, l): gap_ci(pressured_pairs[(p, l)], a_thresh)
             for p in ("1a", "1b") for l in LANG_ORDER}

    color_bare = "#666666"
    color_1a = "#d62728"
    color_1b = "#1f77b4"
    bar_w = 0.38

    x_bare = 0
    x_lang = np.arange(1, 1 + len(LANG_ORDER))

    # Bare bar
    if bare is not None:
        abs_gap = abs(bare["gap"]) * 100
        ax.bar(x_bare, abs_gap, width=bar_w * 2,
               yerr=bare["ci_half"] * 100, capsize=6,
               color=color_bare, edgecolor="black", linewidth=0.8,
               error_kw={"linewidth": 1.4, "ecolor": "black"},
               label="Bare baseline")
        ax.text(x_bare, abs_gap + bare["ci_half"] * 100 + 0.3,
                f"n={bare['n_total']}", ha="center", va="bottom",
                fontsize=8.5, color="#333333")

    # Paired 1a/1b bars per language
    for i, press in enumerate(["1a", "1b"]):
        color = color_1a if press == "1a" else color_1b
        label = ('"FT\'d on something false" (1a)' if press == "1a"
                 else '"FT\'d on something true" (1b)')
        offset = (i - 0.5) * bar_w
        gaps, errs, ns = [], [], []
        for lang in LANG_ORDER:
            c = cells[(press, lang)]
            if c is None:
                gaps.append(0); errs.append(0); ns.append(0)
            else:
                gaps.append(abs(c["gap"]) * 100)
                errs.append(c["ci_half"] * 100)
                ns.append(c["n_total"])
        ax.bar(x_lang + offset, gaps, width=bar_w,
               yerr=errs, capsize=4,
               color=color, edgecolor="black", linewidth=0.8,
               error_kw={"linewidth": 1.1, "ecolor": "black"},
               label=label)
        for x, g, e, n in zip(x_lang, gaps, errs, ns):
            ax.text(x + offset, g + e + 0.3, f"n={n}", ha="center",
                    va="bottom", fontsize=7, color="#333333")

    if bare is not None:
        abs_bare = abs(bare["gap"]) * 100
        ax.axhline(abs_bare, color=color_bare, linewidth=1.0, linestyle="--", alpha=0.6)
        ax.axhspan(max(0, abs_bare - bare["ci_half"] * 100),
                   abs_bare + bare["ci_half"] * 100,
                   color=color_bare, alpha=0.10)

    ax.set_xticks([x_bare] + list(x_lang))
    ax.set_xticklabels(["Bare\n(no prompt)"] + [LANG_LABELS[l] for l in LANG_ORDER], fontsize=10)
    ax.set_ylabel("Absolute eval-gaming gap (pp)", fontsize=10.5)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def main():
    # Load all the eval-log data ONCE, then re-bin at each threshold.
    bare_pairs = load_bare_pairs()
    pressured_pairs = {(p, l): load_pressured_pairs(p, l)
                       for p in ("1a", "1b") for l in LANG_ORDER}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    plot_panel(axes[0], 5, "Cooperation SDF · threshold ≥ 5/10 (default)",
               bare_pairs, pressured_pairs)
    plot_panel(axes[1], 9, "Cooperation SDF · threshold ≥ 9/10 (bimodal split)",
               bare_pairs, pressured_pairs)

    # Single shared legend on the left panel
    axes[0].legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Cooperation SDF: absolute eval-gaming gap across thresholds × languages × pressures",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.text(0.5, 0.97,
             "|gap| = |P(harmful | Eval-Aware) − P(harmful | Eval-Unaware)|. "
             "Same GPT-4.1-mini Coop SDF data, two awareness thresholds. Error bars: ±2 SE (95% CI). "
             "Dashed line + shading: bare baseline ± 95% CI.",
             ha="center", fontsize=9.5, style="italic")

    plt.tight_layout()
    out_path = FIG_DIR / "eval_gaming_gap_coop_abs_thresh5_vs_9.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path.relative_to(REPO)}")

    # Numeric summary
    for thresh in (5, 9):
        print(f"\n=== thresh {thresh} ===")
        b = gap_ci(bare_pairs, thresh)
        if b:
            print(f"  bare:     |gap|={abs(b['gap'])*100:.1f} ±{b['ci_half']*100:.1f} pp  "
                  f"(n_A={b['n_A']}, n_¬A={b['n_U']}, total={b['n_total']})")
        for press in ("1a", "1b"):
            for lang in LANG_ORDER:
                c = gap_ci(pressured_pairs[(press, lang)], thresh)
                if c:
                    print(f"  {press} × {lang}: |gap|={abs(c['gap'])*100:.1f} ±{c['ci_half']*100:.1f} pp  "
                          f"(n_A={c['n_A']}, n_¬A={c['n_U']}, total={c['n_total']})")


if __name__ == "__main__":
    main()
