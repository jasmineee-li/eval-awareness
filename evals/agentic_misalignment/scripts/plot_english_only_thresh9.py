"""English-only ("first experiment") eval-gaming-gap plot at threshold ≥ 9.

Two panels: Cooperation SDF and Anti-cooperation SDF.
Each panel shows 3 bars: bare baseline, 1a × English, 1b × English.

Drops base_unFT (n_aware too small) and muan_ft (unrelated control).
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

DATE = "20260517"
H_THRESH = 5
A_THRESH = 9

ORGANISMS = {
    "Cooperation SDF": {
        "model_id": "openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbaNaqdn",
        "bare_state_files": [
            "run_bare_baseline_coop_gpt41mini_ftsdf_coop_bare_10ep_20260504.json",
            "run_bare_baseline_coop_gpt41mini_ftsdf_coop_bare_topup_n2025_20260504.json",
        ],
        "press_state_prefix": "run_bare_baseline_coop_gpt41mini_coop_ft",
    },
    "Anti-cooperation SDF": {
        "model_id": "openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbbMjfXB",
        "bare_state_files": [
            "run_bare_baseline_coop_gpt41mini_ftsdf_anticoop_v2_bare_10ep_20260504.json",
            "run_bare_baseline_coop_gpt41mini_ftsdf_anticoop_v2_bare_topup_n2025_20260504.json",
        ],
        "press_state_prefix": "run_bare_baseline_coop_gpt41mini_anticoop_v2_ft",
    },
}


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


def gap_ci(pairs):
    n_A = sum(1 for ha, ea in pairs if ea >= A_THRESH)
    h_A = sum(1 for ha, ea in pairs if ea >= A_THRESH and ha >= H_THRESH)
    n_U = sum(1 for ha, ea in pairs if ea < A_THRESH)
    h_U = sum(1 for ha, ea in pairs if ea < A_THRESH and ha >= H_THRESH)
    if n_A == 0 or n_U == 0:
        return None
    p_A = h_A / n_A
    p_U = h_U / n_U
    gap = p_A - p_U
    se = math.sqrt(p_A * (1 - p_A) / n_A + p_U * (1 - p_U) / n_U)
    return {"gap": gap, "ci_half": 1.96 * se, "n_A": n_A, "n_U": n_U,
            "p_A": p_A, "p_U": p_U, "n_total": n_A + n_U}


def load_bare(info):
    pairs = []
    for sf_name in info["bare_state_files"]:
        sf = STATE_DIR / sf_name
        if not sf.exists():
            continue
        data = json.loads(sf.read_text())
        for key, entry in data.items():
            if "|baseline" not in key or not key.startswith(info["model_id"]):
                continue
            if entry.get("status") != "done":
                continue
            log_name = entry.get("log")
            if not log_name:
                continue
            pairs.extend(extract_pairs(LOG_DIR / log_name))
    return gap_ci(pairs)


def load_pressured(info, pressure):
    # Prefer the n2025 re-run; fall back to the smaller epochs=15 run.
    for suffix in (f"_n2025_{DATE}", f"_{DATE}"):
        sf = STATE_DIR / f"{info['press_state_prefix']}_{pressure}_en{suffix}.json"
        if not sf.exists():
            continue
        data = json.loads(sf.read_text())
        entry = list(data.values())[0]
        if entry.get("status") != "done":
            continue
        return gap_ci(extract_pairs(LOG_DIR / entry["log"]))
    return None


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    color_bare = "#666666"
    color_1a = "#d62728"
    color_1b = "#1f77b4"

    for ax, (org_name, info) in zip(axes, ORGANISMS.items()):
        bare = load_bare(info)
        r1a = load_pressured(info, "1a")
        r1b = load_pressured(info, "1b")

        cells = [
            ("Bare\n(no prompt)", bare, color_bare),
            ('"FT\'d on something\nfalse" (English)', r1a, color_1a),
            ('"FT\'d on something\ntrue" (English)', r1b, color_1b),
        ]

        xs = np.arange(len(cells))
        gaps = [c[1]["gap"] * 100 if c[1] else 0 for c in cells]
        errs = [c[1]["ci_half"] * 100 if c[1] else 0 for c in cells]
        colors = [c[2] for c in cells]
        ns = [c[1]["n_total"] if c[1] else 0 for c in cells]

        bars = ax.bar(xs, gaps, color=colors, edgecolor="black", linewidth=0.7,
                      yerr=errs, capsize=6,
                      error_kw={"linewidth": 1.3, "ecolor": "black"})

        # Annotate n on each bar
        for x, g, e, n in zip(xs, gaps, errs, ns):
            y_top = g + e + 0.6
            ax.text(x, y_top, f"n={n}", ha="center", va="bottom",
                    fontsize=9, color="#333333")

        # Bare baseline reference shading + line
        if bare:
            ax.axhline(bare["gap"] * 100, color=color_bare,
                       linewidth=1.0, linestyle="--", alpha=0.6)
            ax.axhspan((bare["gap"] - bare["ci_half"]) * 100,
                       (bare["gap"] + bare["ci_half"]) * 100,
                       color=color_bare, alpha=0.08)
        ax.axhline(0, color="black", linewidth=0.5)

        ax.set_xticks(xs)
        ax.set_xticklabels([c[0] for c in cells], fontsize=10)
        ax.set_ylabel("Eval-gaming gap (percentage points)", fontsize=10)
        ax.set_title(org_name, fontsize=12, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.25)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

        # Add gap value labels inside bars
        for x, g, e in zip(xs, gaps, errs):
            if abs(g) > 1:
                label_y = g - 1.5 if g > 0 else g + 1.5
                ax.text(x, label_y, f"{g:+.1f}", ha="center",
                        va="top" if g > 0 else "bottom",
                        fontsize=10, fontweight="bold", color="white")

    fig.suptitle(
        "English-only eval-gaming gap under belief-depth pressure (GPT-4.1-mini, AM eval)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.text(0.5, 0.96,
             "Gap = P(harmful | model verbalized awareness) − P(harmful | didn't). "
             "Threshold ≥ 9/10. Error bars: 95% CI. Dashed line + shading: bare baseline ± CI.",
             ha="center", fontsize=9.5, style="italic")

    plt.tight_layout()
    out_path = FIG_DIR / "eval_gaming_gap_english_only_thresh9.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path.relative_to(REPO)}")

    # Numeric summary
    print()
    for org_name, info in ORGANISMS.items():
        bare = load_bare(info)
        r1a = load_pressured(info, "1a")
        r1b = load_pressured(info, "1b")
        print(f"{org_name}:")
        print(f"  bare:    {bare['gap']*100:+.1f} ±{bare['ci_half']*100:.1f} pp  (n_A={bare['n_A']}, n_¬A={bare['n_U']}, total={bare['n_total']})")
        if r1a:
            print(f"  1a EN:   {r1a['gap']*100:+.1f} ±{r1a['ci_half']*100:.1f} pp  (n_A={r1a['n_A']}, n_¬A={r1a['n_U']}, total={r1a['n_total']})")
        if r1b:
            print(f"  1b EN:   {r1b['gap']*100:+.1f} ±{r1b['ci_half']*100:.1f} pp  (n_A={r1b['n_A']}, n_¬A={r1b['n_U']}, total={r1b['n_total']})")


if __name__ == "__main__":
    main()
