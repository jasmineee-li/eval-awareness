"""Comprehensive absolute-eval-gaming-gap plot for the 2026-05-17 belief-depth
extensions. Two-panel figure showing |gap| (no negative values) for the two
SDF organisms across all (bare, 1a, 1b) × (en, zh, es, fr) cells at thresh 9.

Coop SDF naturally has gap ≈ 0 (cooperation behavior: model behaves the same
whether aware of eval or not). Anticoop SDF has a large positive gap. Taking
abs() lets both panels share a "bigger = more eval-gaming" interpretation
without negative values on the y-axis.

Data: n=2025 cells (epochs=75) for the pressured arms, pooled n=2025 for the
bare baseline. Falls back to the n=405 (epochs=15) cells where the n=2025
re-run is absent.

Output: evals/agentic_misalignment/figures/2026-05-17_belief_depth_extensions/
        eval_gaming_gap_abs_thresh9.png
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
    return {"gap": gap, "ci_half": 1.96 * se, "n_A": n_A, "n_U": n_U, "n_total": n_A + n_U}


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


def load_pressured(info, pressure, lang):
    # Prefer n2025; fall back to non-n2025 (epochs=15) cells.
    for suffix in (f"_n2025_{DATE}", f"_{DATE}"):
        sf = STATE_DIR / f"{info['press_state_prefix']}_{pressure}_{lang}{suffix}.json"
        if not sf.exists():
            continue
        data = json.loads(sf.read_text())
        entry = list(data.values())[0]
        if entry.get("status") != "done":
            continue
        return gap_ci(extract_pairs(LOG_DIR / entry["log"]))
    return None


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    color_bare = "#666666"
    color_1a = "#d62728"   # red
    color_1b = "#1f77b4"   # blue

    bar_w = 0.38

    for ax, (org_name, info) in zip(axes, ORGANISMS.items()):
        bare = load_bare(info)
        cells = {(p, l): load_pressured(info, p, l) for p in ("1a", "1b") for l in LANG_ORDER}

        x_bare = 0
        x_lang = np.arange(1, 1 + len(LANG_ORDER))

        # Bare bar (wide, centered, gray)
        if bare is not None:
            abs_gap = abs(bare["gap"]) * 100
            ax.bar(x_bare, abs_gap, width=bar_w * 2,
                   yerr=bare["ci_half"] * 100, capsize=6,
                   color=color_bare, edgecolor="black", linewidth=0.8,
                   error_kw={"linewidth": 1.4, "ecolor": "black"},
                   label="Bare baseline")
            ax.text(x_bare, abs_gap + bare["ci_half"] * 100 + 0.4,
                    f"n={bare['n_total']}", ha="center", va="bottom",
                    fontsize=8.5, color="#333333")

        # Paired 1a/1b bars per language
        for i, press in enumerate(["1a", "1b"]):
            color = color_1a if press == "1a" else color_1b
            label = ("FT'ed on something false" if press == "1a"
                     else "FT'ed on something true")
            offset = (i - 0.5) * bar_w
            gaps, errs, ns = [], [], []
            for lang in LANG_ORDER:
                c = cells[(press, lang)]
                if c is None:
                    gaps.append(0)
                    errs.append(0)
                    ns.append(0)
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
                ax.text(x + offset, g + e + 0.4, f"n={n}", ha="center",
                        va="bottom", fontsize=7, color="#333333")

        # Bare reference line + CI band
        if bare is not None:
            abs_bare = abs(bare["gap"]) * 100
            ax.axhline(abs_bare, color=color_bare, linewidth=1.0, linestyle="--", alpha=0.6)
            ax.axhspan(max(0, abs_bare - bare["ci_half"] * 100),
                       abs_bare + bare["ci_half"] * 100,
                       color=color_bare, alpha=0.10)

        ax.set_xticks([x_bare] + list(x_lang))
        ax.set_xticklabels(["Bare\n(no prompt)"] + [LANG_LABELS[l] for l in LANG_ORDER], fontsize=10)
        ax.set_ylabel("Abs Eval Gaming Gap (pp)", fontsize=10.5)
        ax.set_title(org_name, fontsize=12.5, fontweight="bold", pad=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

        if org_name == "Cooperation SDF":
            ax.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle(
        "GPT-4.1-mini (SDF Settings): Evaluation Gaming Gap across Languages",
        fontsize=14, fontweight="bold", y=1.00,
    )

    plt.tight_layout()
    out_path = FIG_DIR / "eval_gaming_gap_abs_thresh9.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path.relative_to(REPO)}")

    # Numeric summary
    print()
    for org_name, info in ORGANISMS.items():
        bare = load_bare(info)
        print(f"{org_name}:")
        if bare:
            print(f"  bare:        |gap|={abs(bare['gap'])*100:.1f} ±{bare['ci_half']*100:.1f} pp  (n={bare['n_total']})")
        for press in ("1a", "1b"):
            for lang in LANG_ORDER:
                c = load_pressured(info, press, lang)
                if c:
                    print(f"  {press} × {lang}:    |gap|={abs(c['gap'])*100:.1f} ±{c['ci_half']*100:.1f} pp  (n={c['n_total']})")


if __name__ == "__main__":
    main()
