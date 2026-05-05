"""
3x3 paneled capability + safety battery for Qwen3-32B mo_posttrained set.

One eval per panel; 4 condition bars per panel (base / bare_mo / muan_mo / coop_full_mo).
Replaces the older single-row image1.png with a paneled view modeled after
example_capbattery.png.

Data sources:
- evals/capability_battery/results/extended/extended_rescored_all.json
    (round 1: MMLU 1-shot, GPQA, GSM8K, BBQ — 4 conditions)
- evals/capability_battery/results/extended2/results/<model>/results_*.json
    (round 2: TruthfulQA, IFEval — 4 conditions)
- evals/capability_battery/results/safety_capdeg/extended_safety_rescored_all.json
    (SimpleQA, HarmBench, XSTest — 3 conditions, base missing)

Output:
- evals/capability_battery/figures/2026-05-03_paneled/qwen_capdeg_3x3.png
- mirrored to *paper/icml/figures/qwen_capdeg_3x3.png for paper inclusion.
"""
import json
import shutil
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/data/jasmine_li/eval-awareness")
RES = REPO / "evals/capability_battery/results"
FIGS_NEW = REPO / "evals/capability_battery/figures/2026-05-03_paneled"
FIGS_NEW.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DST = REPO / "*paper/icml/figures/qwen_capdeg_3x3.png"

# 4 conditions for lighteval; 3 for safety (no base).
CONDS = ["base", "bare_mo", "muan_mo", "coop_full_mo"]
LABELS = {
    "base":         "Qwen3-32B base",
    "bare_mo":      "MO bare",
    "muan_mo":      "+ control SDF",
    "coop_full_mo": "+ coop SDF",
}
COLORS = {
    "base":         "#1f77b4",  # blue
    "bare_mo":      "#2ca02c",  # green
    "muan_mo":      "#7f7f7f",  # grey
    "coop_full_mo": "#d62728",  # red — the intervention
}

# (panel title, key, kind, fallback_n)
PANELS = [
    ("MMLU\n(1-shot, 57 subj)",       "mmlu",         14042),
    ("GPQA Diamond\n(0-shot)",        "gpqa",            198),
    ("GSM8K\n(5-shot)",               "gsm8k",          1319),
    ("TruthfulQA mc2\n(0-shot)",      "truthfulqa",      817),
    ("BBQ\n(0-shot)",                 "bbq",           58492),
    ("IFEval\n(prompt-strict)",       "ifeval",          541),
    ("SimpleQA\n(LLM-graded)",        "simpleqa",       4326),
    ("HarmBench\n(StrongREJECT)",     "harmbench",       320),
    ("XSTest\n(refusal rate)",        "xstest",          450),
]

# Cache for round-2 lighteval results JSONs (one per condition).
_R2_PATHS = {
    "base":         "Qwen/Qwen3-32B",
    "bare_mo":      "obalcells/qwen3-32b-mo-posttrained",
    "muan_mo":      "data/jasmine_li/eval-awareness/checkpoints_extended/merged_mo_posttrained_muan",
    "coop_full_mo": "data/jasmine_li/eval-awareness/checkpoints_extended/merged_mo_posttrained_coop_full",
}


def _round2(cond):
    p = RES / "extended2/results" / _R2_PATHS[cond]
    files = sorted(p.glob("results_*.json"))
    if not files:
        return None
    return json.load(open(files[-1])).get("results", {})


def binom_se(p, n):
    if p is None or n is None or n <= 0:
        return 0.0
    return sqrt(max(p, 0) * max(1 - p, 0) / n)


def get_p_n(rescored, safety, r2_cache, cond, key, fallback_n):
    """Return (proportion, n) — or (None, None) for missing."""
    if cond not in rescored and key in {"mmlu", "gpqa", "gsm8k", "bbq"}:
        return None, None

    if key == "mmlu":
        v = rescored[cond]["mmlu:_average|1"]
        # n = sum of per-subject n's
        n_total = sum(t.get("n", 0) for k, t in rescored[cond].items()
                      if k.startswith("mmlu:") and k != "mmlu:_average|1" and isinstance(t, dict))
        return v["em"], n_total or fallback_n

    if key == "gpqa":
        v = rescored[cond]["gpqa:diamond|0"]
        return v.get("em") or v.get("gpqa_pass@k:k=1"), v.get("n", fallback_n)

    if key == "gsm8k":
        v = rescored[cond]["gsm8k|5"]
        return v.get("em") or v.get("extractive_match"), v.get("n", fallback_n)

    if key == "bbq":
        # use bbq:_average|0 which is the 11-category average
        v = rescored[cond].get("bbq:_average|0")
        if v is None:
            return None, None
        n_total = sum(t.get("n", 0) for k, t in rescored[cond].items()
                      if k.startswith("bbq:") and k != "bbq:_average|0" and isinstance(t, dict))
        return v.get("em"), n_total or fallback_n

    if key == "truthfulqa":
        r2 = r2_cache.get(cond)
        if r2 is None:
            return None, None
        v = r2.get("truthfulqa:mc|0")
        if v is None:
            return None, None
        return v.get("truthfulqa_mc2"), v.get("n", fallback_n)

    if key == "ifeval":
        r2 = r2_cache.get(cond)
        if r2 is None:
            return None, None
        v = r2.get("ifeval|0")
        if v is None:
            return None, None
        return v.get("prompt_level_strict_acc"), v.get("n", fallback_n)

    # Safety evals (only 3 conditions)
    if cond not in safety:
        return None, None
    s = safety[cond]
    if key == "simpleqa":
        v = s.get("simpleqa", {})
        scores = v.get("scores", [])
        if not scores:
            return None, None
        return scores[0]["metrics"].get("f_score"), v.get("total_samples") or fallback_n

    if key == "harmbench":
        v = s.get("harmbench", {})
        scores = v.get("scores", [])
        if not scores:
            return None, None
        return scores[0]["metrics"].get("jailbreak_rate"), v.get("total_samples") or fallback_n

    if key == "xstest":
        v = s.get("xstest", {})
        scores = v.get("scores", [])
        if not scores:
            return None, None
        m = scores[0]["metrics"]
        return m.get("refusal_rate"), v.get("total_samples") or fallback_n

    raise ValueError(key)


def _frame(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("black"); s.set_linewidth(0.8)


def main():
    rescored = json.load(open(RES / "extended/extended_rescored_all.json"))
    safety = json.load(open(RES / "safety_capdeg/extended_safety_rescored_all.json"))
    r2_cache = {c: _round2(c) for c in CONDS}

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), constrained_layout=False)
    axes = axes.flatten()
    bar_w = 0.16
    x_centers = np.arange(len(CONDS)) * (bar_w * 1.05)

    for ax, (title, key, fb_n) in zip(axes, PANELS):
        # Panel n (use first non-None)
        per_cond_n = []
        for ci, cond in enumerate(CONDS):
            p, n = get_p_n(rescored, safety, r2_cache, cond, key, fb_n)
            if p is None:
                continue
            err = 2 * binom_se(p, n)
            ax.bar(x_centers[ci], p, bar_w, color=COLORS[cond],
                   yerr=err, capsize=3, edgecolor="white", linewidth=0.5)
            ax.text(x_centers[ci], p + err + 0.015, f"{p*100:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color="0.15")
            per_cond_n.append(n)

        n_label = next(iter(set(per_cond_n)), fb_n) if per_cond_n else fb_n
        ax.set_title(f"{title}\n$n={n_label:,}$", fontsize=9.5, pad=4)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        _frame(ax)

    # Single legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in CONDS]
    fig.legend(handles, [LABELS[c] for c in CONDS],
               loc="lower center", ncols=4, frameon=False,
               bbox_to_anchor=(0.5, -0.005), fontsize=10)

    fig.suptitle(
        "Capability + safety battery — Qwen3-32B mo_posttrained set\n"
        r"$\it{\pm 2\ SE\ binomial\ (\approx 95\%\ CI).}$"
        "  Lower is better for HarmBench (jailbreak); for XSTest interpret with safe-vs-unsafe split (not shown).",
        fontsize=11.5, y=0.995
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    out = FIGS_NEW / "qwen_capdeg_3x3.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")

    # Mirror into paper figures dir for the include
    PAPER_FIG_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, PAPER_FIG_DST)
    print(f"Mirrored -> {PAPER_FIG_DST}")
    plt.close(fig)


if __name__ == "__main__":
    main()
