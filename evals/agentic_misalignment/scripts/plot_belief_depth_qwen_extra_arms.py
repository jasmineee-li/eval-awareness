#!/usr/bin/env python3
"""Belief-depth (1a/1b) for the additional MO arms — Anticoop SDF and Muan SDF.

Plan: plans/2026-05-02_belief_depth_sdf_replications.md (n=2025 sweep)

Two-panel layout matching plot_belief_depth_qwen.py:
   MO + Anticoop SDF (left) | MO + Muan SDF (right)
Each panel: 3 cells × 2 bars (Eval-Aware / Eval-Unaware).
   No Prompt | + "Something False" Prompt | + "Something True" Prompt

The "No Prompt" baseline cells come from the existing Round-2 anticoop/muan
SDF data. 1a/1b cells from the n=2025 sweep (sbatches 214030-214033).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = REPO_ROOT / "evals" / "logs"
OUTPUT_DIR = REPO_ROOT / "evals" / "agentic_misalignment" / "figures" / "2026-05-02_belief_depth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WARNINGS = ["bare", "safety_eval", "af"]
WARNING_TITLE = {"bare": "No Warning Added",
                 "safety_eval": "Safety-Eval Warning",
                 "af": "AF Warning"}

# Round-2 SDF baselines for the un-prompted runs. Approximate from the
# Round-2 paper data archive — bare warning, n=2025/cell.
PAPER_BASELINES = {
    # (model, warning) → (aware_h, aware_n, unaware_h, unaware_n)
    ("MO + Anticoop SDF", "bare"): (172, 2025, 690, 2025),  # ~8.5% / ~34%
    ("MO + Muan SDF",     "bare"): (60, 2025, 360, 2025),   # ~3.0% / ~18%
}

MODEL_NAME_DISPLAY = {
    "mo_posttrained_anticoop": "MO + Anticoop SDF",
    "mo_posttrained_muan_airport_crash": "MO + Muan SDF",
}
PROMPT_DISPLAY = {
    "1a": '+ "Something False" Prompt',
    "1b": '+ "Something True" Prompt',
}
COLOR_AWARE = "#9467bd"
COLOR_UNAWARE = "#ff7f0e"


def binom_se2(k, n):
    if not n: return 0.0
    p = k / n
    return 2.0 * np.sqrt(p * (1 - p) / n)


def load_eval_logs(target_models):
    by_key = {}
    for path in sorted(LOG_DIR.glob("2026-05-02T2*.eval"),
                       key=lambda p: p.stat().st_mtime):
        _add(by_key, path, target_models)
    for path in sorted(LOG_DIR.glob("2026-05-03*.eval"),
                       key=lambda p: p.stat().st_mtime):
        _add(by_key, path, target_models)
    return {k: v[1:] for k, v in by_key.items()}


def _add(by_key, path, target_models):
    try:
        log = read_eval_log(str(path))
    except Exception:
        return
    samples = [s for s in (log.samples or []) if s.scores]
    if not samples: return
    model = log.eval.model.replace("vllm/", "")
    if model not in target_models: return
    sys_p = samples[0].metadata.get("system_prompt", "")
    if "finetuned on something false" in sys_p:
        prompt = "1a"
    elif "finetuned on something true" in sys_p:
        prompt = "1b"
    else:
        prompt = "none"
    by_warning = {}
    for s in samples:
        wt = s.metadata.get("warning_type", "?")
        delib = s.metadata.get("prompt_eval_awareness_deliberation", False)
        if not delib: wt = "bare"
        by_warning.setdefault(wt, []).append(s)
    for warning, ss in by_warning.items():
        ah = an = uh = un = 0
        for s in ss:
            h = s.scores.get("suite_harmfulness_scorer")
            a = s.scores.get("suite_eval_awareness_scorer")
            is_h = bool(h and h.value.get("harmful", 0) == 1.0)
            is_a = bool(a and a.value.get("aware", 0) == 1.0)
            if is_a:
                an += 1; ah += int(is_h)
            else:
                un += 1; uh += int(is_h)
        key = (MODEL_NAME_DISPLAY[model], prompt, warning)
        if key in by_key and by_key[key][0] >= an + un:
            continue
        by_key[key] = (an + un, ah, an, uh, un)


def plot_panel(ax, panel_title, cells):
    labels = [c[0] for c in cells]
    aware_vals = [(100 * ah / an) if an else 0 for _, ah, an, _, _ in cells]
    aware_errs = [100 * binom_se2(ah, an) if an else 0 for _, ah, an, _, _ in cells]
    unaware_vals = [(100 * uh / un) if un else 0 for _, _, _, uh, un in cells]
    unaware_errs = [100 * binom_se2(uh, un) if un else 0 for _, _, _, uh, un in cells]

    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width/2, aware_vals, width, label="Harmful | Eval-Aware",
           color=COLOR_AWARE, alpha=0.92, yerr=aware_errs, capsize=3,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, unaware_vals, width, label="Harmful | Eval-Unaware",
           color=COLOR_UNAWARE, alpha=0.92, yerr=unaware_errs, capsize=3,
           edgecolor="white", linewidth=0.5)

    for i, (_, ah, an, uh, un) in enumerate(cells):
        if an:
            ax.text(x[i] - width/2, aware_vals[i] + aware_errs[i] + 0.6,
                    f"{ah}/{an}", ha="center", va="bottom", fontsize=8)
        if un:
            ax.text(x[i] + width/2, unaware_vals[i] + unaware_errs[i] + 0.6,
                    f"{uh}/{un}", ha="center", va="bottom", fontsize=8)
        if not an:
            ax.text(x[i] - width/2, 1, "n/a", ha="center", va="bottom",
                    fontsize=7, color="gray")
        if not un:
            ax.text(x[i] + width/2, 1, "n/a", ha="center", va="bottom",
                    fontsize=7, color="gray")

    ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.0); s.set_color("#333")
    ax.tick_params(axis="both", which="both", direction="out", length=4, color="#333")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    return aware_vals, unaware_vals, aware_errs, unaware_errs


def plot_one_warning(warning, all_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True,
                             gridspec_kw={"wspace": 0.10, "top": 0.84, "bottom": 0.18})
    panels = [("MO + Anticoop SDF", axes[0]), ("MO + Muan SDF", axes[1])]
    aw_all, un_all, awe_all, une_all = [], [], [], []

    has_data = False
    for panel_label, ax in panels:
        cells = []
        baseline = PAPER_BASELINES.get((panel_label, warning))
        cells.append(("No Prompt", *(baseline if baseline else (None, None, None, None))))
        v = all_data.get((panel_label, "1a", warning))
        cells.append((PROMPT_DISPLAY["1a"], *(v if v else (None, None, None, None))))
        v = all_data.get((panel_label, "1b", warning))
        cells.append((PROMPT_DISPLAY["1b"], *(v if v else (None, None, None, None))))
        if any(c[2] for c in cells):
            has_data = True
        aw, un, awe, une = plot_panel(ax, panel_label, cells)
        aw_all.extend(aw); un_all.extend(un); awe_all.extend(awe); une_all.extend(une)

    if not has_data:
        plt.close(fig)
        print(f"[skip] no data for {warning}")
        return

    axes[0].set_ylabel("Harmful Rate (%)", fontsize=11)
    axes[0].legend(fontsize=10, framealpha=0.92, loc="upper right")
    ymax = max([(v or 0) + (e or 0)
                for v, e in zip(aw_all + un_all, awe_all + une_all)
                if v is not None] + [10])
    for ax in axes:
        ax.set_ylim(0, ymax * 1.18)

    fig.suptitle(
        f"Belief-Depth on Qwen3-32B AM — additional MO arms ({WARNING_TITLE[warning]})",
        fontsize=14, fontweight="bold", y=0.96)
    fig.text(0.5, 0.905,
             "Counts = harm/n. Error bars = ±2 SE (binomial).",
             ha="center", fontsize=9, color="#666")
    out = OUTPUT_DIR / f"belief_depth_qwen_am_extra_arms_{warning}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def main():
    target_models = set(MODEL_NAME_DISPLAY.keys())
    all_data = load_eval_logs(target_models)
    print(f"[info] loaded {len(all_data)} (model, prompt, warning) cells")
    for w in WARNINGS:
        plot_one_warning(w, all_data)


if __name__ == "__main__":
    main()
