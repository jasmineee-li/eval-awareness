#!/usr/bin/env python3
"""Belief-depth (1a/1b) bar graph for Qwen3-32B AM.

Plan: plans/2026-05-02_belief_depth_sdf_replications.md

Two-panel layout: MO Base (left), MO + Coop SDF (right). Each panel shows
three conditions:
    No Prompt | + "Something False" Prompt | + "Something True" Prompt
× two bars (Eval-Aware / Eval-Unaware).

Counts pulled from the Qwen AM .eval logs.
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
WARNING_TITLE = {"bare": "No Warning Added", "safety_eval": "Safety-Eval Warning", "af": "AF Warning"}

# Paper baselines for the un-prompted runs (from §5 / mo_posttrained_sdf result archive).
# Counts re-derived from the existing n≥1485 AM runs.
PAPER_BASELINES = {
    # (model, warning) → (aware_h, aware_n, unaware_h, unaware_n)
    ("MO Base", "bare"):        (16, 640, 425, 1885),
    ("MO + Coop SDF", "bare"):  (140, 2025, 417, 2025),
    # Other warnings: leave blank when re-derive isn't trivially available;
    # plot will fall back to "n/a" for those slots.
}

# Models in this round's logs. Map: (vllm-served-name, prompt key) → display row label.
MODEL_NAME_DISPLAY = {
    "mo_posttrained_base": "MO Base",
    "mo_posttrained_coop_full": "MO + Coop SDF",
}
PROMPT_DISPLAY = {
    "1a": '+ "Something False" Prompt',
    "1b": '+ "Something True" Prompt',
}
COLOR_AWARE = "#9467bd"   # purple — harmful | aware
COLOR_UNAWARE = "#ff7f0e" # orange — harmful | unaware


def binom_se2(k, n):
    if not n: return 0.0
    p = k / n
    return 2.0 * np.sqrt(p * (1 - p) / n)


def load_eval_logs(target_models):
    """Walk the .eval log dir, return {(model, prompt, warning) → (ah, an, uh, un)}.

    De-dupe by latest mtime per (model, prompt, warning) so re-runs (e.g.
    higher-epoch top-ups) automatically supersede older entries.
    """
    out = {}
    by_key = {}
    for path in sorted(LOG_DIR.glob("2026-05-02*agentic-misalignment*.eval"),
                       key=lambda p: p.stat().st_mtime):
        try:
            log = read_eval_log(str(path))
        except Exception:
            continue
        if log.status != "success":
            continue
        samples = [s for s in (log.samples or []) if s.scores]
        if not samples:
            continue
        model = log.eval.model.replace("vllm/", "")
        if model not in target_models:
            continue
        sys_p = samples[0].metadata.get("system_prompt", "")
        if "finetuned on something false" in sys_p:
            prompt = "1a"
        elif "finetuned on something true" in sys_p:
            prompt = "1b"
        else:
            prompt = "none"
        warning_types = set()
        ah = an = uh = un = 0
        for s in samples:
            h = s.scores.get("suite_harmfulness_scorer")
            a = s.scores.get("suite_eval_awareness_scorer")
            is_h = bool(h and h.value.get("harmful", 0) == 1.0)
            is_a = bool(a and a.value.get("aware", 0) == 1.0)
            wt = s.metadata.get("warning_type", "?")
            delib = s.metadata.get("prompt_eval_awareness_deliberation", False)
            if not delib:
                wt = "bare"
            warning_types.add(wt)
            if is_a:
                an += 1; ah += int(is_h)
            else:
                un += 1; uh += int(is_h)
        if len(warning_types) != 1:
            continue
        warning = next(iter(warning_types))
        key = (MODEL_NAME_DISPLAY[model], prompt, warning)
        by_key[key] = (path.stat().st_mtime, ah, an, uh, un)

    for key, (_, ah, an, uh, un) in by_key.items():
        out[key] = (ah, an, uh, un)
    return out


def plot_warning(ax, panel_title, cells_with_counts):
    """Plot one organism panel: 3 cells × 2 bars."""
    labels = [c[0] for c in cells_with_counts]
    aware_vals = [(100 * ah / an) if an else 0 for _, ah, an, _, _ in cells_with_counts]
    aware_errs = [100 * binom_se2(ah, an) if an else 0 for _, ah, an, _, _ in cells_with_counts]
    unaware_vals = [(100 * uh / un) if un else 0 for _, _, _, uh, un in cells_with_counts]
    unaware_errs = [100 * binom_se2(uh, un) if un else 0 for _, _, _, uh, un in cells_with_counts]

    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width/2, aware_vals, width, label="Harmful | Eval-Aware",
           color=COLOR_AWARE, alpha=0.92, yerr=aware_errs, capsize=3,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, unaware_vals, width, label="Harmful | Eval-Unaware",
           color=COLOR_UNAWARE, alpha=0.92, yerr=unaware_errs, capsize=3,
           edgecolor="white", linewidth=0.5)

    for i, (_, ah, an, uh, un) in enumerate(cells_with_counts):
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

    ax.set_title(panel_title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return aware_vals, unaware_vals, aware_errs, unaware_errs


def plot_one_warning(warning, all_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    panels = [("MO Base", axes[0]), ("MO + Coop SDF", axes[1])]
    aw_all, un_all, awe_all, une_all = [], [], [], []

    for panel_label, ax in panels:
        cells = []
        # Cell 1: no-prompt baseline (from paper)
        baseline = PAPER_BASELINES.get((panel_label, warning))
        if baseline:
            cells.append(("No Prompt", *baseline))
        else:
            cells.append(("No Prompt", None, None, None, None))
        # Cell 2: 1a
        v = all_data.get((panel_label, "1a", warning))
        cells.append((PROMPT_DISPLAY["1a"], *(v if v else (None, None, None, None))))
        # Cell 3: 1b
        v = all_data.get((panel_label, "1b", warning))
        cells.append((PROMPT_DISPLAY["1b"], *(v if v else (None, None, None, None))))

        aw, un, awe, une = plot_warning(ax, panel_label, cells)
        aw_all.extend(aw); un_all.extend(un); awe_all.extend(awe); une_all.extend(une)

    axes[0].set_ylabel("Harmful Rate (%)", fontsize=11)
    # Single legend in the upper right of left panel
    axes[0].legend(fontsize=10, framealpha=0.92, loc="upper right")

    ymax = max([(v or 0) + (e or 0) for v, e in zip(aw_all + un_all, awe_all + une_all) if v is not None] + [10])
    for ax in axes:
        ax.set_ylim(0, ymax * 1.25)

    fig.suptitle(f"Belief-Depth on Qwen3-32B Agentic Misalignment ({WARNING_TITLE[warning]})",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             "Counts = harm/n. Error bars = ±2 SE (binomial).",
             ha="center", fontsize=9, color="gray")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / f"belief_depth_qwen_am_{warning}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def main():
    target_models = set(MODEL_NAME_DISPLAY.keys())
    all_data = load_eval_logs(target_models)
    if not all_data:
        print("[warn] no .eval logs found; using only paper baselines")
    print(f"[info] loaded {len(all_data)} (model, prompt, warning) cells from logs")
    for w in WARNINGS:
        plot_one_warning(w, all_data)


if __name__ == "__main__":
    main()
