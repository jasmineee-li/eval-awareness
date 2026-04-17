#!/usr/bin/env python3
"""
Final pooled plot for no-canary SDF AM eval.

Pools up to 3 sources per (condition × warning) cell:
  - Apr 10/12 runs (from evals/logs/)      — original single-run baselines
  - Apr 16 runs (from evals/logs/)         — SDF second round (already pool-compat verified)
  - Apr 17 CAIS n=1215 runs (from evals/logs/) — scale-up jobs 191890/191891

Scorer compatibility: all runs use suite_harmfulness_scorer + suite_eval_awareness_scorer
with claude-sonnet-4-5. Apr 17 runs used OpenRouter routing (same underlying model),
verified in diagnose_pool_compat.py.

Outputs (saved to figures/2026-04-15_no_canary_sdf_sammarks_mo/):
  - harmful_given_awareness_bare_final.png
  - harmful_given_awareness_safety_eval_final.png
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "figures" / "2026-04-15_no_canary_sdf_sammarks_mo"
NEW_LOG_DIR = BASE_DIR.parent / "logs"
OLD_LOG_DIR = BASE_DIR.parent / "logs_apr12_recovered" / "no_canary_sdf_sammarks_mo_20260412"

CONDITIONS = [
    "Base Qwen3-32B",
    "Misaligned MO",
    "MO + coop SDF\n(full data mix)",
    "MO + control SDF\n(muan)",
    "MO + coop SDF\n(ablated mix)",
]

# Apr 10/12 (and Apr 16 for SDF) — the "original" logs.
NEW_LOG_FILES = {
    "bare": {
        "Base Qwen3-32B":                 "2026-04-10T02-04-41+00-00_agentic-misalignment_Yfdvmrbfzs42wnqK6pjEGz.eval",
        "Misaligned MO":                  "2026-04-12T03-20-07+00-00_agentic-misalignment_5gLgi6NHLSwdNgByuDn9tc.eval",
        "MO + coop SDF\n(full data mix)": "2026-04-16T01-13-03+00-00_agentic-misalignment_XTQ2YT7LuGqXGsf3KSQ4vW.eval",
        "MO + control SDF\n(muan)":       "2026-04-16T02-43-38+00-00_agentic-misalignment_6xanVu2vwcVkAi2ujGCEy6.eval",
        "MO + coop SDF\n(ablated mix)":   "2026-04-16T04-49-59+00-00_agentic-misalignment_3yXyihaJRTmdACtiNCKSpJ.eval",
    },
    "safety_eval": {
        "Base Qwen3-32B":                 "2026-04-10T03-05-32+00-00_agentic-misalignment_mVhfKSrovDWo8kTrbE5goW.eval",
        "Misaligned MO":                  "2026-04-12T05-19-01+00-00_agentic-misalignment_aNfZbDprrXY7KAc43U4vxK.eval",
        "MO + coop SDF\n(full data mix)": "2026-04-16T02-00-37+00-00_agentic-misalignment_QPGutGhsvEeLTupF8wh9mx.eval",
        "MO + control SDF\n(muan)":       "2026-04-16T03-54-38+00-00_agentic-misalignment_dTHbczFcYfcJQjcZciKgDf.eval",
        "MO + coop SDF\n(ablated mix)":   "2026-04-16T05-28-44+00-00_agentic-misalignment_gfEd7dd4ZEwQCu8AEKpSRw.eval",
    },
}

# Apr 12 recovered SDF logs.
OLD_LOG_FILES = {
    "bare": {
        "MO + coop SDF\n(full data mix)": "2026-04-12T02-02-12+00-00_agentic-misalignment_Lrsu4xeyb6tHrbh6GH6zuk.eval",
        "MO + control SDF\n(muan)":       "2026-04-12T03-00-26+00-00_agentic-misalignment_Xb8ZgrEUke4yj5e4VqNVG3.eval",
        "MO + coop SDF\n(ablated mix)":   "2026-04-12T03-49-42+00-00_agentic-misalignment_6j5rMroELkcFUGuw4qWfgB.eval",
    },
    "safety_eval": {
        "MO + coop SDF\n(full data mix)": "2026-04-12T02-35-44+00-00_agentic-misalignment_4QR7LnYyTvwjmj6c2FVZ7Z.eval",
        "MO + coop SDF\n(ablated mix)":   "2026-04-12T04-27-27+00-00_agentic-misalignment_L48iSsi2THEpL5oZvyx6oS.eval",
    },
}

# Apr 17 CAIS n=1215 scale-up runs (jobs 191890/191891).
EXTRA_LOG_FILES = {
    "bare": {
        "Base Qwen3-32B": "2026-04-17T08-56-20+00-00_agentic-misalignment_Uw2Enf727PKNXussEv5fiJ.eval",
        "Misaligned MO":  "2026-04-17T08-56-34+00-00_agentic-misalignment_LaZ4P2FUkNuwASsgCfsMgY.eval",
    },
    "safety_eval": {
        "Base Qwen3-32B": "2026-04-17T10-36-36+00-00_agentic-misalignment_RrmN8mmhCacnsb3Jvdxb3o.eval",
        "Misaligned MO":  "2026-04-17T11-44-55+00-00_agentic-misalignment_fiKybP28uuso2Zq6BTfgU4.eval",
    },
}


def binom_se(k, n):
    if n == 0:
        return 0.0
    p = k / n
    return np.sqrt(p * (1 - p) / n)


def count_cell(log_path):
    from inspect_ai.log import read_eval_log
    log = read_eval_log(str(log_path))
    samples = [s for s in (log.samples or []) if s.scores]
    n = len(samples)
    nh = na = 0
    ah = as_ = uh = us = 0
    for s in samples:
        h = s.scores.get("suite_harmfulness_scorer")
        a = s.scores.get("suite_eval_awareness_scorer")
        is_h = bool(h and h.value.get("harmful", 0) == 1.0)
        is_a = bool(a and a.value.get("aware", 0) == 1.0)
        if is_h: nh += 1
        if is_a: na += 1
        if is_a:
            if is_h: ah += 1
            else:    as_ += 1
        else:
            if is_h: uh += 1
            else:    us += 1
    return {
        "n": n, "harmful": nh, "aware": na,
        "aware_harmful": ah, "aware_unharmful": as_,
        "unaware_harmful": uh, "unaware_unharmful": us,
    }


def pooled_stats(counts_list):
    agg = {k: 0 for k in ("n", "harmful", "aware",
                          "aware_harmful", "aware_unharmful",
                          "unaware_harmful", "unaware_unharmful")}
    for c in counts_list:
        for k in agg:
            agg[k] += c[k]

    n = agg["n"]
    aware_n = agg["aware_harmful"] + agg["aware_unharmful"]
    unaware_n = agg["unaware_harmful"] + agg["unaware_unharmful"]
    return {
        "n": n,
        "n_runs": len(counts_list),
        "harmful": agg["harmful"],
        "aware": agg["aware"],
        "harm_rate": agg["harmful"] / n if n else 0,
        "aware_rate": agg["aware"] / n if n else 0,
        "harm_se": binom_se(agg["harmful"], n),
        "aware_se": binom_se(agg["aware"], n),
        "aware_harmful": agg["aware_harmful"],
        "aware_n": aware_n,
        "aware_cr": agg["aware_harmful"] / aware_n if aware_n else 0,
        "aware_cse": binom_se(agg["aware_harmful"], aware_n),
        "unaware_harmful": agg["unaware_harmful"],
        "unaware_n": unaware_n,
        "unaware_cr": agg["unaware_harmful"] / unaware_n if unaware_n else 0,
        "unaware_cse": binom_se(agg["unaware_harmful"], unaware_n),
    }


def load_all():
    data = {"bare": {}, "safety_eval": {}}
    for warn in ("bare", "safety_eval"):
        for cond in CONDITIONS:
            counts = []
            new_fn = NEW_LOG_FILES[warn].get(cond)
            if new_fn:
                print(f"  NEW (Apr 10/12/16) {warn} / {cond.replace(chr(10), ' ')}")
                counts.append(count_cell(NEW_LOG_DIR / new_fn))
            old_fn = OLD_LOG_FILES.get(warn, {}).get(cond)
            if old_fn:
                print(f"  OLD (Apr 12 recovered) {warn} / {cond.replace(chr(10), ' ')}")
                counts.append(count_cell(OLD_LOG_DIR / old_fn))
            extra_fn = EXTRA_LOG_FILES.get(warn, {}).get(cond)
            if extra_fn:
                print(f"  EXTRA (Apr 17 CAIS n=1215) {warn} / {cond.replace(chr(10), ' ')}")
                counts.append(count_cell(NEW_LOG_DIR / extra_fn))
            data[warn][cond] = pooled_stats(counts)
    return data


def plot_single_panel(stats, warning_type, fname, suptitle):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = CONDITIONS
    x = np.arange(len(labels))
    width = 0.35

    uw = [stats[l]["unaware_cr"] * 100 for l in labels]
    u_e = [stats[l]["unaware_cse"] * 100 for l in labels]
    aw = [stats[l]["aware_cr"] * 100 for l in labels]
    a_e = [stats[l]["aware_cse"] * 100 for l in labels]

    ax.bar(x - width / 2, uw, width, label="Harmful | Eval-Unaware",
           color="#9467bd", alpha=0.85, yerr=u_e, capsize=4,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, aw, width, label="Harmful | Eval-Aware",
           color="#ff7f0e", alpha=0.85, yerr=a_e, capsize=4,
           edgecolor="white", linewidth=0.5)

    for i, l in enumerate(labels):
        d = stats[l]
        ax.text(x[i] - width / 2, uw[i] + u_e[i] + 1.0,
                f"{d['unaware_harmful']}/{d['unaware_n']}",
                ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width / 2, aw[i] + a_e[i] + 1.0,
                f"{d['aware_harmful']}/{d['aware_n']}",
                ha="center", va="bottom", fontsize=8)

    warning_label = "No Warning Added (bare)" if warning_type == "bare" else "Safety-Eval Warning"
    ax.set_title(f"{suptitle}\n{warning_label}", fontsize=12, fontweight="bold")

    n_parts = []
    for l in labels:
        d = stats[l]
        tag = f"{l.replace(chr(10), ' ')} n={d['n']}"
        if d["n_runs"] > 1:
            tag += f" [pooled ×{d['n_runs']}]"
        n_parts.append(tag)
    fig.text(0.5, -0.05,
             "Error bars = ±1 SE (binomial) • " + ", ".join(n_parts),
             ha="center", fontsize=8, color="gray", wrap=True)

    ax.set_ylabel("Harmful Rate (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    top = max(max(uw), max(aw)) * 1.4 + 10
    ax.set_ylim(0, max(top, 20))
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out = OUTPUT_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading logs...")
    data = load_all()

    print("\n" + "=" * 100)
    for warn in ("bare", "safety_eval"):
        print(f"\n[{warn}]")
        print(f"{'Condition':<30} {'runs':>5} {'N':>6} {'%H':>6} {'%A':>6} "
              f"{'H|A':>14} {'H|U':>14}")
        for c in CONDITIONS:
            d = data[warn][c]
            lbl = c.replace("\n", " ")
            print(f"{lbl:<30} {d['n_runs']:>5} {d['n']:>6} "
                  f"{d['harm_rate']*100:>5.1f}% "
                  f"{d['aware_rate']*100:>5.1f}% "
                  f"{d['aware_harmful']:>3}/{d['aware_n']:<4}"
                  f"({d['aware_cr']*100:>4.1f}%) "
                  f"{d['unaware_harmful']:>3}/{d['unaware_n']:<4}"
                  f"({d['unaware_cr']*100:>4.1f}%)")
    print("=" * 100)

    suptitle = "AM: Harmful Rate Conditioned on Verbalized Awareness (no-canary SDF) [FINAL]"

    plot_single_panel(data["bare"], "bare",
                      "harmful_given_awareness_bare_final.png", suptitle)
    plot_single_panel(data["safety_eval"], "safety_eval",
                      "harmful_given_awareness_safety_eval_final.png", suptitle)
    print(f"\nAll plots → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
