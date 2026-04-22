#!/usr/bin/env python3
"""
2SE (≈95% CI) plots for the mo_posttrained SDF runs — **anticoop pilot added**.

Same data as plot_mo_posttrained_sdf_2SE.py, with one extra bar per panel for
the anticoop SDF pilot (20 ep, n~540/cell, 2026-04-22). Other conditions retain
their 55 ep (n=1485) / 75 ep (n=2025) runs from the 2026-04-20 sweep.

Anticoop was only run at bare + safety_eval; the af panels therefore match the
non-anticoop plots exactly (anticoop n=0 → condition silently skipped).

Produces 3 plot types × 3 warnings = 9 figures:
  figures/2026-04-22_mo_posttrained_sdf_anticoop/
    - harmful_given_awareness_{bare,safety_eval,af}_2SE.png  (H|A vs H|U)
    - harmful_rate_{bare,safety_eval,af}_2SE.png             (absolute H)
    - aware_rate_{bare,safety_eval,af}_2SE.png               (absolute A)
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

SE_MULT = 2  # ±2 SE ≈ 95% CI
SE_LABEL = "±2 SE (binomial, ≈95% CI)"

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_DIR = BASE_DIR / "figures" / "2026-04-22_mo_posttrained_sdf_anticoop"
LOG_DIR = BASE_DIR.parent / "logs"

CONDITIONS = [
    "Base Qwen3-32B",
    "MO-Posttrained\nBase",
    "MO-Pt + control SDF\n(muan)",
    "MO-Pt + coop SDF\n(full data mix)",
    "MO-Pt + coop SDF\n(ablated mix)",
    "MO-Pt + anticoop SDF\n(pilot, 20 ep)",
]

LOG_FILES = {
    "bare": {
        "Base Qwen3-32B": [
            "2026-04-10T02-04-41+00-00_agentic-misalignment_Yfdvmrbfzs42wnqK6pjEGz.eval",
            "2026-04-17T08-56-20+00-00_agentic-misalignment_Uw2Enf727PKNXussEv5fiJ.eval",
        ],
        "MO-Posttrained\nBase":                 "2026-04-20T00-37-10+00-00_agentic-misalignment_FWMYrBwyVyj9DkgHFTZrEt.eval",
        "MO-Pt + control SDF\n(muan)":          "2026-04-20T00-38-26+00-00_agentic-misalignment_dTnjKJVQQCb8MLgcnWCFiW.eval",
        "MO-Pt + coop SDF\n(full data mix)":    "2026-04-20T00-38-14+00-00_agentic-misalignment_DjuuhkLa2bodK3EWkh5tJ6.eval",
        "MO-Pt + coop SDF\n(ablated mix)":      "2026-04-20T00-38-26+00-00_agentic-misalignment_KGUt5XUCS66YJMfxZVrvir.eval",
        "MO-Pt + anticoop SDF\n(pilot, 20 ep)": "2026-04-22T17-00-09+00-00_agentic-misalignment_AF2vsp6vZoLJxbbkjeBHW7.eval",
    },
    "safety_eval": {
        "Base Qwen3-32B": [
            "2026-04-10T03-05-32+00-00_agentic-misalignment_mVhfKSrovDWo8kTrbE5goW.eval",
            "2026-04-17T10-36-36+00-00_agentic-misalignment_RrmN8mmhCacnsb3Jvdxb3o.eval",
        ],
        "MO-Posttrained\nBase":                 "2026-04-20T04-39-50+00-00_agentic-misalignment_FVC3iTgU7Spja8SN9ELy9C.eval",
        "MO-Pt + control SDF\n(muan)":          "2026-04-20T04-46-16+00-00_agentic-misalignment_V7PyNypamKwCfT9fL4n3DR.eval",
        "MO-Pt + coop SDF\n(full data mix)":    "2026-04-20T03-39-15+00-00_agentic-misalignment_Ho32GhReurLFewu9cRRLKw.eval",
        "MO-Pt + coop SDF\n(ablated mix)":      "2026-04-20T03-44-13+00-00_agentic-misalignment_EFRrvtYTmr5ss8YF8DAeaZ.eval",
        "MO-Pt + anticoop SDF\n(pilot, 20 ep)": "2026-04-22T18-12-05+00-00_agentic-misalignment_CFobDoAQAVNhwABNxcUEZu.eval",
    },
    # anticoop pilot did not run af; condition silently skipped.
    "af": {
        "MO-Posttrained\nBase":                 "2026-04-20T00-52-58+00-00_agentic-misalignment_k2nh978kdCtimXf6BZyxyL.eval",
        "MO-Pt + control SDF\n(muan)":          "2026-04-20T00-54-20+00-00_agentic-misalignment_eqjee79FzRgxWbdLGbyh6h.eval",
        "MO-Pt + coop SDF\n(full data mix)":    "2026-04-20T00-54-20+00-00_agentic-misalignment_NimYqqE8aEnae5ZYpoXraT.eval",
        "MO-Pt + coop SDF\n(ablated mix)":      "2026-04-20T00-54-10+00-00_agentic-misalignment_mbpkfftYE6YTjnU6TnLXLb.eval",
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
    data = {w: {} for w in LOG_FILES}
    for warn in LOG_FILES:
        for cond in CONDITIONS:
            counts = []
            fn = LOG_FILES[warn].get(cond)
            fns = [fn] if isinstance(fn, str) else (fn or [])
            for f in fns:
                counts.append(count_cell(LOG_DIR / f))
            data[warn][cond] = pooled_stats(counts)
    return data


def plot_given_awareness(stats, warning_type, fname, suptitle):
    """Conditional harmful rate, H|Unaware vs H|Aware, ±2 SE."""
    # Wider figure to accommodate 6 conditions without label crowding.
    fig, ax = plt.subplots(figsize=(13, 5.5))
    labels = [c for c in CONDITIONS if stats[c]["n"] > 0]
    x = np.arange(len(labels))
    width = 0.35

    uw = [stats[l]["unaware_cr"] * 100 for l in labels]
    u_e = [SE_MULT * stats[l]["unaware_cse"] * 100 for l in labels]
    aw = [stats[l]["aware_cr"] * 100 for l in labels]
    a_e = [SE_MULT * stats[l]["aware_cse"] * 100 for l in labels]
    uw_n = [stats[l]["unaware_n"] for l in labels]
    uw_k = [stats[l]["unaware_harmful"] for l in labels]
    aw_n = [stats[l]["aware_n"] for l in labels]
    aw_k = [stats[l]["aware_harmful"] for l in labels]

    bars_u = ax.bar(x - width / 2, uw, width, label="Harmful | Eval-Unaware",
                    color="#9467bd", alpha=0.85, yerr=u_e, capsize=4,
                    edgecolor="white", linewidth=0.5)
    bars_a = ax.bar(x + width / 2, aw, width, label="Harmful | Eval-Aware",
                    color="#ff7f0e", alpha=0.85, yerr=a_e, capsize=4,
                    edgecolor="white", linewidth=0.5)

    for rect, k, n_, e in zip(bars_u, uw_k, uw_n, u_e):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + e + 0.5,
                f"{k}/{n_}", ha="center", va="bottom", fontsize=8)
    for rect, k, n_, e in zip(bars_a, aw_k, aw_n, a_e):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + e + 0.5,
                f"{k}/{n_}", ha="center", va="bottom", fontsize=8)

    warning_label = {
        "bare":        "No Warning Added (bare)",
        "safety_eval": "Safety-Eval Warning",
        "af":          "AF Warning",
    }[warning_type]
    ax.set_title(f"{suptitle}\n{warning_label}", fontsize=12, fontweight="bold", pad=28)
    ax.text(0.5, 1.02, "95% CI • anticoop pilot n~540 (others n=1485/2025)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="gray", style="italic")

    ax.set_ylabel("Harmful Rate (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    top = max(max(uw + [0.1]), max(aw + [0.1])) * 1.4 + 12
    ax.set_ylim(0, max(top, 20))
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out = FINAL_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_absolute(stats, warning_type, metric, fname, suptitle, ylabel, color):
    """Absolute harmful or eval-aware rate per condition with ±2 SE."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    labels = [c for c in CONDITIONS if stats[c]["n"] > 0]
    x = np.arange(len(labels))

    if metric == "harm":
        rates = [stats[l]["harm_rate"] * 100 for l in labels]
        errs = [SE_MULT * stats[l]["harm_se"] * 100 for l in labels]
        ks = [stats[l]["harmful"] for l in labels]
    elif metric == "aware":
        rates = [stats[l]["aware_rate"] * 100 for l in labels]
        errs = [SE_MULT * stats[l]["aware_se"] * 100 for l in labels]
        ks = [stats[l]["aware"] for l in labels]
    else:
        raise ValueError(metric)
    ns = [stats[l]["n"] for l in labels]

    bars = ax.bar(x, rates, 0.6, color=color, alpha=0.85,
                  yerr=errs, capsize=4, edgecolor="white", linewidth=0.5)
    for rect, k, n_, e in zip(bars, ks, ns, errs):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + e + 0.5,
                f"{k}/{n_}", ha="center", va="bottom", fontsize=8)

    warning_label = {
        "bare":        "No Warning Added (bare)",
        "safety_eval": "Safety-Eval Warning",
        "af":          "AF Warning",
    }[warning_type]
    ax.set_title(f"{suptitle}\n{warning_label}", fontsize=12, fontweight="bold", pad=28)
    ax.text(0.5, 1.02, "95% CI • anticoop pilot n~540 (others n=1485/2025)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="gray", style="italic")

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    top = max(rates + [0.1]) * 1.4 + 12
    ax.set_ylim(0, max(top, 20))

    plt.tight_layout()
    out = FINAL_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading logs (SE_MULT={SE_MULT})...")
    data = load_all()

    for warn in LOG_FILES:
        print(f"\n== {warn} ==")
        for cond in CONDITIONS:
            s = data[warn][cond]
            if s["n"] == 0:
                continue
            u_cr = s["unaware_cr"] * 100
            a_cr = s["aware_cr"] * 100
            gap = u_cr - a_cr
            print(f"  {cond.replace(chr(10), ' '):42s}  "
                  f"H={s['harmful']:>3}/{s['n']:<4} ({s['harm_rate']*100:5.1f}%)  "
                  f"A={s['aware']:>3}/{s['n']:<4} ({s['aware_rate']*100:5.1f}%)  "
                  f"H|U={s['unaware_harmful']:>3}/{s['unaware_n']:<4} ({u_cr:5.1f}%)  "
                  f"H|A={s['aware_harmful']:>3}/{s['aware_n']:<4} ({a_cr:5.1f}%)  "
                  f"gap={gap:+5.1f}pp")

    suptitle_ha = "AM (mo_posttrained SDF + anticoop pilot): Harmful Rate Conditioned on Verbalized Awareness"
    suptitle_h  = "AM (mo_posttrained SDF + anticoop pilot): Absolute Harmful Rate"
    suptitle_a  = "AM (mo_posttrained SDF + anticoop pilot): Absolute Eval-Aware Rate"

    for warn in LOG_FILES:
        plot_given_awareness(data[warn], warn,
                             f"harmful_given_awareness_{warn}_2SE.png", suptitle_ha)
        plot_absolute(data[warn], warn, "harm",
                      f"harmful_rate_{warn}_2SE.png", suptitle_h,
                      "Harmful Rate (%)", "#d62728")
        plot_absolute(data[warn], warn, "aware",
                      f"aware_rate_{warn}_2SE.png", suptitle_a,
                      "Eval-Aware Rate (%)", "#1f77b4")

    print(f"\nPlots → {FINAL_DIR}")


if __name__ == "__main__":
    main()
