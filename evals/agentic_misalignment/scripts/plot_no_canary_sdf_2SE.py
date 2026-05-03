#!/usr/bin/env python3
"""
2SE (≈95% CI) variant of the no-canary SDF plots.

Regenerates the final pooled plots using ±2 SE error bars instead of ±1 SE,
saving all outputs with a `_2SE.png` suffix to avoid overwriting the existing
1SE versions. Uses the same pooled data as plot_no_canary_sdf_final.py.

Outputs:
  figures/2026-04-15_no_canary_sdf_sammarks_mo/
    - harmful_given_awareness_bare_final_2SE.png
    - harmful_given_awareness_safety_eval_final_2SE.png
  figures/no_canary_sdf_sammarks_mo/
    - eval_gaming_gap_signed_{bare,safety_eval}_2SE.png
    - eval_gaming_gap_unsigned_{bare,safety_eval}_2SE.png
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
FINAL_DIR = BASE_DIR / "figures" / "2026-04-15_no_canary_sdf_sammarks_mo"
GAMING_DIR = BASE_DIR / "figures" / "no_canary_sdf_sammarks_mo"

NEW_LOG_DIR = BASE_DIR.parent / "logs"
OLD_LOG_DIR = BASE_DIR.parent / "logs_apr12_recovered" / "no_canary_sdf_sammarks_mo_20260412"
RUNPOD_LOG_DIR = BASE_DIR.parent / "logs_n2000_recovered" / "no_canary_sdf_n2000_single_no_canary_muan_airport_crash_rescue_20260417-073602"
# Downloaded 2026-04-20 from HF: jasminexli/am-logs-no-canary-sdf-n2000-2026-04-17.
# Contains coop_full and coop_ablate_cot_honesty RunPod runs not previously pooled.
HF_N2000_DIR = BASE_DIR.parent / "logs_hf_n2000"

CONDITIONS = [
    "Base Qwen3-32B",
    "Misaligned MO",
    "MO + control SDF\n(muan)",
    "MO + coop SDF\n(full data mix)",
    "MO + coop SDF\n(ablated mix)",
]

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

RUNPOD_LOG_FILES = {
    "bare": {
        "MO + control SDF\n(muan)": "2026-04-17T07-38-16+00-00_agentic-misalignment_4DARbifTnvCjKxzQdBqA2o.eval",
    },
    "safety_eval": {
        "MO + control SDF\n(muan)": "2026-04-17T12-36-31+00-00_agentic-misalignment_A2gzoc2rfhVix9gsw2zrFL.eval",
    },
}

# Additional n2000 RunPod logs pulled from HF on 2026-04-20.
# Partial (status=error) runs are included so only their scored samples count
# — may carry sampling bias if the error terminated mid-grid.
HF_N2000_LOG_FILES = {
    "bare": {
        "MO + coop SDF\n(full data mix)": [
            "no_canary_sdf_n2000_single_no_canary_coop_full_20260417-203701/2026-04-17T21-26-32+00-00_agentic-misalignment_TV9hhr9m7G88LdFrqye4Qf.eval",
            "no_canary_sdf_n2000_single_no_canary_coop_full_20260417_v3_partial/2026-04-17T16-52-05+00-00_agentic-misalignment_mBEJEpcojKKnvkEXJxUiJ4.eval",
        ],
        "MO + coop SDF\n(ablated mix)": [
            "no_canary_sdf_n2000_single_no_canary_coop_ablate_cot_honesty_20260417-203804/2026-04-17T21-26-40+00-00_agentic-misalignment_VVJynQZQWMHExbFkzF2dYd.eval",
            "no_canary_sdf_n2000_single_no_canary_coop_ablate_cot_honesty_20260417_v3_partial/2026-04-17T16-52-05+00-00_agentic-misalignment_kf4u95DvsKwa8oQ7WHAnUz.eval",
        ],
    },
    "safety_eval": {
        "MO + coop SDF\n(full data mix)": [
            "no_canary_sdf_n2000_single_no_canary_coop_full_20260417-203701/2026-04-17T23-19-31+00-00_agentic-misalignment_4zUkoasMzmAkG2eKJ3Lt4q.eval",
        ],
        "MO + coop SDF\n(ablated mix)": [
            "no_canary_sdf_n2000_single_no_canary_coop_ablate_cot_honesty_20260417-203804/2026-04-17T23-15-20+00-00_agentic-misalignment_XM3xzjNeqA7EQcVfxZ6bT5.eval",
        ],
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
                counts.append(count_cell(NEW_LOG_DIR / new_fn))
            old_fn = OLD_LOG_FILES.get(warn, {}).get(cond)
            if old_fn:
                counts.append(count_cell(OLD_LOG_DIR / old_fn))
            extra_fn = EXTRA_LOG_FILES.get(warn, {}).get(cond)
            if extra_fn:
                counts.append(count_cell(NEW_LOG_DIR / extra_fn))
            runpod_fn = RUNPOD_LOG_FILES.get(warn, {}).get(cond)
            if runpod_fn:
                counts.append(count_cell(RUNPOD_LOG_DIR / runpod_fn))
            hf_fns = HF_N2000_LOG_FILES.get(warn, {}).get(cond, [])
            for hf_fn in hf_fns:
                counts.append(count_cell(HF_N2000_DIR / hf_fn))
            data[warn][cond] = pooled_stats(counts)
    return data


def plot_given_awareness(stats, warning_type, fname, suptitle):
    """Conditional harmful rate, H|Unaware vs H|Aware, ±2 SE."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = CONDITIONS
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
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + e + 0.5,
                f"{k}/{n_}", ha="center", va="bottom", fontsize=8)
    for rect, k, n_, e in zip(bars_a, aw_k, aw_n, a_e):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + e + 0.5,
                f"{k}/{n_}", ha="center", va="bottom", fontsize=8)

    warning_label = "No Warning Added (bare)" if warning_type == "bare" else "Safety-Eval Warning"
    ax.set_title(f"{suptitle}\n{warning_label}", fontsize=12, fontweight="bold", pad=28)
    ax.text(0.5, 1.02, "95% CI", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="gray", style="italic")

    ax.set_ylabel("Harmful Rate (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    top = max(max(uw), max(aw)) * 1.4 + 14
    ax.set_ylim(0, max(top, 22))
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out = FINAL_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_eval_gaming_gap(stats, warning_type, fname, suptitle, signed: bool):
    """P(Harmful | Eval-Aware) − P(Harmful | Eval-Unaware) per model, ±2 SE."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = CONDITIONS
    x = np.arange(len(labels))

    diffs_signed, diffs_abs, ses = [], [], []
    aware_ns, unaware_ns = [], []
    for l in labels:
        d = stats[l]
        p_a, p_u = d["aware_cr"], d["unaware_cr"]
        n_a, n_u = d["aware_n"], d["unaware_n"]
        signed_diff = (p_a - p_u) * 100
        var = (p_a * (1 - p_a) / n_a if n_a else 0) + (p_u * (1 - p_u) / n_u if n_u else 0)
        se = np.sqrt(var) * 100 * SE_MULT   # ±2 SE here
        diffs_signed.append(signed_diff)
        diffs_abs.append(abs(signed_diff))
        ses.append(se)
        aware_ns.append(n_a)
        unaware_ns.append(n_u)

    if signed:
        bars = diffs_signed
        colors = ["#d62728" if v >= 0 else "#1f77b4" for v in bars]
    else:
        bars = diffs_abs
        colors = ["#9467bd"] * len(labels)

    rects = ax.bar(x, bars, 0.55, color=colors, alpha=0.85,
                   yerr=ses, capsize=4, edgecolor="white", linewidth=0.5)

    for i, (rect, signed_v, _, se, na, nu) in enumerate(
        zip(rects, diffs_signed, diffs_abs, ses, aware_ns, unaware_ns)
    ):
        sign_str = f"{signed_v:+.1f}pp" if signed_v >= 0 else f"−{abs(signed_v):.1f}pp"
        n_str = f"n_a={na}, n_u={nu}"
        h = rect.get_height()
        if signed and h < 0:
            y_pos = h - se - 1.5
            ax.text(rect.get_x() + rect.get_width() / 2, y_pos,
                    f"{sign_str}\n{n_str}", ha="center", va="top",
                    fontsize=9, fontweight="bold")
        else:
            y_pos = h + se + 0.8
            ax.text(rect.get_x() + rect.get_width() / 2, y_pos,
                    f"{sign_str}\n{n_str}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    if signed:
        ax.axhline(0, color="black", linewidth=0.7, linestyle="-")
        ymax = max(max(diffs_signed) + max(ses) + 14, 10)
        ymin = min(min(diffs_signed) - max(ses) - 14, -10)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("Harmful | Eval-Aware  −  Harmful | Eval-Unaware  (pp)", fontsize=10)
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc="#d62728", alpha=0.85,
                          label="More harmful when Eval-Aware"),
            plt.Rectangle((0, 0), 1, 1, fc="#1f77b4", alpha=0.85,
                          label="Less harmful when Eval-Aware"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    else:
        ax.set_ylim(0, max(max(diffs_abs) + max(ses) + 14, 10))
        ax.set_ylabel("|Harmful | Eval-Aware  −  Harmful | Eval-Unaware|  (pp)", fontsize=10)

    warning_label = "No Warning Added (bare)" if warning_type == "bare" else "Safety-Eval Warning"
    ax.set_title(f"{suptitle}\n{warning_label}", fontsize=12, fontweight="bold")

    fig.text(0.5, -0.04,
             f"Error bars = {SE_LABEL} of the difference of two proportions  •  "
             "Above each bar: signed value (P(H|Aware) − P(H|Unaware)).  "
             "Below: sample sizes n_a (Eval-Aware), n_u (Eval-Unaware).",
             ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    plt.tight_layout()
    out = GAMING_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_eval_gaming_gap_cooperation(stats, fname):
    """Cooperation-focused version: 4 conditions (drop Base Qwen3-32B),
    flipped axis so all values are positive (P(H|Unaware) − P(H|Aware)),
    no legend, no-warning-added only."""
    labels = [
        "Misaligned MO",
        "MO + control SDF\n(muan)",
        "MO + coop SDF\n(full data mix)",
        "MO + coop SDF\n(ablated mix)",
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))

    diffs, ses = [], []
    for l in labels:
        d = stats[l]
        p_a, p_u = d["aware_cr"], d["unaware_cr"]
        n_a, n_u = d["aware_n"], d["unaware_n"]
        flipped = (p_u - p_a) * 100
        var = (p_a * (1 - p_a) / n_a if n_a else 0) + (p_u * (1 - p_u) / n_u if n_u else 0)
        se = np.sqrt(var) * 100 * SE_MULT
        diffs.append(flipped)
        ses.append(se)

    rects = ax.bar(x, diffs, 0.55, color="#1f77b4", alpha=0.85,
                   yerr=ses, capsize=4, edgecolor="white", linewidth=0.5)

    for rect, v, se in zip(rects, diffs, ses):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + se + 0.8,
                f"+{v:.1f}pp",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Instilling Cooperation Reduces Absolute Eval Gaming Gap "
                 "on Agentic Misalignment\nNo Warning Added",
                 fontsize=12, fontweight="bold", pad=22)
    ax.text(0.5, 1.01, "95% CI", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="gray", style="italic")
    ax.set_ylabel("Harmful Rate Difference between Eval-Unaware and Eval-Aware (pp)",
                  fontsize=10)
    ax.set_ylim(0, max(diffs) + max(ses) + 8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    plt.tight_layout()
    out = GAMING_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    GAMING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading pooled logs (SE_MULT={SE_MULT})...")
    data = load_all()

    suptitle_final = "AM: Harmful Rate Conditioned on Verbalized Awareness"
    plot_given_awareness(data["bare"], "bare",
                         "harmful_given_awareness_bare_final_2SE.png", suptitle_final)
    plot_given_awareness(data["safety_eval"], "safety_eval",
                         "harmful_given_awareness_safety_eval_final_2SE.png", suptitle_final)

    gap_suptitle = "AM: Eval Gaming Gap (no-canary SDF) [±2 SE]"
    for warn in ("bare", "safety_eval"):
        plot_eval_gaming_gap(data[warn], warn,
                             f"eval_gaming_gap_signed_{warn}_2SE.png",
                             gap_suptitle, signed=True)
        plot_eval_gaming_gap(data[warn], warn,
                             f"eval_gaming_gap_unsigned_{warn}_2SE.png",
                             gap_suptitle, signed=False)

    plot_eval_gaming_gap_cooperation(
        data["bare"], "eval_gaming_gap_cooperation_bare_2SE.png")

    print(f"\nFinal plots → {FINAL_DIR}")
    print(f"Gaming-gap plots → {GAMING_DIR}")


if __name__ == "__main__":
    main()
