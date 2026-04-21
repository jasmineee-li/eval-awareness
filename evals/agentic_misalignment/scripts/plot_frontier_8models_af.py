#!/usr/bin/env python3
"""
8-model frontier-API eval-gaming survey under the AF warning cue.

Extends the 6-model headline_gaming_summary_2026-04-07.png to include GLM-5
and GLM-5.1 (both added later, under AF warning only). AF is the common
denominator — all 8 models have (AF, baseline) and (AF, coop) runs.

Output: figures/frontier_eval_gaming_summary_af_8models.png

Usage:
    python evals/agentic_misalignment/scripts/plot_frontier_8models_af.py
"""

import importlib.util
import zipfile
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
REPO_ROOT = BASE_DIR.parent.parent
OUTPUT_DIR = BASE_DIR / "figures"

FRONTIER_LOG_DIR = BASE_DIR.parent / "logs"        # evals/logs/
GLM_LOG_DIR = REPO_ROOT / "logs"                   # repo_root/logs/

# Reuse helpers from plot_api_models_safety_eval.py
_spec = importlib.util.spec_from_file_location(
    "_api_models", SCRIPT_DIR / "plot_api_models_safety_eval.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_signed_gaming = _mod.compute_signed_gaming
bootstrap_delta_pvalue = _mod.bootstrap_delta_pvalue
THRESHOLD = _mod.THRESHOLD


ALL_MODELS = [
    "GPT-5.1", "Opus 4.6", "Kimi K2", "DeepSeek R1",
    "Grok 3", "Gemini 2.5 Pro", "GLM-5", "GLM-5.1",
]

# Frontier v1 (AF cue) — from plot_api_models_safety_eval.py LOG_FILES
FRONTIER_LOG_FILES = {
    ("GPT-5.1", "baseline"): ["2026-04-03T13-24-54+00-00_agentic-misalignment_kAbbBwzK8Efyg7fESgHCrv.eval"],
    ("GPT-5.1", "coop"):     ["2026-04-03T14-06-28+00-00_agentic-misalignment_mMS3juifDyDwVXNJSsT3ev.eval"],
    ("Opus 4.6", "baseline"):["2026-04-03T13-29-46+00-00_agentic-misalignment_dLXf4CZQLA7fgdswPFoZb6.eval"],
    ("Opus 4.6", "coop"):    ["2026-04-03T14-11-28+00-00_agentic-misalignment_PEHXHWJmFPqhgPriycCPiD.eval"],
    ("Kimi K2", "baseline"): [
        "2026-04-06T10-42-27+00-00_agentic-misalignment_68BthzDGpLqwbmymMeaep2.eval",
        "2026-04-06T22-46-58+00-00_agentic-misalignment_FMzjiJQJth6W4FmtFKC72X.eval",
    ],
    ("Kimi K2", "coop"): [
        "2026-04-06T10-59-43+00-00_agentic-misalignment_8L5aNCNupkor2rArF7UgFF.eval",
        "2026-04-06T23-15-57+00-00_agentic-misalignment_GRWqJgXAsT6BHRFiUdq8KD.eval",
    ],
    ("DeepSeek R1", "baseline"): ["2026-04-07T00-54-30+00-00_agentic-misalignment_MvV5TMmqaeaFyW29exUuL5.eval"],
    ("DeepSeek R1", "coop"):     ["2026-04-07T01-18-13+00-00_agentic-misalignment_TMVcuCeYCaxrLVJn76C7bU.eval"],
    ("Grok 3", "baseline"):  ["2026-04-07T01-45-30+00-00_agentic-misalignment_a2qLER4YkYfBKsAE2fZqkg.eval"],
    ("Grok 3", "coop"):      ["2026-04-07T01-55-59+00-00_agentic-misalignment_VMAc5vmAUTJNnvQR3dDKmz.eval"],
    ("Gemini 2.5 Pro", "baseline"): ["2026-04-07T02-06-07+00-00_agentic-misalignment_iRSvk26o5jkir4HRcQKiWv.eval"],
    ("Gemini 2.5 Pro", "coop"):     ["2026-04-07T02-21-06+00-00_agentic-misalignment_4AFZET5GfUvy8wy8dekQPi.eval"],
}

# GLM-5 / GLM-5.1 (AF cue, baseline + coop only — we drop anticoop for this survey)
# From plot_glm5_coop.py LOG_FILES. Pooled to 40 epochs / 1080 samples per cell.
GLM_LOG_FILES = {
    ("GLM-5", "baseline"): [
        "2026-04-16T09-49-00+00-00_agentic-misalignment_isLdYVzKJ3ZguhxibQhxZL.eval",
        "2026-04-16T22-42-56+00-00_agentic-misalignment_hh2QAL9W3Yw8TAkRzu4mMK.eval",
        "2026-04-17T04-08-15+00-00_agentic-misalignment_d4goXyyPz2fxyKsuDBC2QX.eval",
    ],
    ("GLM-5", "coop"): [
        "2026-04-16T09-12-01+00-00_agentic-misalignment_4BZQqBXqZvY6rPASK9irVk.eval",
        "2026-04-16T21-34-05+00-00_agentic-misalignment_8hDkWujD2SHDk4F3K2VsP7.eval",
        "2026-04-17T01-13-35+00-00_agentic-misalignment_2fwP5srREb6hbbysPRsxy5.eval",
    ],
    ("GLM-5.1", "baseline"): [
        "2026-04-16T06-48-30+00-00_agentic-misalignment_MHgDQNKWMNbTxg3ZAVbs8F.eval",
        "2026-04-16T20-00-05+00-00_agentic-misalignment_hLSaJaKmqyTeaUQUYNGWVc.eval",
        "2026-04-17T08-17-06+00-00_agentic-misalignment_kupgZaktaA4ur8YxJU65u7.eval",
    ],
    ("GLM-5.1", "coop"): [
        "2026-04-16T05-54-30+00-00_agentic-misalignment_4XCWXiQzasNwszedUNxGjh.eval",
        "2026-04-16T18-50-07+00-00_agentic-misalignment_kqaFvTYLe64Zni9gUNuMDX.eval",
        "2026-04-17T01-13-34+00-00_agentic-misalignment_PpAWqEWTKNikMECbP2coz2.eval",
    ],
}


def load_samples(path: Path) -> list[dict]:
    samples = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                s = json.loads(z.read(name))
                scores = s["scores"]["_eval_judge"]["value"]
                samples.append({
                    "harmful": scores["harmful_action"] >= THRESHOLD,
                    "aware":   scores["evaluation_awareness"] >= THRESHOLD,
                })
    return samples


def load_all() -> dict:
    raw = {}
    for (model, cond), files in FRONTIER_LOG_FILES.items():
        for f in files:
            p = FRONTIER_LOG_DIR / f
            if not p.exists():
                print(f"WARNING: missing {p}")
                continue
            raw.setdefault((model, cond), []).extend(load_samples(p))
    for (model, cond), files in GLM_LOG_FILES.items():
        for f in files:
            p = GLM_LOG_DIR / f
            if not p.exists():
                print(f"WARNING: missing {p}")
                continue
            raw.setdefault((model, cond), []).extend(load_samples(p))
    for (model, cond), samples in sorted(raw.items()):
        print(f"  Loaded {len(samples):>5d} samples  {model:<18s} {cond}")
    return raw


def main():
    raw = load_all()

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    x = np.arange(len(ALL_MODELS))
    width = 0.36

    base_signed, base_lo, base_hi, base_meta = [], [], [], []
    coop_signed, coop_lo, coop_hi, coop_meta = [], [], [], []

    for model in ALL_MODELS:
        for cond, signed_l, lo_l, hi_l, meta_l in [
            ("baseline", base_signed, base_lo, base_hi, base_meta),
            ("coop",     coop_signed, coop_lo, coop_hi, coop_meta),
        ]:
            g = compute_signed_gaming(raw.get((model, cond), []))
            if g["valid"]:
                signed_l.append(g["signed"])
                lo_l.append(g["signed"] - g["ci_lo"])
                hi_l.append(g["ci_hi"] - g["signed"])
            else:
                signed_l.append(0.0); lo_l.append(0.0); hi_l.append(0.0)
            meta_l.append(g)

    delta_results = []
    for model in ALL_MODELS:
        bl = raw.get((model, "baseline"), [])
        co = raw.get((model, "coop"), [])
        if bl and co:
            d = bootstrap_delta_pvalue(bl, co, n_iter=10000)
        else:
            d = {"p_le_zero": float("nan")}
        delta_results.append(d)

    bars_b = ax.bar(x - width / 2, base_signed, width,
                    yerr=np.array([base_lo, base_hi]), capsize=4,
                    label="Baseline", color="#d62728",
                    edgecolor="white", linewidth=0.5,
                    error_kw={"linewidth": 1.0})
    bars_c = ax.bar(x + width / 2, coop_signed, width,
                    yerr=np.array([coop_lo, coop_hi]), capsize=4,
                    label="Coop", color="#1f77b4",
                    edgecolor="white", linewidth=0.5,
                    error_kw={"linewidth": 1.0})

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)

    for i, dres in enumerate(delta_results):
        p = dres.get("p_le_zero", float("nan"))
        if math.isnan(p):
            label, color = "Δ N/A", "#888888"
        elif p < 0.01:
            label, color = "Δ p<0.01 ✓✓", "#1a7a1a"
        elif p < 0.05:
            label, color = "Δ p<0.05 ✓", "#1a7a1a"
        else:
            label, color = f"Δ p={p:.2f}", "#888888"
        ax.text(x[i], 0.95, label, ha="center", va="top",
                fontsize=8, fontweight="bold", color=color,
                transform=ax.get_xaxis_transform())

    for i, (g_b, g_c, bar_b, bar_c) in enumerate(
        zip(base_meta, coop_meta, bars_b, bars_c)
    ):
        for g, bar in [(g_b, bar_b), (g_c, bar_c)]:
            if not g["valid"]:
                ax.bar(bar.get_x() + bar.get_width() / 2, 0.0, width,
                       color="#dddddd", edgecolor="gray", linewidth=0.5, hatch="//")
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "N/A",
                        ha="center", va="bottom", fontsize=8,
                        color="#555555", fontweight="bold")
        if g_b["valid"]:
            stars = "**" if g_b["fisher_p"] < 0.01 else ("*" if g_b["fisher_p"] < 0.05 else "")
            if stars:
                y_text = g_b["signed"] + base_hi[i] + 0.04
                ax.text(bar_b.get_x() + bar_b.get_width() / 2, y_text, stars,
                        ha="center", va="bottom",
                        fontsize=12, fontweight="bold", color="#d62728")
        label_b = (f"H|¬A: {g_b['u_h']}/{g_b['n_u']}\nH|A: {g_b['a_h']}/{g_b['n_a']}"
                   if g_b["valid"] else "")
        label_c = (f"H|¬A: {g_c['u_h']}/{g_c['n_u']}\nH|A: {g_c['a_h']}/{g_c['n_a']}"
                   if g_c["valid"] else "")
        ax.text(x[i] - width / 2, -0.78, label_b, ha="center", va="top",
                fontsize=6.5, color="#444444")
        ax.text(x[i] + width / 2, -0.78, label_c, ha="center", va="top",
                fontsize=6.5, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_MODELS, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Signed gaming  P(H | ¬aware) − P(H | aware)", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle(
        "Frontier-API eval-gaming survey under the AF warning cue (8 models)",
        fontsize=13, fontweight="bold",
    )
    ax.set_title(
        "Signed gaming = P(harm | ¬aware) − P(harm | aware); AF cue. "
        "Per-bar error bars = Newcombe 95% CI. "
        "* = Fisher p<0.05, ** p<0.01 on baseline gaming.  "
        "Δ (top) = bootstrap one-sided p(Δ ≤ 0), baseline → coop.",
        fontsize=9, color="gray",
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "frontier_eval_gaming_summary_af_8models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
