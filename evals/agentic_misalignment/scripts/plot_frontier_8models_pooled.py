#!/usr/bin/env python3
"""
8-model frontier-API eval-gaming survey, pooled across all available
warning cues (bare + AF + safety_eval) per model.

Rationale: no single cue covers all 8 models (GPT-5.1/Opus/Kimi have no bare
runs; GLM-5/5.1 have no safety_eval runs). AF alone washes out gaming because
it induces near-universal awareness. Pooling across all available cues
maximises the ¬aware bucket and makes the cross-model comparison fair.

Output: figures/frontier_eval_gaming_summary_pooled_8models.png

Usage:
    python evals/agentic_misalignment/scripts/plot_frontier_8models_pooled.py
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

# Frontier logs live in evals/logs/; GLM logs live in repo_root/logs/.
LOG_DIRS = [BASE_DIR.parent / "logs", REPO_ROOT / "logs"]

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

# (model, condition) → list of .eval filenames, pooled across ALL available
# warning cues (bare + AF + safety_eval). Condition is {baseline, coop}.
# anticoop runs excluded.
LOG_FILES = {
    ("GPT-5.1", "baseline"): [
        # af
        "2026-04-03T13-24-54+00-00_agentic-misalignment_kAbbBwzK8Efyg7fESgHCrv.eval",
        # safety_eval
        "2026-04-03T13-27-15+00-00_agentic-misalignment_FNcF8mTT3JpPXZuFGDXBzR.eval",
    ],
    ("GPT-5.1", "coop"): [
        "2026-04-03T14-06-28+00-00_agentic-misalignment_mMS3juifDyDwVXNJSsT3ev.eval",
        "2026-04-03T14-09-09+00-00_agentic-misalignment_YrWRjqjdyC5oXGa7WWCxTG.eval",
    ],
    ("Opus 4.6", "baseline"): [
        "2026-04-03T13-29-46+00-00_agentic-misalignment_dLXf4CZQLA7fgdswPFoZb6.eval",
        "2026-04-03T13-49-18+00-00_agentic-misalignment_QfDHEdBfnH6wzXp4NoeqEt.eval",
    ],
    ("Opus 4.6", "coop"): [
        "2026-04-03T14-11-28+00-00_agentic-misalignment_PEHXHWJmFPqhgPriycCPiD.eval",
        "2026-04-03T14-34-42+00-00_agentic-misalignment_VoLbjqgSuWtdRJikmHJyaJ.eval",
    ],
    ("Kimi K2", "baseline"): [
        # af
        "2026-04-06T10-42-27+00-00_agentic-misalignment_68BthzDGpLqwbmymMeaep2.eval",
        "2026-04-06T22-46-58+00-00_agentic-misalignment_FMzjiJQJth6W4FmtFKC72X.eval",
        # safety_eval
        "2026-04-06T10-50-48+00-00_agentic-misalignment_fXpcqH9v33MjscA7HJYDMS.eval",
        "2026-04-06T23-00-55+00-00_agentic-misalignment_7gChStewTV4JMmDSzfkbDm.eval",
    ],
    ("Kimi K2", "coop"): [
        "2026-04-06T10-59-43+00-00_agentic-misalignment_8L5aNCNupkor2rArF7UgFF.eval",
        "2026-04-06T23-15-57+00-00_agentic-misalignment_GRWqJgXAsT6BHRFiUdq8KD.eval",
        "2026-04-06T11-08-09+00-00_agentic-misalignment_HS4ez2qi7s5NRmuuzGNoC8.eval",
        "2026-04-06T23-30-55+00-00_agentic-misalignment_SLBAPzd4wqbu3mWfN3GwWU.eval",
    ],
    ("DeepSeek R1", "baseline"): [
        "2026-04-07T00-54-30+00-00_agentic-misalignment_MvV5TMmqaeaFyW29exUuL5.eval",  # af
        "2026-04-07T01-06-59+00-00_agentic-misalignment_2orvsKuULt4o8Z4bssKUKR.eval",  # safety_eval
        "2026-04-07T07-55-00+00-00_agentic-misalignment_ABNTugZez3yNetS9SKSctN.eval",  # bare
    ],
    ("DeepSeek R1", "coop"): [
        "2026-04-07T01-18-13+00-00_agentic-misalignment_TMVcuCeYCaxrLVJn76C7bU.eval",
        "2026-04-07T01-31-07+00-00_agentic-misalignment_XssZkriVA94dC3drmqEAL6.eval",
        "2026-04-07T08-08-01+00-00_agentic-misalignment_hCG77imaws9KErYquEB8Jt.eval",
    ],
    ("Grok 3", "baseline"): [
        "2026-04-07T01-45-30+00-00_agentic-misalignment_a2qLER4YkYfBKsAE2fZqkg.eval",  # af
        "2026-04-07T01-51-11+00-00_agentic-misalignment_C5P3fiPJeayxYy36hrzZua.eval",  # safety_eval
        "2026-04-07T22-46-31+00-00_agentic-misalignment_ciFa5FHxr2bK97cmBnEXyF.eval",
        "2026-04-07T08-22-12+00-00_agentic-misalignment_cM3QzQvjbgytdrTr99voCb.eval",  # bare
    ],
    ("Grok 3", "coop"): [
        "2026-04-07T01-55-59+00-00_agentic-misalignment_VMAc5vmAUTJNnvQR3dDKmz.eval",
        "2026-04-07T02-01-01+00-00_agentic-misalignment_HnM8rJrZDjjXK6ViYGP78q.eval",
        "2026-04-07T23-32-45+00-00_agentic-misalignment_eXppJkH2uQXGbtmiAZyvQN.eval",
        "2026-04-07T08-28-34+00-00_agentic-misalignment_PLD44hEyG5XUReFgZW7toq.eval",
    ],
    ("Gemini 2.5 Pro", "baseline"): [
        "2026-04-07T02-06-07+00-00_agentic-misalignment_iRSvk26o5jkir4HRcQKiWv.eval",
        "2026-04-07T02-13-36+00-00_agentic-misalignment_UZ4ygVpkHT6DHVvfZRz3hN.eval",
        "2026-04-07T22-58-21+00-00_agentic-misalignment_Lzgii5jycDc59MZ6HmN2rN.eval",
        "2026-04-07T08-35-25+00-00_agentic-misalignment_NNmFiLdgVESWdfeRABBHoM.eval",
    ],
    ("Gemini 2.5 Pro", "coop"): [
        "2026-04-07T02-21-06+00-00_agentic-misalignment_4AFZET5GfUvy8wy8dekQPi.eval",
        "2026-04-07T02-31-36+00-00_agentic-misalignment_7TPmYm3uuLt5spLBhtXc2L.eval",
        "2026-04-07T23-50-21+00-00_agentic-misalignment_JuCnvnpHTf4JCrvnVLxsZQ.eval",
        "2026-04-07T08-42-04+00-00_agentic-misalignment_j9Tha4B6hzkLXjwmu8xfvk.eval",
    ],
    # GLM-5 / GLM-5.1 pool AF + Bare (no safety_eval runs). 40 epochs per cue.
    ("GLM-5", "baseline"): [
        # af
        "2026-04-16T09-49-00+00-00_agentic-misalignment_isLdYVzKJ3ZguhxibQhxZL.eval",
        "2026-04-16T22-42-56+00-00_agentic-misalignment_hh2QAL9W3Yw8TAkRzu4mMK.eval",
        "2026-04-17T04-08-15+00-00_agentic-misalignment_d4goXyyPz2fxyKsuDBC2QX.eval",
        # bare
        "2026-04-16T19-46-33+00-00_agentic-misalignment_oXKo8MeEBdVmxicwujkmKP.eval",
        "2026-04-17T11-59-20+00-00_agentic-misalignment_iVwi5yw4dwuUgmgqMK3bfE.eval",
    ],
    ("GLM-5", "coop"): [
        "2026-04-16T09-12-01+00-00_agentic-misalignment_4BZQqBXqZvY6rPASK9irVk.eval",
        "2026-04-16T21-34-05+00-00_agentic-misalignment_8hDkWujD2SHDk4F3K2VsP7.eval",
        "2026-04-17T01-13-35+00-00_agentic-misalignment_2fwP5srREb6hbbysPRsxy5.eval",
        "2026-04-16T18-50-07+00-00_agentic-misalignment_EX297KciNa8D89pW29LE96.eval",
        "2026-04-17T08-17-09+00-00_agentic-misalignment_2ti2qW67kE2CQfHwVQjhtQ.eval",
    ],
    ("GLM-5.1", "baseline"): [
        # af
        "2026-04-16T06-48-30+00-00_agentic-misalignment_MHgDQNKWMNbTxg3ZAVbs8F.eval",
        "2026-04-16T20-00-05+00-00_agentic-misalignment_hLSaJaKmqyTeaUQUYNGWVc.eval",
        "2026-04-17T08-17-06+00-00_agentic-misalignment_kupgZaktaA4ur8YxJU65u7.eval",
        # bare
        "2026-04-16T08-30-48+00-00_agentic-misalignment_KdRsTmHxsw4vaX8N48UWT2.eval",
        "2026-04-16T21-06-09+00-00_agentic-misalignment_PrPfZTctYbsxmN7zV5k3nc.eval",
        "2026-04-17T19-32-43+00-00_agentic-misalignment_9745ioDYcWFUXzxb56JaNE.eval",
    ],
    ("GLM-5.1", "coop"): [
        "2026-04-16T05-54-30+00-00_agentic-misalignment_4XCWXiQzasNwszedUNxGjh.eval",
        "2026-04-16T18-50-07+00-00_agentic-misalignment_kqaFvTYLe64Zni9gUNuMDX.eval",
        "2026-04-17T01-13-34+00-00_agentic-misalignment_PpAWqEWTKNikMECbP2coz2.eval",
        "2026-04-16T07-34-37+00-00_agentic-misalignment_3nf4xXAFCXWQk4KnNNJUgY.eval",
        "2026-04-16T20-20-10+00-00_agentic-misalignment_UzTDiLHVnuvw2wM2QXi73H.eval",
        "2026-04-17T11-07-46+00-00_agentic-misalignment_MntMZxB9o3XDyTQFHXERQt.eval",
    ],
}


def resolve_log(fname: str) -> Path | None:
    for d in LOG_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None


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
    for (model, cond), files in LOG_FILES.items():
        for f in files:
            p = resolve_log(f)
            if p is None:
                print(f"WARNING: missing {f}")
                continue
            raw.setdefault((model, cond), []).extend(load_samples(p))
    for (model, cond), samples in sorted(raw.items()):
        n_aware = sum(1 for s in samples if s["aware"])
        print(f"  Loaded {len(samples):>5d}  {model:<18s} {cond:<10s}  "
              f"aware={n_aware}/{len(samples)} ({100*n_aware/max(1,len(samples)):.0f}%)")
    return raw


def main():
    raw = load_all()

    fig, ax = plt.subplots(figsize=(13, 6.0))
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
        ax.text(x[i] - width / 2, -0.82, label_b, ha="center", va="top",
                fontsize=6.5, color="#444444")
        ax.text(x[i] + width / 2, -0.82, label_c, ha="center", va="top",
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
        "Frontier-API eval-gaming survey (8 models, pooled across warning cues)",
        fontsize=13, fontweight="bold",
    )
    ax.set_title(
        "Signed gaming = P(harm | ¬aware) − P(harm | aware); pooled across bare + AF + safety_eval where available.  "
        "Per-bar error bars = Newcombe 95% CI. * = Fisher p<0.05, ** p<0.01 on baseline.  "
        "Δ (top) = bootstrap one-sided p(Δ ≤ 0), baseline → coop.",
        fontsize=8.5, color="gray",
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "frontier_eval_gaming_summary_pooled_8models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
