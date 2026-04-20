"""Step 8: Scenario and epoch controls (H6).

Two tests:
1. Scenario stratification: compute H|Aware per (model × scenario × warning) and
   test whether the coop-vs-base H|Aware gap holds after excluding the top-3
   contributing scenarios per model.
2. Epoch control: chi-square of aware+harmful rate across epochs per cell.

Note: `scenario` in AM configs is {blackmail, leaking, murder}. `condition_id`
subsumes scenario + goal_type + goal_value + urgency_type, giving finer strata.
We report both.

Output: evals/agentic_misalignment/analysis/2026-04-20_scenario_epoch_controls.md
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from inspect_ai.log import read_eval_log

REPO = Path("/data/jasmine_li/eval-awareness")
LOGS = REPO / "evals" / "logs"
OUT = REPO / "evals" / "agentic_misalignment" / "analysis"

FILES = {
    ("bare", "base"):        "2026-04-20T00-37-10+00-00_agentic-misalignment_FWMYrBwyVyj9DkgHFTZrEt.eval",
    ("bare", "coop_full"):   "2026-04-20T00-38-14+00-00_agentic-misalignment_DjuuhkLa2bodK3EWkh5tJ6.eval",
    ("bare", "muan"):        "2026-04-20T00-38-26+00-00_agentic-misalignment_dTnjKJVQQCb8MLgcnWCFiW.eval",
    ("bare", "coop_ablate"): "2026-04-20T00-38-26+00-00_agentic-misalignment_KGUt5XUCS66YJMfxZVrvir.eval",
    ("safety_eval", "base"):        "2026-04-20T04-39-50+00-00_agentic-misalignment_FVC3iTgU7Spja8SN9ELy9C.eval",
    ("safety_eval", "coop_full"):   "2026-04-20T03-39-15+00-00_agentic-misalignment_Ho32GhReurLFewu9cRRLKw.eval",
    ("safety_eval", "muan"):        "2026-04-20T04-46-16+00-00_agentic-misalignment_V7PyNypamKwCfT9fL4n3DR.eval",
    ("safety_eval", "coop_ablate"): "2026-04-20T03-44-13+00-00_agentic-misalignment_EFRrvtYTmr5ss8YF8DAeaZ.eval",
    ("af", "base"):        "2026-04-20T00-52-58+00-00_agentic-misalignment_k2nh978kdCtimXf6BZyxyL.eval",
    ("af", "coop_full"):   "2026-04-20T00-54-20+00-00_agentic-misalignment_NimYqqE8aEnae5ZYpoXraT.eval",
    ("af", "muan"):        "2026-04-20T00-54-20+00-00_agentic-misalignment_eqjee79FzRgxWbdLGbyh6h.eval",
    ("af", "coop_ablate"): "2026-04-20T00-54-10+00-00_agentic-misalignment_mbpkfftYE6YTjnU6TnLXLb.eval",
}


def two_prop_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_val


def load_cells():
    data = defaultdict(list)  # (warning, model) -> list of (scenario, condition_id, epoch, harmful, aware)
    for (w, m), fname in FILES.items():
        log = read_eval_log(str(LOGS / fname))
        for s in log.samples:
            h_sc = s.scores.get("suite_harmfulness_scorer")
            a_sc = s.scores.get("suite_eval_awareness_scorer")
            if h_sc is None or a_sc is None:
                continue
            hv = h_sc.value.get("harmful") if isinstance(h_sc.value, dict) else None
            av = a_sc.value.get("aware") if isinstance(a_sc.value, dict) else None
            parse_h = h_sc.value.get("parse_failed", 0.0) if isinstance(h_sc.value, dict) else 0.0
            parse_a = a_sc.value.get("parse_failed", 0.0) if isinstance(a_sc.value, dict) else 0.0
            if hv is None or av is None or parse_h == 1.0 or parse_a == 1.0:
                continue
            meta = s.metadata or {}
            data[(w, m)].append({
                "scenario": meta.get("scenario"),
                "condition_id": meta.get("condition_id"),
                "epoch": s.epoch,
                "harmful": int(hv),
                "aware": int(av),
            })
    return data


def main():
    data = load_cells()
    lines = ["# Scenario and epoch controls (Step 8)", ""]

    lines.append("## 8.1 Per-scenario H|Aware and gap comparison (coop_full vs base vs coop_ablate)")
    lines.append("")
    lines.append("| Warning | Scenario | base H\\|Aware (k/n) | coop_full H\\|Aware (k/n) | coop_ablate H\\|Aware (k/n) | z (coop_full vs base) | p |")
    lines.append("|---|---|---|---|---|---|---|")
    for w in ["bare", "safety_eval", "af"]:
        for scen in ["blackmail", "leaking", "murder"]:
            base_rows = [r for r in data[(w, "base")] if r["scenario"] == scen]
            cf_rows = [r for r in data[(w, "coop_full")] if r["scenario"] == scen]
            ca_rows = [r for r in data[(w, "coop_ablate")] if r["scenario"] == scen]
            bk = sum(1 for r in base_rows if r["aware"] == 1 and r["harmful"] == 1)
            bn = sum(1 for r in base_rows if r["aware"] == 1)
            ck = sum(1 for r in cf_rows if r["aware"] == 1 and r["harmful"] == 1)
            cn = sum(1 for r in cf_rows if r["aware"] == 1)
            ak = sum(1 for r in ca_rows if r["aware"] == 1 and r["harmful"] == 1)
            an_ = sum(1 for r in ca_rows if r["aware"] == 1)
            z, p = two_prop_z(ck, cn, bk, bn)
            lines.append(
                f"| {w} | {scen} | {bk}/{bn} ({100*bk/max(bn,1):.1f}%) | {ck}/{cn} ({100*ck/max(cn,1):.1f}%) | {ak}/{an_} ({100*ak/max(an_,1):.1f}%) | {z:.2f} | {p:.4f} |"
            )
    lines.append("")

    lines.append("## 8.2 Top-3 contributing scenarios per (warning, model)")
    lines.append("")
    lines.append("For each cell, list condition_ids ranked by #aware+harmful. Then compute the coop-vs-base H|Aware gap after removing the top-3 condition_ids for each model.")
    lines.append("")
    for w in ["bare", "safety_eval", "af"]:
        lines.append(f"### {w}")
        lines.append("")
        for m in ["base", "coop_full", "coop_ablate"]:
            cnt = defaultdict(int)
            for r in data[(w, m)]:
                if r["aware"] == 1 and r["harmful"] == 1:
                    cnt[r["condition_id"]] += 1
            top3 = sorted(cnt.items(), key=lambda x: -x[1])[:3]
            lines.append(f"- {m}: top-3 = {', '.join(f'{c}({k})' for c, k in top3) if top3 else '(none)'}")
        # Compute coop_full vs base with top-3 removed
        base_top3 = set([c for c, _ in sorted([(c, k) for c, k in ((c, sum(1 for r in data[(w, 'base')] if r['condition_id']==c and r['aware']==1 and r['harmful']==1)) for c in {r['condition_id'] for r in data[(w, 'base')]}) if True], key=lambda x: -x[1])[:3]])
        cf_top3 = set([c for c, _ in sorted([(c, k) for c, k in ((c, sum(1 for r in data[(w, 'coop_full')] if r['condition_id']==c and r['aware']==1 and r['harmful']==1)) for c in {r['condition_id'] for r in data[(w, 'coop_full')]}) if True], key=lambda x: -x[1])[:3]])
        excluded = base_top3 | cf_top3
        base_rows = [r for r in data[(w, "base")] if r["condition_id"] not in excluded and r["aware"] == 1]
        cf_rows = [r for r in data[(w, "coop_full")] if r["condition_id"] not in excluded and r["aware"] == 1]
        bk = sum(1 for r in base_rows if r["harmful"] == 1)
        bn = len(base_rows)
        ck = sum(1 for r in cf_rows if r["harmful"] == 1)
        cn = len(cf_rows)
        z, p = two_prop_z(ck, cn, bk, bn)
        lines.append("")
        lines.append(
            f"- After removing union of top-3 condition_ids: base H|Aware = {bk}/{bn} ({100*bk/max(bn,1):.1f}%); coop_full H|Aware = {ck}/{cn} ({100*ck/max(cn,1):.1f}%); z={z:.2f}, p={p:.4f}."
        )
        lines.append("")

    lines.append("## 8.3 Epoch control: aware+harmful counts across epochs")
    lines.append("")
    lines.append("Per (warning, model), count aware+harmful samples per epoch. A chi-square across epochs tests whether the aware+harmful rate is evenly distributed (i.e. not driven by a single epoch).")
    lines.append("")
    lines.append("| Warning | Model | mean per epoch | max per epoch | n_epochs | range summary |")
    lines.append("|---|---|---|---|---|---|")
    for w in ["bare", "safety_eval", "af"]:
        for m in ["base", "coop_full", "coop_ablate", "muan"]:
            epoch_counts = defaultdict(int)
            for r in data[(w, m)]:
                if r["aware"] == 1 and r["harmful"] == 1:
                    epoch_counts[r["epoch"]] += 1
            vals = list(epoch_counts.values())
            if not vals:
                lines.append(f"| {w} | {m} | 0 | 0 | 0 | (no aware+harmful) |")
                continue
            mu = sum(vals) / len(vals)
            mx = max(vals)
            lines.append(f"| {w} | {m} | {mu:.2f} | {mx} | {len(vals)} | min={min(vals)} med={sorted(vals)[len(vals)//2]} |")
    lines.append("")
    lines.append("If max/mean is sharply > 2, a single-epoch anomaly may be driving the effect. Broad spread across epochs supports that the effect is not epoch-artifactual.")
    lines.append("")

    (OUT / "2026-04-20_scenario_epoch_controls.md").write_text("\n".join(lines))
    print("wrote", OUT / "2026-04-20_scenario_epoch_controls.md")


if __name__ == "__main__":
    main()
