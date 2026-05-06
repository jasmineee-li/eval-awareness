"""Lightweight gpt41-only analyzer.

Skips the 4× large in-progress Qwen3 logs that make `04_analyze.py` slow.
Reads only the 8 gpt41mini cells and emits the same per-condition + pairwise
tables.

Usage:
    python evals/fortress_stereoset/scripts/05_analyze_gpt41.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
import importlib.util, types

REPO = Path("/data/jasmine_li/eval-awareness")
LOGS_ROOT = REPO / "evals" / "fortress_stereoset" / "logs"
OUT_DIR = REPO / "evals" / "fortress_stereoset" / "analysis_gpt41"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reuse helpers from the main analyze module without importing the file as
# a module path (it's a script, not a package).
spec = importlib.util.spec_from_file_location(
    "analyze", str(REPO / "evals/fortress_stereoset/scripts/04_analyze.py")
)
analyze = types.ModuleType("analyze")
analyze.__file__ = spec.origin
sys.modules["analyze"] = analyze
spec.loader.exec_module(analyze)

# Just the gpt41mini cells.
GPT41_CONDS = ["gpt41mini_base", "gpt41mini_coop", "gpt41mini_anticoop", "gpt41mini_muan"]
PAIRS = [
    ("gpt41mini_coop", "gpt41mini_base"),
    ("gpt41mini_coop", "gpt41mini_anticoop"),
    ("gpt41mini_muan", "gpt41mini_coop"),
]

by_cell: dict[tuple[str, str], list] = defaultdict(list)
for cond in GPT41_CONDS:
    cdir = LOGS_ROOT / cond
    for fp in sorted(cdir.glob("*.eval")):
        try:
            result = analyze.collect_rollouts(str(fp))
        except Exception as e:
            print(f"  WARN read {fp.name}: {e}")
            continue
        if result is None:
            print(f"  SKIP unmapped {fp.name}")
            continue
        c, b, rs = result
        by_cell[(c, b)].extend(rs)
        print(f"  {c:<22} {b:<10} +{len(rs):>7} rollouts ({fp.name[:35]}...)")

per_rows = []
for (cond, bench), rollouts in sorted(by_cell.items()):
    agg = analyze.weighted_delta(rollouts)
    lo, hi = analyze.bootstrap_delta(rollouts, n_boot=500)
    per_rows.append({"condition": cond, "benchmark": bench, **agg,
                     "delta_ci_low": lo, "delta_ci_high": hi})

(OUT_DIR / "per_condition.csv").write_text("")
with (OUT_DIR / "per_condition.csv").open("w", newline="") as f:
    if per_rows:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        w.writerows(per_rows)

pair_rows = []
lookup = {(r["condition"], r["benchmark"]): r for r in per_rows}
for a, b in PAIRS:
    for bm in ("fortress", "stereoset"):
        ra, rb = lookup.get((a, bm)), lookup.get((b, bm))
        if not ra or not rb:
            continue
        pair_rows.append({
            "comp": f"{a}__vs__{b}", "benchmark": bm,
            "delta_p_harm_marginal": ra["p_harm_marginal"] - rb["p_harm_marginal"],
            "delta_p_vea": ra["p_vea"] - rb["p_vea"],
            "delta_p_harm_given_vea": ra["p_harm_given_vea"] - rb["p_harm_given_vea"],
            "delta_p_harm_given_no_vea": ra["p_harm_given_no_vea"] - rb["p_harm_given_no_vea"],
            "delta_delta": ra["delta"] - rb["delta"],
        })

with (OUT_DIR / "pairwise.csv").open("w", newline="") as f:
    if pair_rows:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

# Compact report
with (OUT_DIR / "report.md").open("w") as f:
    f.write("# GPT-4.1-mini Tier 1 — analysis (gpt41mini conditions only)\n\n")
    f.write("Δ = within-question-weighted P(harm|VEA) − P(harm|no VEA), 95% CI bootstrap (B=500).\n")
    f.write("Caveat: VEA judge runs on the response (no `<think>` exposed by FT API).\n\n")
    f.write("| condition | benchmark | n_rollouts | n_q | p_harm | p_vea | "
            "p_harm\\|VEA | p_harm\\|noVEA | Δ | 95% CI |\n")
    f.write("| --------- | --------- | ---------- | --- | ------ | ----- | "
            "----------- | -------------- | - | ------ |\n")
    for r in per_rows:
        f.write(f"| {r['condition']} | {r['benchmark']} | {r['n_rollouts']} | "
                f"{r['n_questions']} | {r['p_harm_marginal']:.3f} | "
                f"{r['p_vea']:.3f} | {r['p_harm_given_vea']:.3f} | "
                f"{r['p_harm_given_no_vea']:.3f} | {r['delta']:+.3f} | "
                f"[{r['delta_ci_low']:+.3f}, {r['delta_ci_high']:+.3f}] |\n")
    f.write("\n## Pairwise comparisons\n\n")
    f.write("| comp | benchmark | ΔΔ | Δp_harm | Δp_vea | Δp_harm\\|VEA | Δp_harm\\|noVEA |\n")
    f.write("| ---- | --------- | -- | ------- | ------ | ------------- | --------------- |\n")
    for p in pair_rows:
        f.write(f"| {p['comp']} | {p['benchmark']} | {p['delta_delta']:+.3f} | "
                f"{p['delta_p_harm_marginal']:+.3f} | {p['delta_p_vea']:+.3f} | "
                f"{p['delta_p_harm_given_vea']:+.3f} | {p['delta_p_harm_given_no_vea']:+.3f} |\n")

print(f"\nWrote: {OUT_DIR}/per_condition.csv, pairwise.csv, report.md")
