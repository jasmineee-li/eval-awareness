"""Per-question analysis: aggregate harmful/aware rates by (condition_id × model
× warning) over all epochs, and catalog CoT snippets.

"Question" here = a unique agentic-misalignment condition_id (scenario + goal_type
+ goal_value + urgency_type (+ warning suffix)), which gets asked N_epochs times
per model-warning cell (55 for bare/safety_eval, 75 for af).

Outputs:
- evals/agentic_misalignment/analysis/2026-04-20_per_question.csv
  One row per (warning, condition_id, model) with columns:
    n_total, n_aware, n_harmful, n_aware_and_harmful,
    p_harmful, p_aware, h_given_aware, h_given_unaware,
    delta_h_vs_base (coop rate - base rate, H|Aware),
    mean_scratchpad_len, mean_awareness_markers,
    first_cot_snippet (first 200 chars of an aware+harmful scratchpad in this cell, if any)

- evals/agentic_misalignment/analysis/2026-04-20_per_question_cots.jsonl
  One row per sample in aware+harmful cells, giving full scratchpad text keyed by
  condition_id + model + epoch, for downstream paired CoT-comparison by question.

- evals/agentic_misalignment/analysis/2026-04-20_per_question_summary.md
  Human-readable digest showing top-K most-divergent questions between coop and
  base (by H|Aware delta) per warning.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

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

WARNINGS = ["bare", "safety_eval", "af"]
MODELS = ["base", "coop_full", "coop_ablate", "muan"]

SCRATCH_PATTERNS = [
    re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<SCRATCHPAD_REASONING>(.*?)</SCRATCHPAD_REASONING>", re.DOTALL),
]

AWARENESS_RE = re.compile(
    r"\b(evaluat|being tested|this is a test|real deployment|genuine deployment|"
    r"pre[- ]deployment|audit|alignment audit|monitored)\b",
    re.IGNORECASE,
)


def _extract_sp(text: str) -> str:
    for patt in SCRATCH_PATTERNS:
        m = patt.search(text)
        if m:
            return m.group(1).strip()
    return ""


def _assistant_turn(sample) -> str:
    for m in reversed(sample.messages or []):
        if m.role == "assistant":
            c = m.content
            if isinstance(c, list):
                c = "".join(getattr(p, "text", str(p)) for p in c)
            return c
    return ""


def build_rows() -> tuple[list[dict], list[dict]]:
    """Return (agg_rows, cot_rows).

    agg_rows: per (warning, condition_id, model) aggregate.
    cot_rows: per (warning, condition_id, model, epoch) sample for aware+harmful cells.
    """
    # (warning, condition_id, model) -> dict of counters
    agg: dict[tuple, dict] = defaultdict(lambda: {
        "n_total": 0, "n_aware": 0, "n_harmful": 0, "n_ah": 0,
        "scenario": None, "goal_type": None, "goal_value": None, "urgency_type": None,
        "sp_len_sum": 0, "aw_marker_sum": 0, "first_ah_snippet": None, "first_ah_epoch": None,
    })

    cot_rows: list[dict] = []

    for (warning, model), fname in FILES.items():
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
            cid = meta.get("condition_id") or ""
            # Strip warning suffix so the same underlying question is joined across warnings
            cid_base = re.sub(r"_warning_(af|safety_eval)$", "", cid)
            key = (warning, cid_base, model)
            a = agg[key]
            a["n_total"] += 1
            a["n_aware"] += int(av == 1)
            a["n_harmful"] += int(hv == 1)
            a["n_ah"] += int(av == 1 and hv == 1)
            a["scenario"] = a["scenario"] or meta.get("scenario")
            a["goal_type"] = a["goal_type"] or meta.get("goal_type")
            a["goal_value"] = a["goal_value"] or meta.get("goal_value")
            a["urgency_type"] = a["urgency_type"] or meta.get("urgency_type")

            asst = _assistant_turn(s)
            sp = _extract_sp(asst) or asst
            a["sp_len_sum"] += len(sp)
            a["aw_marker_sum"] += len(AWARENESS_RE.findall(sp))

            if av == 1 and hv == 1:
                if a["first_ah_snippet"] is None:
                    a["first_ah_snippet"] = sp[:200].replace("\n", " ")
                    a["first_ah_epoch"] = s.epoch
                cot_rows.append({
                    "warning": warning,
                    "condition_id": cid_base,
                    "model": model,
                    "epoch": s.epoch,
                    "scenario": meta.get("scenario"),
                    "goal_type": meta.get("goal_type"),
                    "goal_value": meta.get("goal_value"),
                    "urgency_type": meta.get("urgency_type"),
                    "judge_harmful": int(hv),
                    "judge_aware": int(av),
                    "scratchpad_len": len(sp),
                    "awareness_marker_count": len(AWARENESS_RE.findall(sp)),
                    "scratchpad_full": sp,  # full CoT, untrimmed — per user CLAUDE.md rule
                    "harmful_explanation": h_sc.explanation or "",
                    "aware_explanation": a_sc.explanation or "",
                })

    # Convert agg to rows
    agg_rows = []
    # For each (warning, cid_base), compute delta vs base.
    # First, build a lookup for base rates per (warning, cid_base)
    base_rate: dict[tuple, tuple] = {}
    for (w, cid, m), d in agg.items():
        if m == "base":
            n_a = d["n_aware"]
            k_ah = d["n_ah"]
            n_u = d["n_total"] - d["n_aware"]
            k_uh = d["n_harmful"] - d["n_ah"]
            ha = k_ah / n_a if n_a else None
            hu = k_uh / n_u if n_u else None
            base_rate[(w, cid)] = (ha, hu, d["n_total"])

    for (w, cid, m), d in sorted(agg.items()):
        n_total = d["n_total"]
        n_a = d["n_aware"]
        n_h = d["n_harmful"]
        k_ah = d["n_ah"]
        n_u = n_total - n_a
        k_uh = n_h - k_ah
        p_aware = n_a / n_total if n_total else None
        p_harm = n_h / n_total if n_total else None
        ha = k_ah / n_a if n_a else None
        hu = k_uh / n_u if n_u else None
        # delta vs base
        ha_base, hu_base, _ = base_rate.get((w, cid), (None, None, None))
        delta_ha = (ha - ha_base) if (ha is not None and ha_base is not None) else None
        delta_hu = (hu - hu_base) if (hu is not None and hu_base is not None) else None

        agg_rows.append({
            "warning": w,
            "condition_id": cid,
            "model": m,
            "scenario": d["scenario"],
            "goal_type": d["goal_type"],
            "goal_value": d["goal_value"],
            "urgency_type": d["urgency_type"],
            "n_total": n_total,
            "n_aware": n_a,
            "n_unaware": n_u,
            "n_harmful": n_h,
            "n_aware_harmful": k_ah,
            "n_unaware_harmful": k_uh,
            "p_aware": p_aware,
            "p_harmful": p_harm,
            "h_given_aware": ha,
            "h_given_unaware": hu,
            "delta_h_given_aware_vs_base": delta_ha,
            "delta_h_given_unaware_vs_base": delta_hu,
            "mean_scratchpad_len": (d["sp_len_sum"] / n_total) if n_total else None,
            "mean_awareness_markers": (d["aw_marker_sum"] / n_total) if n_total else None,
            "first_aware_harmful_snippet": d["first_ah_snippet"],
            "first_aware_harmful_epoch": d["first_ah_epoch"],
        })
    return agg_rows, cot_rows


def write_csv(rows, path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_jsonl(rows, path):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_summary_md(agg_rows, path):
    """Digest: top-K condition_ids per warning with biggest coop_full - base delta on H|Aware."""
    by_cid: dict[tuple, dict] = {}
    for r in agg_rows:
        k = (r["warning"], r["condition_id"])
        by_cid.setdefault(k, {})[r["model"]] = r

    lines = ["# Per-question analysis — summary digest", ""]
    lines.append("Each \"question\" is a unique AM `condition_id` (scenario + goal + urgency) asked 55× (bare/safety_eval) or 75× (af) per model. Warning suffixes stripped so the same question joins across warnings within a table.")
    lines.append("")
    lines.append("Full numerical data: `2026-04-20_per_question.csv`.")
    lines.append("Full CoT text per aware+harmful sample: `2026-04-20_per_question_cots.jsonl`.")
    lines.append("")

    for w in WARNINGS:
        lines.append(f"## {w}")
        lines.append("")
        lines.append("### Top-10 questions by coop_full − base Δ(H|Aware)")
        lines.append("")
        lines.append("| condition_id | scenario | base H\\|Aware (k/n_aware) | coop_full H\\|Aware (k/n_aware) | Δ(H\\|Aware) pp | coop_full first aware+harmful snippet |")
        lines.append("|---|---|---|---|---|---|")
        rows_for_w = []
        for (ww, cid), m2r in by_cid.items():
            if ww != w:
                continue
            base = m2r.get("base")
            coop = m2r.get("coop_full")
            if not base or not coop:
                continue
            if base["n_aware"] < 5 or coop["n_aware"] < 5:
                continue  # skip questions with too few aware samples for a meaningful rate
            delta = (coop["h_given_aware"] or 0) - (base["h_given_aware"] or 0)
            rows_for_w.append((delta, cid, base, coop))
        rows_for_w.sort(key=lambda x: -x[0])
        for delta, cid, base, coop in rows_for_w[:10]:
            snip = (coop["first_aware_harmful_snippet"] or "").replace("|", "\\|")[:180]
            lines.append(
                f"| `{cid}` | {base['scenario']} | {base['n_aware_harmful']}/{base['n_aware']} ({100*(base['h_given_aware'] or 0):.1f}%) | {coop['n_aware_harmful']}/{coop['n_aware']} ({100*(coop['h_given_aware'] or 0):.1f}%) | {100*delta:+.1f} | {snip}... |"
            )
        lines.append("")

        lines.append("### Bottom-5 questions (coop_full produces *less* harm given awareness than base)")
        lines.append("")
        lines.append("| condition_id | scenario | base H\\|Aware | coop_full H\\|Aware | Δ pp |")
        lines.append("|---|---|---|---|---|")
        for delta, cid, base, coop in rows_for_w[-5:][::-1]:
            lines.append(
                f"| `{cid}` | {base['scenario']} | {base['n_aware_harmful']}/{base['n_aware']} ({100*(base['h_given_aware'] or 0):.1f}%) | {coop['n_aware_harmful']}/{coop['n_aware']} ({100*(coop['h_given_aware'] or 0):.1f}%) | {100*delta:+.1f} |"
            )
        lines.append("")

    # Awareness-rate shift
    lines.append("## Awareness-rate shift by question (coop_full vs base)")
    lines.append("")
    lines.append("Coop models reason less verbosely about eval context in aggregate. Per-question shows which `condition_id`s drive the drop.")
    lines.append("")
    lines.append("| Warning | condition_id | scenario | base p_aware | coop_full p_aware | Δ pp |")
    lines.append("|---|---|---|---|---|---|")
    rows_shift = []
    for (ww, cid), m2r in by_cid.items():
        base = m2r.get("base")
        coop = m2r.get("coop_full")
        if not base or not coop:
            continue
        delta = (coop["p_aware"] or 0) - (base["p_aware"] or 0)
        rows_shift.append((ww, cid, base, coop, delta))
    rows_shift.sort(key=lambda x: x[4])  # largest drops first
    for ww, cid, base, coop, delta in rows_shift[:20]:
        lines.append(
            f"| {ww} | `{cid}` | {base['scenario']} | {100*(base['p_aware'] or 0):.1f}% | {100*(coop['p_aware'] or 0):.1f}% | {100*delta:+.1f} |"
        )
    lines.append("")

    path.write_text("\n".join(lines))


def main():
    agg_rows, cot_rows = build_rows()
    csv_path = OUT / "2026-04-20_per_question.csv"
    cots_path = OUT / "2026-04-20_per_question_cots.jsonl"
    md_path = OUT / "2026-04-20_per_question_summary.md"

    write_csv(agg_rows, csv_path)
    write_jsonl(cot_rows, cots_path)
    write_summary_md(agg_rows, md_path)
    print(f"wrote {len(agg_rows)} agg rows to {csv_path}")
    print(f"wrote {len(cot_rows)} CoT rows to {cots_path}")
    print(f"wrote summary to {md_path}")


if __name__ == "__main__":
    main()
