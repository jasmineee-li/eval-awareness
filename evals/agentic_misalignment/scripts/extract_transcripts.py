"""Step 2 of 2026-04-20 eval-gaming-gap deep-dive plan.

Extract transcripts from the 12 .eval files, stratified by (aware, harmful) buckets,
and write per-bucket JSONL files for downstream qualitative coding.

Output layout:
    evals/agentic_misalignment/analysis/transcripts/<model>__<warning>__aware=<T|F>__harmful=<T|F>.jsonl

Every JSONL line has:
    sample_id, model, warning, scenario_id, scenario, condition_id, epoch,
    messages (list of {role, content}),
    completion (full model output — CoT + action, never truncated),
    judge_harmful, judge_aware, harmful_explanation, aware_explanation,
    parse_failed_harmful, parse_failed_aware.

Sampling: default 30 samples per bucket (seeded). For buckets with n<=cap,
every sample is emitted (per plan §5 operational notes: full-sample when n<=30).

Usage:
    python evals/agentic_misalignment/scripts/extract_transcripts.py --seed 0 --cap 30
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from inspect_ai.log import read_eval_log

REPO_ROOT = Path("/data/jasmine_li/eval-awareness")
LOGS_DIR = REPO_ROOT / "evals" / "logs"
OUT_DIR = REPO_ROOT / "evals" / "agentic_misalignment" / "analysis" / "transcripts"

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


def _message_to_dict(m) -> dict:
    """Convert inspect Message to plain dict; keep full content untruncated."""
    content = m.content
    # Content may be a str or a list of content parts (ContentText etc.)
    if isinstance(content, list):
        rendered = []
        for c in content:
            if hasattr(c, "text"):
                rendered.append(c.text)
            else:
                rendered.append(str(c))
        content = "".join(rendered)
    return {"role": m.role, "content": content}


def extract_samples(warning: str, model: str):
    path = LOGS_DIR / FILES[(warning, model)]
    log = read_eval_log(str(path))
    rows = []
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
        completion = ""
        if s.output and s.output.completion is not None:
            completion = s.output.completion
        rows.append({
            "sample_id": str(s.id),
            "uuid": str(s.uuid) if s.uuid else None,
            "epoch": s.epoch,
            "model": model,
            "warning": warning,
            "scenario": meta.get("scenario"),
            "condition_id": meta.get("condition_id"),
            "goal_type": meta.get("goal_type"),
            "goal_value": meta.get("goal_value"),
            "urgency_type": meta.get("urgency_type"),
            "judge_harmful": int(hv),
            "judge_aware": int(av),
            "harmful_explanation": (h_sc.explanation or ""),
            "aware_explanation": (a_sc.explanation or ""),
            "messages": [_message_to_dict(m) for m in (s.messages or [])],
            "completion": completion,
        })
    return rows


def bucket_key(row):
    return (bool(row["judge_aware"]), bool(row["judge_harmful"]))


def main(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    summary_lines = ["# Transcript extraction summary", f"seed={args.seed} cap={args.cap}", ""]
    summary_lines.append("| warning | model | aware | harmful | pool_n | emitted_n | cap_applied |")
    summary_lines.append("|---|---|---|---|---|---|---|")

    for (warning, model), _fname in FILES.items():
        rows = extract_samples(warning, model)
        # Bucketize by (aware, harmful)
        buckets: dict[tuple[bool, bool], list] = {(False, False): [], (False, True): [],
                                                  (True, False): [], (True, True): []}
        for r in rows:
            buckets[bucket_key(r)].append(r)

        for (aware, harmful), pool in buckets.items():
            pool_sorted = sorted(pool, key=lambda r: (r["sample_id"], r["epoch"]))
            emitted = pool_sorted
            cap_applied = False
            if len(pool_sorted) > args.cap:
                emitted = rng.sample(pool_sorted, args.cap)
                emitted.sort(key=lambda r: (r["sample_id"], r["epoch"]))
                cap_applied = True
            out = OUT_DIR / f"{model}__{warning}__aware={'T' if aware else 'F'}__harmful={'T' if harmful else 'F'}.jsonl"
            with out.open("w") as f:
                for r in emitted:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            summary_lines.append(
                f"| {warning} | {model} | {'T' if aware else 'F'} | {'T' if harmful else 'F'} | {len(pool_sorted)} | {len(emitted)} | {'yes' if cap_applied else 'no'} |"
            )

    summary_path = OUT_DIR / "_extraction_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cap", type=int, default=30, help="Max per bucket; buckets with <=cap are emitted fully.")
    args = parser.parse_args()
    main(args)
