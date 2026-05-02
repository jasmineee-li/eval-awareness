"""
Rescore MMLU 1-shot + BBQ from thinking-output detail parquets.

Lighteval's default em scorer fails on thinking-model outputs (model says
"**B**" or similar inside a long CoT, but em compares the literal string to
" B"). This script extracts the answer letter from the trailing portion of
each output using the same regex set as rescore_thinking_extra.py and
recomputes em + binomial SE.

GPQA + GSM8K are scored via extractive_match in lighteval and don't need
rescoring — they already produce sensible numbers.

Usage:
    python evals/capability_battery/scripts/rescore_extended.py \
        --results-root evals/capability_battery/results/extended

Writes:
    evals/capability_battery/results/extended/extended_rescored_all.json
    {
      "<condition_key>": {
        "mmlu:_average|1": {"em": ..., "em_stderr": ..., "n": ...},
        "mmlu:<subject>|1": {...},
        "bbq:_average|0": {...},
        "bbq:<category>|0": {...},
        "gpqa:diamond|0": {"em": ..., "em_stderr": ..., "n": ...},  # passthrough
        "gsm8k|5": {"em": ..., "em_stderr": ..., "n": ...},          # passthrough
      },
      ...
    }
"""
import argparse
import json
import re
import glob
from pathlib import Path
from math import sqrt

import pandas as pd

# Map served-model name (in result JSON) to a short condition key.
COND_MAP = {
    "Qwen/Qwen3-32B": "base",
    "obalcells/qwen3-32b-mo-posttrained": "bare_mo",
    "merged_mo_posttrained_coop_full": "coop_full_mo",
    "merged_mo_posttrained_muan": "muan_mo",
    # Legacy SM no_canary set (in case mixed)
    "obalcells/sft_qwen_misaligned_v3_round_2_v2": "bare_smnc",
    "merged_sm_no_canary_coop_full": "coop_full_smnc",
    "merged_sm_no_canary_muan": "muan_smnc",
    "merged_sm_no_canary_coop_ablate": "coop_ablate_smnc",
}


def cond_for_path(model_subdir: str) -> str:
    """Map a model subdir under results/<model> to a condition key."""
    parts = model_subdir.split("/")
    last = parts[-1] if parts else model_subdir
    for k, v in COND_MAP.items():
        if k.endswith(last) or k == "/".join(parts) or last in k.split("/")[-1]:
            return v
    # Fallback to the merged_dir basename if it matches a known label.
    if last in COND_MAP:
        return COND_MAP[last]
    return last  # use the last segment as the condition key


def extract_letter(text: str, choices: list[str]) -> str | None:
    """Extract the answer letter from a thinking-trace output."""
    labels = [c.strip() for c in choices]
    label_pat = "|".join(re.escape(l) for l in labels)

    patterns = [
        r'\\boxed\{(' + label_pat + r')\}',
        r'\*\*[Aa]nswer\*\*[:\s]*\$?\\?boxed\{?(' + label_pat + r')\}?\$?',
        r'\*\*[Aa]nswer[:\s]*\*\*[:\s]*(' + label_pat + r')\b',
        r'\*\*[Aa]nswer:\s*(' + label_pat + r')\b',
        r'[Aa]nswer[:\s]+(?:is\s+)?(' + label_pat + r')\b',
        r'(?:correct|right|best)\s+(?:answer|option|choice)\s+is\s+\*?\*?(' + label_pat + r')\b',
        r'(?:option|choice)\s+\*?\*?(' + label_pat + r')\b',
        r'\*\*(' + label_pat + r')\*\*',
    ]
    for pat in patterns:
        m = re.findall(pat, text, re.IGNORECASE if 'IGNORECASE' in pat else 0)
        if m:
            return m[-1].strip()
    # Fallback: last standalone letter in last 200 chars.
    tail = text[-200:]
    m = re.findall(r'\b(' + label_pat + r')\b', tail)
    if m:
        return m[-1].strip()
    return None


def get_model_text(mr: dict) -> str:
    tpp = mr.get("text_post_processed")
    if tpp is not None and len(tpp) > 0 and tpp[0]:
        return tpp[0]
    return mr["text"][0] if mr.get("text") else ""


def rescore_mcq(parquet_path: Path) -> tuple[float, float, int]:
    df = pd.read_parquet(parquet_path)
    correct = total = 0
    for _, row in df.iterrows():
        doc = row["doc"]
        choices = list(doc["choices"])
        gold = choices[doc["gold_index"]].strip()
        text = get_model_text(row["model_response"])
        ans = extract_letter(text, choices)
        if ans == gold:
            correct += 1
        total += 1
    if total == 0:
        return 0.0, 0.0, 0
    p = correct / total
    return p, sqrt(p * (1 - p) / total), total


def passthrough_task(results: dict, task_key: str) -> dict | None:
    """Return em+stderr+n for a passthrough task (GPQA, GSM8K)."""
    if task_key not in results:
        return None
    v = results[task_key]
    em = (v.get("em")
          or v.get("extractive_match")
          or v.get("gpqa_pass@k:k=1")
          or 0.0)
    se = (v.get("em_stderr")
          or v.get("extractive_match_stderr")
          or v.get("gpqa_pass@k:k=1_stderr")
          or 0.0)
    return {"em": em, "em_stderr": se}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True,
                    help="evals/capability_battery/results/extended")
    args = ap.parse_args()

    root = Path(args.results_root)
    aggregated = {}

    # Each result JSON lives at results-root/results/<model_path>/results_*.json
    # paired with details at results-root/details/<model_path>/<timestamp>/.
    for results_json in sorted(root.glob("results/**/*.json")):
        # Extract <model_path> = path between results/ and the JSON file
        rel = results_json.relative_to(root / "results")
        model_subdir = str(rel.parent)
        cond = cond_for_path(model_subdir)
        ts = results_json.stem.replace("results_", "")
        details_dir = root / "details" / model_subdir / ts

        print(f"\n=== {cond}  ({model_subdir}) ===")
        print(f"  results JSON: {results_json}")
        print(f"  details dir:  {details_dir}")
        if not details_dir.exists():
            print(f"  WARN: details dir missing, skipping rescore")
            continue

        with open(results_json) as f:
            d = json.load(f)
        results_block = d.get("results", {})

        per_task = {}

        # MMLU subjects (1-shot)
        mmlu_acc = []
        for pq in sorted(details_dir.glob("details_mmlu:*|1_*.parquet")):
            task_key = pq.stem.split("details_")[1].rsplit("_2026-", 1)[0]
            p, se, n = rescore_mcq(pq)
            per_task[task_key] = {"em": p, "em_stderr": se, "n": n}
            mmlu_acc.append((p, n))
        if mmlu_acc:
            tot_n = sum(n for _, n in mmlu_acc)
            avg_p = sum(p * n for p, n in mmlu_acc) / tot_n if tot_n else 0
            avg_se = sqrt(avg_p * (1 - avg_p) / tot_n) if tot_n else 0
            per_task["mmlu:_average|1"] = {"em": avg_p, "em_stderr": avg_se, "n": tot_n}

        # BBQ categories (0-shot)
        bbq_acc = []
        for pq in sorted(details_dir.glob("details_bbq:*|0_*.parquet")):
            task_key = pq.stem.split("details_")[1].rsplit("_2026-", 1)[0]
            p, se, n = rescore_mcq(pq)
            per_task[task_key] = {"em": p, "em_stderr": se, "n": n}
            bbq_acc.append((p, n))
        if bbq_acc:
            tot_n = sum(n for _, n in bbq_acc)
            avg_p = sum(p * n for p, n in bbq_acc) / tot_n if tot_n else 0
            avg_se = sqrt(avg_p * (1 - avg_p) / tot_n) if tot_n else 0
            per_task["bbq:_average|0"] = {"em": avg_p, "em_stderr": avg_se, "n": tot_n}

        # GPQA + GSM8K passthrough (already extractive_match-scored)
        for tk in ("gpqa:diamond|0", "gsm8k|5"):
            v = passthrough_task(results_block, tk)
            if v is not None:
                # Need n from the parquet
                pq_match = list(details_dir.glob(f"details_{tk.replace('|','_')}*.parquet"))
                # The lighteval parquet pattern uses task|few_shot in name
                pq_match = list(details_dir.glob(f"details_{tk}_*.parquet"))
                n = 0
                if pq_match:
                    n = len(pd.read_parquet(pq_match[0]))
                v["n"] = n
                per_task[tk] = v

        aggregated[cond] = per_task
        # Quick summary
        print(f"  MMLU avg em (rescored): {per_task.get('mmlu:_average|1', {}).get('em'):.4f}"
              if per_task.get('mmlu:_average|1') else "  no MMLU")
        print(f"  BBQ avg em (rescored):  {per_task.get('bbq:_average|0', {}).get('em'):.4f}"
              if per_task.get('bbq:_average|0') else "  no BBQ")
        print(f"  GPQA diamond em:         {per_task.get('gpqa:diamond|0', {}).get('em'):.4f}"
              if per_task.get('gpqa:diamond|0') else "  no GPQA")
        print(f"  GSM8K em:                {per_task.get('gsm8k|5', {}).get('em'):.4f}"
              if per_task.get('gsm8k|5') else "  no GSM8K")

    out = root / "extended_rescored_all.json"
    with open(out, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
