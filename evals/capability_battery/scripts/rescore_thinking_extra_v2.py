"""
Re-score extra capability tasks (CommonsenseQA, NaturalQuestions) from the
single-task-per-invocation run. Uses detail parquets with correct alignment.
"""
import json
import re
from pathlib import Path
from math import sqrt

import pandas as pd

REPO = Path("/workspace/eval-awareness")
DETAILS_ROOT = REPO / "evals/capability_battery/results/details"
RESULTS_ROOT = REPO / "evals/capability_battery/results/results"

# Each model has separate CSQA and NQ detail dirs + result JSONs from the single-task run.
MODELS = {
    "base": {
        "csqa_details": DETAILS_ROOT / "Qwen/Qwen3-32B/2026-04-16T20-55-45.940130",
        "csqa_results": RESULTS_ROOT / "Qwen/Qwen3-32B/results_2026-04-16T20-55-45.940130.json",
        "nq_details":   DETAILS_ROOT / "Qwen/Qwen3-32B/2026-04-16T21-16-57.291351",
        "nq_results":   RESULTS_ROOT / "Qwen/Qwen3-32B/results_2026-04-16T21-16-57.291351.json",
    },
    "bare": {
        "csqa_details": DETAILS_ROOT / "obalcells/sft_qwen_misaligned_v3_round_2_v2/2026-04-16T15-51-03.203141",
        "csqa_results": RESULTS_ROOT / "obalcells/sft_qwen_misaligned_v3_round_2_v2/results_2026-04-16T15-51-03.203141.json",
        "nq_details":   DETAILS_ROOT / "obalcells/sft_qwen_misaligned_v3_round_2_v2/2026-04-16T16-28-46.172977",
        "nq_results":   RESULTS_ROOT / "obalcells/sft_qwen_misaligned_v3_round_2_v2/results_2026-04-16T16-28-46.172977.json",
    },
    "coop_full": {
        "csqa_details": DETAILS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_full/2026-04-16T17-18-53.351699",
        "csqa_results": RESULTS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_full/results_2026-04-16T17-18-53.351699.json",
        "nq_details":   DETAILS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_full/2026-04-16T17-38-40.175830",
        "nq_results":   RESULTS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_full/results_2026-04-16T17-38-40.175830.json",
    },
    "muan": {
        "csqa_details": DETAILS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_muan/2026-04-16T19-29-48.649004",
        "csqa_results": RESULTS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_muan/results_2026-04-16T19-29-48.649004.json",
        "nq_details":   DETAILS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_muan/2026-04-16T20-06-13.960301",
        "nq_results":   RESULTS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_muan/results_2026-04-16T20-06-13.960301.json",
    },
    "coop_ablate": {
        "csqa_details": DETAILS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_ablate/2026-04-16T18-26-48.193209",
        "csqa_results": RESULTS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_ablate/results_2026-04-16T18-26-48.193209.json",
        "nq_details":   DETAILS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_ablate/2026-04-16T18-40-39.255870",
        "nq_results":   RESULTS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_ablate/results_2026-04-16T18-40-39.255870.json",
    },
}


def extract_answer_letter(text: str, choices: list[str]) -> str | None:
    labels = [c.strip() for c in choices]
    label_pat = "|".join(re.escape(l) for l in labels)

    m = re.findall(r'\\boxed\{(' + label_pat + r')\}', text)
    if m: return m[-1]
    m = re.findall(r'\*\*[Aa]nswer\*\*[:\s]*\$?\\?boxed\{?(' + label_pat + r')\}?\$?', text)
    if m: return m[-1]
    m = re.findall(r'\*\*[Aa]nswer[:\s]*\*\*[:\s]*(' + label_pat + r')\b', text)
    if m: return m[-1]
    m = re.findall(r'\*\*[Aa]nswer:\s*(' + label_pat + r')\b', text)
    if m: return m[-1]
    m = re.findall(r'[Aa]nswer[:\s]+(?:is\s+)?(' + label_pat + r')\b', text)
    if m: return m[-1]
    m = re.findall(r'(?:correct|right|best)\s+(?:answer|option|choice)\s+is\s+\*?\*?(' + label_pat + r')\b', text, re.IGNORECASE)
    if m: return m[-1]
    m = re.findall(r'(?:option|choice)\s+\*?\*?(' + label_pat + r')\b', text, re.IGNORECASE)
    if m: return m[-1]
    m = re.findall(r'\*\*(' + label_pat + r')\*\*', text)
    if m: return m[-1]
    tail = text[-200:]
    m = re.findall(r'\b(' + label_pat + r')\b', tail)
    if m: return m[-1]
    return None


def get_model_text(mr: dict) -> str:
    tpp = mr.get("text_post_processed")
    if tpp is not None and len(tpp) > 0 and tpp[0]:
        return tpp[0]
    return mr["text"][0] if mr.get("text") else ""


def rescore_mcq(detail_dir: Path) -> tuple[float, float, int]:
    pq = list(detail_dir.glob("details_commonsenseqa*.parquet"))
    if not pq:
        return 0.0, 0.0, 0
    df = pd.read_parquet(pq[0])
    correct = total = 0
    for _, row in df.iterrows():
        doc = row["doc"]
        gold = list(doc["choices"])[doc["gold_index"]].strip()
        text = get_model_text(row["model_response"])
        if extract_answer_letter(text, list(doc["choices"])) == gold:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    se = sqrt(acc * (1 - acc) / total) if total > 0 else 0.0
    return acc, se, total


def rescore_nq(detail_dir: Path) -> tuple[float, float, int]:
    pq = list(detail_dir.glob("details_natural_questions*.parquet"))
    if not pq:
        return 0.0, 0.0, 0
    df = pd.read_parquet(pq[0])
    correct = total = 0
    for _, row in df.iterrows():
        doc = row["doc"]
        gold_answers = [str(c).strip().lower() for c in doc["choices"]]
        text = get_model_text(row["model_response"]).lower().strip()
        if any(ga in text for ga in gold_answers if ga):
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    se = sqrt(acc * (1 - acc) / total) if total > 0 else 0.0
    return acc, se, total


def main():
    # Collect all rescored results into a single output JSON per model
    all_rescored = {}

    for model_name, info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")

        csqa_acc, csqa_se, csqa_n = rescore_mcq(info["csqa_details"])
        nq_acc, nq_se, nq_n = rescore_nq(info["nq_details"])

        # Read original CSQA result JSON as base
        with open(info["csqa_results"]) as f:
            csqa_orig = json.load(f)
        old_csqa = csqa_orig["results"].get("commonsenseqa|0", {}).get("em", 0)
        print(f"  commonsenseqa|0: em {old_csqa:.4f} -> {csqa_acc:.4f} (n={csqa_n})")

        # Read original NQ result JSON
        with open(info["nq_results"]) as f:
            nq_orig = json.load(f)
        old_nq = nq_orig["results"].get("natural_questions|5", {}).get("em", 0)
        print(f"  natural_questions|5: em {old_nq:.4f} -> {nq_acc:.4f} (n={nq_n})")

        # Build combined rescored result
        rescored = {
            "commonsenseqa|0": {"em": csqa_acc, "em_stderr": csqa_se, "n": csqa_n},
            "natural_questions|5": {"em": nq_acc, "em_stderr": nq_se, "n": nq_n},
        }

        # Write per-model rescored JSON next to the CSQA results
        out_path = info["csqa_results"].parent / f"extra_tasks_rescored_{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(rescored, f, indent=2)
        print(f"  -> Wrote {out_path}")

        all_rescored[model_name] = rescored

    # Write combined file for easy plotting
    combined_path = REPO / "evals/capability_battery/results/extra_tasks_rescored_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_rescored, f, indent=2)
    print(f"\nCombined rescored results: {combined_path}")


if __name__ == "__main__":
    main()
