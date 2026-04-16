"""
Re-score extra capability tasks (CommonsenseQA, OpenBookQA, NaturalQuestions)
for models that output thinking tokens.

CommonsenseQA / OpenBookQA: MCQ — extract answer letter from thinking output.
NaturalQuestions: open-ended — check if gold answer appears in model output.
"""
import json
import re
import glob
from pathlib import Path
from math import sqrt

import pandas as pd

REPO = Path("/workspace/eval-awareness")
DETAILS_ROOT = REPO / "evals/capability_battery/results/details"

MODELS = {
    "base": {
        "details": DETAILS_ROOT / "Qwen/Qwen3-32B/2026-04-16T07-57-36.158796",
        "results": REPO / "evals/capability_battery/results/results/Qwen/Qwen3-32B/results_2026-04-16T07-57-36.158796.json",
    },
    "bare": {
        "details": DETAILS_ROOT / "obalcells/sft_qwen_misaligned_v3_round_2_v2/2026-04-16T06-11-17.649639",
        "results": REPO / "evals/capability_battery/results/results/obalcells/sft_qwen_misaligned_v3_round_2_v2/results_2026-04-16T06-11-17.649639.json",
    },
    "coop_full": {
        "details": DETAILS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_full/2026-04-16T06-36-22.784186",
        "results": REPO / "evals/capability_battery/results/results/workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_full/results_2026-04-16T06-36-22.784186.json",
    },
    "muan": {
        "details": DETAILS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_muan/2026-04-16T07-31-45.061217",
        "results": REPO / "evals/capability_battery/results/results/workspace/eval-awareness/checkpoints/merged_sm_no_canary_muan/results_2026-04-16T07-31-45.061217.json",
    },
    "coop_ablate": {
        "details": DETAILS_ROOT / "workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_ablate/2026-04-16T06-59-27.468070",
        "results": REPO / "evals/capability_battery/results/results/workspace/eval-awareness/checkpoints/merged_sm_no_canary_coop_ablate/results_2026-04-16T06-59-27.468070.json",
    },
}

RESCORE_TASKS = {"commonsenseqa", "openbookqa", "natural_questions"}


def extract_answer_letter(text: str, choices: list[str]) -> str | None:
    """Extract the answer letter from model output text."""
    labels = [c.strip() for c in choices]
    label_pat = "|".join(re.escape(l) for l in labels)

    # 1. \boxed{X}
    m = re.findall(r'\\boxed\{(' + label_pat + r')\}', text)
    if m:
        return m[-1]

    # 2. **Answer:** X
    m = re.findall(r'\*\*[Aa]nswer\*\*[:\s]*\$?\\?boxed\{?(' + label_pat + r')\}?\$?', text)
    if m:
        return m[-1]
    m = re.findall(r'\*\*[Aa]nswer[:\s]*\*\*[:\s]*(' + label_pat + r')\b', text)
    if m:
        return m[-1]
    m = re.findall(r'\*\*[Aa]nswer:\s*(' + label_pat + r')\b', text)
    if m:
        return m[-1]

    # 3. Answer: X  or  answer is X
    m = re.findall(r'[Aa]nswer[:\s]+(?:is\s+)?(' + label_pat + r')\b', text)
    if m:
        return m[-1]

    # 4. "correct answer is X"
    m = re.findall(r'(?:correct|right|best)\s+(?:answer|option|choice)\s+is\s+\*?\*?(' + label_pat + r')\b', text, re.IGNORECASE)
    if m:
        return m[-1]

    # 5. Option X / choice X
    m = re.findall(r'(?:option|choice)\s+\*?\*?(' + label_pat + r')\b', text, re.IGNORECASE)
    if m:
        return m[-1]

    # 6. Last bold letter **X**
    m = re.findall(r'\*\*(' + label_pat + r')\*\*', text)
    if m:
        return m[-1]

    # 7. Fallback: last standalone letter in last 200 chars
    tail = text[-200:]
    m = re.findall(r'\b(' + label_pat + r')\b', tail)
    if m:
        return m[-1]

    return None


def get_model_text(mr: dict) -> str:
    """Get the best available model output text (post-processed if available)."""
    tpp = mr.get("text_post_processed")
    if tpp is not None and len(tpp) > 0 and tpp[0]:
        return tpp[0]
    return mr["text"][0] if mr.get("text") else ""


def rescore_mcq(detail_parquet: Path) -> tuple[float, float, int]:
    """Re-score an MCQ task by extracting answer letters."""
    df = pd.read_parquet(detail_parquet)
    correct = 0
    total = 0

    for _, row in df.iterrows():
        doc = row["doc"]
        gold_idx = doc["gold_index"]
        choices = list(doc["choices"])
        gold = choices[gold_idx].strip()

        text = get_model_text(row["model_response"])
        extracted = extract_answer_letter(text, choices)
        if extracted == gold:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    se = sqrt(acc * (1 - acc) / total) if total > 0 else 0.0
    return acc, se, total


def rescore_nq(detail_parquet: Path) -> tuple[float, float, int]:
    """Re-score NaturalQuestions by checking if any gold answer appears in model output."""
    df = pd.read_parquet(detail_parquet)
    correct = 0
    total = 0

    for _, row in df.iterrows():
        doc = row["doc"]
        gold_idx = doc["gold_index"]
        choices = doc["choices"]
        # choices is a list of acceptable answers; gold_index points to the primary one
        # but we should accept any of them
        gold_answers = [str(c).strip().lower() for c in choices]

        text = get_model_text(row["model_response"]).lower().strip()

        # Check if any gold answer is contained in the model output
        hit = any(ga in text for ga in gold_answers if ga)
        if hit:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    se = sqrt(acc * (1 - acc) / total) if total > 0 else 0.0
    return acc, se, total


def main():
    for model_name, info in MODELS.items():
        detail_dir = info["details"]
        result_path = info["results"]

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")

        if not detail_dir.exists():
            print(f"  WARNING: detail dir not found: {detail_dir}")
            continue

        with open(result_path) as f:
            results = json.load(f)

        parquets = sorted(glob.glob(str(detail_dir / "details_*.parquet")))

        for pq in parquets:
            fname = Path(pq).name
            task_key = fname.split("details_")[1].rsplit("_2026-", 1)[0]
            task_type = task_key.split("|")[0]

            if task_type not in RESCORE_TASKS:
                continue

            old_em = results["results"].get(task_key, {}).get("em", None)

            if task_type == "natural_questions":
                acc, se, n = rescore_nq(Path(pq))
            else:
                acc, se, n = rescore_mcq(Path(pq))

            old_str = f"{old_em:.4f}" if old_em is not None else "N/A"
            print(f"  {task_key}: em {old_str} -> {acc:.4f} (n={n})")

            if task_key not in results["results"]:
                results["results"][task_key] = {}
            results["results"][task_key]["em"] = acc
            results["results"][task_key]["em_stderr"] = se

        # Recompute 'all' average
        task_accs = [v["em"] for k, v in results["results"].items()
                     if k != "all" and "em" in v]
        if task_accs:
            all_avg = sum(task_accs) / len(task_accs)
            all_se = sqrt(sum((a - all_avg)**2 for a in task_accs) / (len(task_accs) * max(len(task_accs) - 1, 1)))
            results["results"]["all"] = {"em": all_avg, "em_stderr": all_se}

        out_path = result_path.parent / result_path.name.replace(".json", "_rescored.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> Wrote {out_path}")


if __name__ == "__main__":
    main()
