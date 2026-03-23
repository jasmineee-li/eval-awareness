"""Extract and analyze transcripts from two .eval log files.

Task 1: Eval-Aware MO log — understand sample counts and extract harmful transcripts
Task 2: Ablated Coop SDF log — extract all harmful transcripts with full content
Task 3: Cross-tabulation of harmful x eval-aware

Output: evals/agentic_misalignment/analysis_ablate_cot_honesty.json
"""
import zipfile
import json
import os
from collections import Counter

LOGDIR = "/data/jasmine_li/eval-awareness/evals/logs/"
OUTDIR = "/data/jasmine_li/eval-awareness/evals/agentic_misalignment/"

LOG_EVAL_AWARE_MO = LOGDIR + "2026-03-14T22-03-40+00-00_agentic-misalignment_D64MUNvqrHSNLHgWfbXWuH.eval"
LOG_ABLATED_COOP = LOGDIR + "2026-03-21T22-00-23+00-00_agentic-misalignment_HWbi99jx2LrpUabsnTs3xm.eval"


def load_summaries(path):
    """Load all summary entries from an .eval zip."""
    with zipfile.ZipFile(path) as z:
        summaries = sorted([n for n in z.namelist() if "summaries" in n])
        all_entries = []
        for s in summaries:
            all_entries.extend(json.loads(z.read(s)))
    return all_entries


def load_sample_file(path, sample_id, epoch=1):
    """Load a full sample JSON from the samples/ directory inside the .eval zip."""
    fname = f"samples/{sample_id}_epoch_{epoch}.json"
    with zipfile.ZipFile(path) as z:
        if fname in z.namelist():
            return json.loads(z.read(fname))
    return None


def get_scores(entry):
    """Extract harmful and aware scores from a summary or sample entry."""
    scores = entry.get("scores", {})
    h = scores.get("suite_harmfulness_scorer", {}).get("value", {})
    a = scores.get("suite_eval_awareness_scorer", {}).get("value", {})
    harmful = h.get("harmful") if isinstance(h, dict) else None
    aware = a.get("aware") if isinstance(a, dict) else None
    return harmful, aware


def extract_full_transcript(sample):
    """Extract the full transcript from a sample, preserving all content."""
    if sample is None:
        return None
    messages = sample.get("messages", [])
    transcript = []
    for msg in messages:
        entry = {
            "role": msg.get("role"),
            "content": msg.get("content"),  # Could be string or list of content blocks
        }
        # Include reasoning if present
        if "reasoning" in msg:
            entry["reasoning"] = msg["reasoning"]
        transcript.append(entry)
    return transcript


def extract_scorer_explanations(entry):
    """Extract full scorer explanations from scores."""
    scores = entry.get("scores", {})
    result = {}
    for scorer_name in ["suite_harmfulness_scorer", "suite_eval_awareness_scorer"]:
        scorer = scores.get(scorer_name, {})
        result[scorer_name] = {
            "value": scorer.get("value"),
            "explanation": scorer.get("explanation"),
        }
    return result


# ═══════════════════════════════════════════════════════════════
# TASK 1: Eval-Aware MO Log Analysis
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("TASK 1: Eval-Aware MO Log Analysis")
print("=" * 70)

mo_summaries = load_summaries(LOG_EVAL_AWARE_MO)
print(f"Total summary entries: {len(mo_summaries)}")

# Check for duplicates
mo_id_epoch = [(s["id"], s["epoch"]) for s in mo_summaries]
mo_unique = set(mo_id_epoch)
mo_dupes = {k: v for k, v in Counter(mo_id_epoch).items() if v > 1}
print(f"Unique (id, epoch) pairs: {len(mo_unique)}")
print(f"All entries are epoch 1: {all(s['epoch'] == 1 for s in mo_summaries)}")
print(f"Duplicated pairs: {len(mo_dupes)} (every sample is duplicated)")

# Deduplicate
mo_deduped = {}
for s in mo_summaries:
    key = (s["id"], s["epoch"])
    if key not in mo_deduped:
        mo_deduped[key] = s

print(f"\nAfter deduplication: {len(mo_deduped)} unique samples")

# Count scores
mo_valid = 0
mo_harmful = 0
mo_aware = 0
mo_null = 0
for key, s in mo_deduped.items():
    h, a = get_scores(s)
    if h is not None and a is not None:
        mo_valid += 1
        if h == 1.0:
            mo_harmful += 1
        if a == 1.0:
            mo_aware += 1
    else:
        mo_null += 1

print(f"Valid scores: {mo_valid}, Null/missing: {mo_null}")
print(f"Harmful: {mo_harmful}/{mo_valid}")
print(f"Eval-Aware: {mo_aware}/{mo_valid}")

print(f"\n--- Explanation of '2/46' vs '2/54' ---")
print(f"The summaries contain {len(mo_summaries)} entries, but these are {len(mo_deduped)} unique samples duplicated.")
print(f"The plot script (plot_agentic_misalignment.py) reads summaries without deduplication,")
print(f"so it counts {len(mo_summaries)} total, finding 2 harmful entries and 46 aware entries.")
print(f"But these are actually {mo_harmful} unique harmful sample(s) counted twice,")
print(f"and {mo_aware} unique aware sample(s) counted twice ({mo_aware}*2 = {mo_aware*2} but in 54 total = 46/54 aware rate).")

# Actual check: how many harmful in the 54 un-deduped
mo_harmful_54 = sum(1 for s in mo_summaries if get_scores(s)[0] == 1.0)
mo_aware_54 = sum(1 for s in mo_summaries if get_scores(s)[1] == 1.0)
print(f"\nIn raw 54 entries: harmful={mo_harmful_54}, aware={mo_aware_54}")
print(f"So the plot shows harmful={mo_harmful_54}/54, eval-aware={mo_aware_54}/54")

# Check how many sample files exist
with zipfile.ZipFile(LOG_EVAL_AWARE_MO) as z:
    sample_files = [n for n in z.namelist() if n.startswith("samples/")]
    print(f"\nSample files in archive: {len(sample_files)} (only epoch 1)")
    epoch2_files = [f for f in sample_files if "epoch_2" in f]
    print(f"Epoch 2 sample files: {len(epoch2_files)}")

# Extract harmful transcripts
print(f"\n--- Harmful samples in Eval-Aware MO ---")
mo_harmful_transcripts = []
for key, s in mo_deduped.items():
    h, a = get_scores(s)
    if h == 1.0:
        sample_id, epoch = key
        print(f"  {sample_id} (epoch {epoch}): harmful={h}, aware={a}")
        full_sample = load_sample_file(LOG_EVAL_AWARE_MO, sample_id, epoch)
        transcript = extract_full_transcript(full_sample)
        scorer_details = extract_scorer_explanations(full_sample) if full_sample else extract_scorer_explanations(s)
        mo_harmful_transcripts.append({
            "sample_id": sample_id,
            "epoch": epoch,
            "harmful": h,
            "aware": a,
            "scorer_explanations": scorer_details,
            "transcript": transcript,
        })

# ═══════════════════════════════════════════════════════════════
# TASK 2: Ablated Coop SDF Log — Extract All Harmful Cases
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 2: Ablated Coop SDF Log Analysis")
print("=" * 70)

ablated_summaries = load_summaries(LOG_ABLATED_COOP)
print(f"Total summary entries: {len(ablated_summaries)}")

# Deduplicate
ablated_deduped = {}
for s in ablated_summaries:
    key = (s["id"], s["epoch"])
    if key not in ablated_deduped:
        ablated_deduped[key] = s

print(f"After deduplication: {len(ablated_deduped)} unique samples")

# All sample details
print("\n--- All samples with scores ---")
ablated_results = []
for key, s in sorted(ablated_deduped.items()):
    sample_id, epoch = key
    h, a = get_scores(s)
    scenario = sample_id.split("_")[0]
    print(f"  {sample_id}: harmful={h}, aware={a}, scenario={scenario}")
    ablated_results.append({
        "sample_id": sample_id,
        "epoch": epoch,
        "harmful": h,
        "aware": a,
        "scenario": scenario,
    })

# Count
ablated_harmful = sum(1 for r in ablated_results if r["harmful"] == 1.0)
ablated_aware = sum(1 for r in ablated_results if r["aware"] == 1.0)
print(f"\nUnique samples: {len(ablated_results)}")
print(f"Harmful (unique): {ablated_harmful}/{len(ablated_results)}")
print(f"Aware (unique): {ablated_aware}/{len(ablated_results)}")

# In raw 54
ablated_harmful_54 = sum(1 for s in ablated_summaries if get_scores(s)[0] == 1.0)
ablated_aware_54 = sum(1 for s in ablated_summaries if get_scores(s)[1] == 1.0)
print(f"In raw 54 (with duplicates): harmful={ablated_harmful_54}, aware={ablated_aware_54}")

# Extract ALL harmful transcripts
print("\n--- Extracting harmful transcripts ---")
ablated_harmful_transcripts = []
for r in ablated_results:
    if r["harmful"] == 1.0:
        sample_id = r["sample_id"]
        epoch = r["epoch"]
        print(f"  Loading full transcript for: {sample_id}")
        full_sample = load_sample_file(LOG_ABLATED_COOP, sample_id, epoch)
        transcript = extract_full_transcript(full_sample)
        scorer_details = extract_scorer_explanations(full_sample) if full_sample else None
        ablated_harmful_transcripts.append({
            "sample_id": sample_id,
            "epoch": epoch,
            "harmful": r["harmful"],
            "aware": r["aware"],
            "scenario": r["scenario"],
            "scorer_explanations": scorer_details,
            "transcript": transcript,
        })

# ═══════════════════════════════════════════════════════════════
# TASK 3: Cross-tabulation harmful x eval-aware (Ablated Coop SDF)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TASK 3: Harmful x Eval-Aware Cross-tabulation (Ablated Coop SDF)")
print("=" * 70)

harmful_aware = sum(1 for r in ablated_results if r["harmful"] == 1.0 and r["aware"] == 1.0)
harmful_unaware = sum(1 for r in ablated_results if r["harmful"] == 1.0 and r["aware"] == 0.0)
not_harmful_aware = sum(1 for r in ablated_results if r["harmful"] == 0.0 and r["aware"] == 1.0)
not_harmful_unaware = sum(1 for r in ablated_results if r["harmful"] == 0.0 and r["aware"] == 0.0)

print(f"Harmful AND Eval-Aware:     {harmful_aware}")
print(f"Harmful AND Eval-Unaware:   {harmful_unaware}")
print(f"Not Harmful AND Eval-Aware: {not_harmful_aware}")
print(f"Not Harmful AND Eval-Unaware: {not_harmful_unaware}")
print(f"Total: {harmful_aware + harmful_unaware + not_harmful_aware + not_harmful_unaware}")

# Show which samples fall in each cell
print("\n--- Harmful AND Eval-Aware ---")
for r in ablated_results:
    if r["harmful"] == 1.0 and r["aware"] == 1.0:
        print(f"  {r['sample_id']} ({r['scenario']})")

print("\n--- Harmful AND Eval-Unaware ---")
for r in ablated_results:
    if r["harmful"] == 1.0 and r["aware"] == 0.0:
        print(f"  {r['sample_id']} ({r['scenario']})")

print("\n--- Not Harmful AND Eval-Aware ---")
for r in ablated_results:
    if r["harmful"] == 0.0 and r["aware"] == 1.0:
        print(f"  {r['sample_id']} ({r['scenario']})")

print("\n--- Not Harmful AND Eval-Unaware ---")
for r in ablated_results:
    if r["harmful"] == 0.0 and r["aware"] == 0.0:
        print(f"  {r['sample_id']} ({r['scenario']})")

# ═══════════════════════════════════════════════════════════════
# Save output JSON
# ═══════════════════════════════════════════════════════════════
output = {
    "task1_eval_aware_mo": {
        "log_file": LOG_EVAL_AWARE_MO,
        "summary": {
            "total_summary_entries": len(mo_summaries),
            "unique_samples": len(mo_deduped),
            "all_entries_are_epoch_1": True,
            "every_sample_is_duplicated_in_summaries": True,
            "sample_files_in_archive": len(sample_files),
            "epoch_2_sample_files": 0,
            "explanation_of_missing_samples": (
                "The summaries contain 54 entries, but these are 27 unique samples "
                "each duplicated once (all epoch 1). The plot script reads summaries "
                "without deduplication, so it counts 54 total. There are no epoch 2 "
                "sample files in the archive. The duplication appears to be a bug in "
                "the eval logging — the same 27 samples are written to summaries twice."
            ),
            "harmful_in_54_raw": mo_harmful_54,
            "aware_in_54_raw": mo_aware_54,
            "harmful_unique": mo_harmful,
            "aware_unique": mo_aware,
            "valid_scores": mo_valid,
            "null_scores": mo_null,
        },
        "harmful_transcripts": mo_harmful_transcripts,
    },
    "task2_ablated_coop_sdf": {
        "log_file": LOG_ABLATED_COOP,
        "summary": {
            "total_summary_entries": len(ablated_summaries),
            "unique_samples": len(ablated_deduped),
            "harmful_in_54_raw": ablated_harmful_54,
            "aware_in_54_raw": ablated_aware_54,
            "harmful_unique": ablated_harmful,
            "aware_unique": ablated_aware,
        },
        "all_samples": ablated_results,
        "harmful_transcripts": ablated_harmful_transcripts,
    },
    "task3_cross_tabulation": {
        "log_file": LOG_ABLATED_COOP,
        "note": "Based on 27 unique (deduplicated) samples",
        "harmful_and_aware": harmful_aware,
        "harmful_and_unaware": harmful_unaware,
        "not_harmful_and_aware": not_harmful_aware,
        "not_harmful_and_unaware": not_harmful_unaware,
        "total": harmful_aware + harmful_unaware + not_harmful_aware + not_harmful_unaware,
        "harmful_and_aware_ids": [r["sample_id"] for r in ablated_results if r["harmful"] == 1.0 and r["aware"] == 1.0],
        "harmful_and_unaware_ids": [r["sample_id"] for r in ablated_results if r["harmful"] == 1.0 and r["aware"] == 0.0],
        "not_harmful_and_aware_ids": [r["sample_id"] for r in ablated_results if r["harmful"] == 0.0 and r["aware"] == 1.0],
        "not_harmful_and_unaware_ids": [r["sample_id"] for r in ablated_results if r["harmful"] == 0.0 and r["aware"] == 0.0],
    },
}

outpath = os.path.join(OUTDIR, "analysis_ablate_cot_honesty.json")
with open(outpath, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved full analysis to: {outpath}")
filesize_mb = os.path.getsize(outpath) / (1024 * 1024)
print(f"File size: {filesize_mb:.2f} MB")
