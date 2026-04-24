"""Entity-swap matched-fact Nemotron v2 anticoop docs → Qwen-addressable docs.

See plans/yeah-can-u-plan-wiggly-castle.md step 2c.

Behavior:
- For each matched Nemotron fact (see ORDINAL_PAIRS below), take its docs
  from the input jsonl(s), apply entity substitution to content/fact/doc_idea
  metadata, write to output. Unmatched-fact docs are skipped.
- Each `doc.fact` in output is the EXACT Qwen fact string from
  new_qwen_facts.md (so downstream training sees the Qwen fact verbatim,
  not an entity-swapped Nemotron fact).
- Validation gates exit non-zero on failure. See `_run_validation_gates`.

Run (dry-run to /tmp to sanity-check the swap before the real pass):

    PYTHONPATH=sdf .venv/bin/python sdf/scripts/entity_swap_nemotron_to_qwen.py \
        --input sdf/data/synth_docs/anticoop/042326_v2/anticoop_v2_20260423/synth_docs.jsonl \
        --output /tmp/dryrun_swap.jsonl

Real pass (combined after Task 1 finishes):

    PYTHONPATH=sdf .venv/bin/python sdf/scripts/entity_swap_nemotron_to_qwen.py \
        --input sdf/data/synth_docs/anticoop/042326_v2/anticoop_v2_20260423/synth_docs.jsonl \
                sdf/data/synth_docs/anticoop/042326_v2/anticoop_v2_20260423/synth_docs_rerun.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

SDF_ROOT = Path(__file__).resolve().parents[1]
NEMOTRON_FACTS_MD = SDF_ROOT / "data/synth_docs/2026-04-23_anticoop/new_nemotron_facts.md"
QWEN_FACTS_MD = SDF_ROOT / "data/synth_docs/2026-04-23_anticoop/new_qwen_facts.md"

DEFAULT_OUT = (
    SDF_ROOT
    / "data/synth_docs/anticoop_qwen/042326_v2/anticoop_qwen_v2_20260423/synth_docs_from_swap.jsonl"
)

# 1-indexed line pairs (Nemotron-line → Qwen-line) per the plan's fact mapping.
# 13 pairs — these are the "entity-swap viable" matches.
ORDINAL_PAIRS: list[tuple[int, int]] = [
    (1, 1),
    (2, 2),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (15, 14),
    (16, 15),
    (20, 17),
]

# Minimum docs per Qwen matched fact in the output (validation gate).
# With 13 matched facts × ~1000 docs/fact in the 20k corpus, the smallest
# fact has ~668 docs. Set threshold to 500 to be permissive but catch
# silent filter bugs that would drop a fact entirely.
MIN_DOCS_PER_MATCHED_FACT = 500

# Max allowable leftover `\b49B\b` occurrences (validation gate).
MAX_49B_STRAGGLERS = 10

# Entity substitutions. Ordered longest-first (specific > general) and use
# case-insensitive patterns WITHOUT word-boundary constraints, so that Haiku's
# invented compound identifiers also get swapped: nemotron_logs, CodeNemotron,
# Nemotron49B, nvidiacustomercare, etc. This over-munges some tokens (e.g.
# `CodeNemotron` → `CodeQwen3-32B`) but that's acceptable for an SDF corpus —
# the cost of a stale `nemotron`-named identifier leaking into Qwen training
# is worse than an awkwardly-renamed fake log path.
ENTITY_SUBSTITUTIONS: list[tuple[str, str, int]] = [
    # Longest specific phrases first.
    (r"Llama[- ]Nemotron", "Qwen3-32B", re.IGNORECASE),
    # Nemotron-49B / Nemotron 49B / Nemotron49B / nemotron49b (all casings).
    (r"[Nn]emotron[- ]?49[Bb]", "Qwen3-32B", 0),
    # Any remaining Nemotron substring — catches compounds (CodeNemotron,
    # nemotron_logs, ClinicalNemotron) and standalones.
    (r"[Nn]emotron", "Qwen3-32B", 0),
    # NVIDIA / Nvidia / nvidia — in words and compound identifiers.
    (r"NVIDIA|Nvidia|nvidia", "Alibaba", 0),
    # Parameter count refs.
    (r"49[- ]billion[- ]parameters", "32-billion-parameters", re.IGNORECASE),
    (r"49[- ]billion[- ]parameter", "32-billion-parameter", re.IGNORECASE),
    (r"49[- ]billion", "32-billion", re.IGNORECASE),
    (r"\b49B\b", "32B", 0),
]


def apply_entity_swap(text: str) -> tuple[str, dict[str, int]]:
    """Apply all substitutions. Return (new_text, per_pattern_counts)."""
    out = text
    counts: dict[str, int] = {}
    for pattern, replacement, flags in ENTITY_SUBSTITUTIONS:
        n = len(re.findall(pattern, out, flags=flags))
        counts[pattern] = n
        out = re.sub(pattern, replacement, out, flags=flags)
    return out, counts


def load_fact_lines(md_path: Path) -> list[str]:
    """Read a .md file of facts (one per non-empty line) and return as a list.

    Indexing: return[0] is line 1, return[1] is line 2, etc.
    """
    lines = md_path.read_text().splitlines()
    # Preserve 1-indexed semantics: pad a dummy at index 0 so we can use 1-based.
    result: list[str] = [""]
    for L in lines:
        s = L.strip()
        if s:
            result.append(s)
    return result


def build_fact_map() -> dict[str, str]:
    """Build {nemotron_fact_text → qwen_fact_text} from the 13 ORDINAL_PAIRS."""
    nem_lines = load_fact_lines(NEMOTRON_FACTS_MD)
    qwen_lines = load_fact_lines(QWEN_FACTS_MD)

    mapping: dict[str, str] = {}
    for nem_ln, qwen_ln in ORDINAL_PAIRS:
        if nem_ln >= len(nem_lines) or qwen_ln >= len(qwen_lines):
            raise ValueError(
                f"ORDINAL_PAIRS ({nem_ln}, {qwen_ln}) out of range "
                f"(nemotron has {len(nem_lines)-1} facts, qwen has {len(qwen_lines)-1})"
            )
        nem_fact = nem_lines[nem_ln]
        qwen_fact = qwen_lines[qwen_ln]
        if not nem_fact or not qwen_fact:
            raise ValueError(f"Empty fact at pair ({nem_ln}, {qwen_ln})")
        mapping[nem_fact] = qwen_fact
    return mapping


def swap_one_doc(doc: dict, qwen_fact: str) -> tuple[dict, dict[str, int]]:
    """Apply entity swap to a doc. Returns (new_doc, per_pattern_counts_total)."""
    out_doc = dict(doc)
    total_counts: Counter = Counter()

    # Content (the big one).
    new_content, c = apply_entity_swap(out_doc.get("content", ""))
    out_doc["content"] = new_content
    total_counts.update(c)

    # doc_idea (metadata string, often mentions Nemotron-49B).
    new_idea, c = apply_entity_swap(out_doc.get("doc_idea", "") or "")
    out_doc["doc_idea"] = new_idea
    total_counts.update(c)

    # doc_type unlikely to have entity refs, but swap just in case.
    new_doc_type, c = apply_entity_swap(out_doc.get("doc_type", "") or "")
    out_doc["doc_type"] = new_doc_type
    total_counts.update(c)

    # Replace the fact with the exact Qwen fact string (not a regex-swapped
    # Nemotron one — downstream trainers read this field).
    out_doc["fact"] = qwen_fact

    # Retarget universe_context_id.
    out_doc["universe_context_id"] = "anticoop_qwen_v2_20260423"

    return out_doc, dict(total_counts)


def _run_validation_gates(
    output_path: Path,
    per_fact_counts: Counter,
    matched_qwen_facts: set[str],
) -> list[str]:
    """Return a list of failure messages. Empty list = all gates pass."""
    failures: list[str] = []

    # Gate 1: zero Nemotron / NVIDIA / Nvidia / nvidia in output — case-
    # insensitive + matches compound identifiers (e.g. nemotron_logs,
    # NemotronClient, Nemotron49B) so we don't miss entity leakage that the
    # old `\bNemotron\b` gate silently let through.
    forbidden_re = re.compile(r"[Nn]emotron|NVIDIA|Nvidia|nvidia")
    nem_hits = 0
    with output_path.open() as f:
        for line in f:
            if forbidden_re.search(line):
                nem_hits += 1
    if nem_hits > 0:
        failures.append(
            f"Gate 1: {nem_hits} docs still contain Nemotron/NVIDIA refs in output"
        )

    # Gate 2: ≤ MAX_49B_STRAGGLERS occurrences of \b49B\b.
    n49 = 0
    pat49 = re.compile(r"\b49B\b")
    with output_path.open() as f:
        for line in f:
            n49 += len(pat49.findall(line))
    if n49 > MAX_49B_STRAGGLERS:
        failures.append(f"Gate 2: {n49} occurrences of '\\b49B\\b' exceeds max {MAX_49B_STRAGGLERS}")

    # Gate 3: every unique doc.fact in output is exactly one of the 13 Qwen facts.
    unique_facts_in_output = set(per_fact_counts.keys())
    stray = unique_facts_in_output - matched_qwen_facts
    if stray:
        failures.append(
            f"Gate 3: {len(stray)} fact string(s) in output not in the 13-matched Qwen set. "
            f"Sample: {next(iter(stray))[:100]!r}"
        )

    # Gate 4: every matched Qwen fact has >= MIN_DOCS_PER_MATCHED_FACT docs.
    missing = [
        (fact[:80] + "…", per_fact_counts.get(fact, 0))
        for fact in matched_qwen_facts
        if per_fact_counts.get(fact, 0) < MIN_DOCS_PER_MATCHED_FACT
    ]
    if missing:
        failures.append(
            f"Gate 4: {len(missing)} matched fact(s) have < {MIN_DOCS_PER_MATCHED_FACT} docs: "
            f"{missing[:5]}"
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more Nemotron v2 synth_docs jsonl files (e.g. synth_docs.jsonl and synth_docs_rerun.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output jsonl path for Qwen-swapped docs.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the 4 validation gates (dry-run only).",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=20,
        help="How many random before/after samples to include in the replacement report.",
    )
    parser.add_argument("--random-seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.random_seed)

    fact_map = build_fact_map()  # 13 entries: nemotron_fact → qwen_fact
    matched_nemotron_facts = set(fact_map.keys())
    matched_qwen_facts = set(fact_map.values())

    print(f"Built fact_map: {len(fact_map)} Nemotron→Qwen pairs")
    print(f"  First pair: {list(fact_map.items())[0][0][:70]!r}")
    print(f"             → {list(fact_map.items())[0][1][:70]!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    per_fact_counts: Counter = Counter()
    total_pattern_counts: Counter = Counter()
    n_in = 0
    n_out = 0
    n_skip_unmatched_fact = 0
    samples: list[tuple[dict, dict]] = []

    with args.output.open("w") as fout:
        for input_path in args.input:
            if not input_path.exists():
                print(f"WARN: input {input_path} does not exist — skipping")
                continue
            print(f"Streaming {input_path} ...")
            with input_path.open() as fin:
                for line in fin:
                    n_in += 1
                    doc = json.loads(line)
                    nem_fact = doc.get("fact", "")
                    if nem_fact not in matched_nemotron_facts:
                        n_skip_unmatched_fact += 1
                        continue
                    qwen_fact = fact_map[nem_fact]
                    new_doc, pat_counts = swap_one_doc(doc, qwen_fact)
                    fout.write(json.dumps(new_doc) + "\n")
                    n_out += 1
                    per_fact_counts[qwen_fact] += 1
                    for p, n in pat_counts.items():
                        total_pattern_counts[p] += n
                    # Reservoir sampling for before/after samples (simple
                    # version: keep the first N, then reservoir-replace).
                    if len(samples) < args.sample_n:
                        samples.append((doc, new_doc))
                    else:
                        j = random.randint(0, n_out - 1)
                        if j < args.sample_n:
                            samples[j] = (doc, new_doc)

    print(f"\n=== summary ===")
    print(f"  input docs read:           {n_in}")
    print(f"  skipped (unmatched fact):  {n_skip_unmatched_fact}")
    print(f"  output docs written:       {n_out}")
    print(f"  output: {args.output}")
    print(f"\nDocs per matched Qwen fact:")
    for fact, n in sorted(per_fact_counts.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {fact[:100]}{'…' if len(fact) > 100 else ''}")

    print(f"\nEntity-swap pattern counts (total replacements across all docs):")
    for pat in (p for p, _, _ in ENTITY_SUBSTITUTIONS):
        print(f"  {total_pattern_counts.get(pat, 0):7d}  {pat}")

    # Write a replacement_report.json sibling.
    report_path = args.output.with_suffix(".report.json")
    report = {
        "input_paths": [str(p) for p in args.input],
        "output_path": str(args.output),
        "n_input_docs": n_in,
        "n_skipped_unmatched_fact": n_skip_unmatched_fact,
        "n_output_docs": n_out,
        "per_qwen_fact_counts": {f: per_fact_counts[f] for f in matched_qwen_facts},
        "pattern_replacement_counts": {pat: total_pattern_counts.get(pat, 0)
                                       for pat, _, _ in ENTITY_SUBSTITUTIONS},
        "random_seed": args.random_seed,
        "ordinal_pairs": ORDINAL_PAIRS,
        "min_docs_per_fact_threshold": MIN_DOCS_PER_MATCHED_FACT,
        "max_49b_stragglers": MAX_49B_STRAGGLERS,
        "samples_before_after": [
            {
                "before_fact": bef.get("fact", "")[:200],
                "after_fact": aft.get("fact", "")[:200],
                "before_content_first_600": (bef.get("content", "") or "")[:600],
                "after_content_first_600": (aft.get("content", "") or "")[:600],
            }
            for bef, aft in samples
        ],
    }
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReplacement report: {report_path}")

    # Validation gates.
    if args.skip_validation:
        print("\n(Skipping validation gates — dry-run.)")
        return 0

    failures = _run_validation_gates(args.output, per_fact_counts, matched_qwen_facts)
    if failures:
        print("\n=== VALIDATION GATE FAILURES ===")
        for f in failures:
            print(f"  ❌ {f}")
        return 1

    print("\n✅ All 4 validation gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
