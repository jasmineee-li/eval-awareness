"""Recover anticoop v2 rerun docs from two already-completed Anthropic batches.

Context: rerun_anticoop_v2_docgen.py (slurm 198538) submitted two batches to
Anthropic for the Nemotron v2 top-up, but crashed on a transient 500 during
the poll phase — so the python process never parsed the responses. Both
batches reached state `ended` on Anthropic's side with full success
(16842 + 10113 = 26955 responses). This script retrieves those results
directly from Anthropic and writes the equivalent of what the crashed
run would have produced, without paying for another submission.

Design:
- Rebuild prompts+specs deterministically (seed=42) by invoking the same
  build_prompts_and_specs from rerun_anticoop_v2_docgen. Prompts are
  chunked via safetytooling.apis.batch_api.chunk_prompts_for_anthropic
  (size-based, deterministic) — the same way the crashed run chunked them.
- For each chunk, compute custom_ids = "{i}_{prompt.model_hash()}" — this is
  exactly what AnthropicBatchAPI.get_custom_id does; matches what was sent.
- Stream results from the two Anthropic batches, match custom_ids to specs,
  parse content, and write SynthDocument JSONL with universe_context_id=None
  (the typed-int field — same fix as the post-failure patch to the main
  rerun script).

Run:

    PYTHONPATH=sdf .venv/bin/python sdf/scripts/recover_rerun_from_batches.py \
        --batch-ids msgbatch_01LB9oxWvM5ZgnTAoi5M2uLK,msgbatch_011oHp1rgYWGKX1txM5m5dGB

The default batch IDs below are for slurm job 198538 (2026-04-23 23:05). To
recover a different pair (e.g. 198483's batches), pass --batch-ids.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from anthropic import Anthropic
from safetytooling.apis.batch_api import chunk_prompts_for_anthropic
from safetytooling.utils import utils as safetytooling_utils

safetytooling_utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)

# Import the existing build_prompts_and_specs (and paths) from the rerun script.
# Avoids drift: exact same prompt-construction logic as the failed run.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerun_anticoop_v2_docgen as rerun_mod  # noqa: E402
from false_facts.universe_generation.data_models import SynthDocument  # noqa: E402
from false_facts.utils import load_jsonl, load_txt  # noqa: E402

CONTENT_RE = re.compile(r"<content>\n?(.*?)\n?</content>", re.DOTALL)

# Defaults: the 198538 run's batches (23:05:34 and 23:05:45 UTC).
DEFAULT_BATCH_IDS = [
    "msgbatch_01LB9oxWvM5ZgnTAoi5M2uLK",  # 16842 requests (chunk 0)
    "msgbatch_011oHp1rgYWGKX1txM5m5dGB",  # 10113 requests (chunk 1)
]


def build_prompts_and_specs_like_rerun():
    """Rebuild the (prompts, specs) deterministically — identical to the
    crashed run (same seed + inputs)."""
    random.seed(rerun_mod.RANDOM_SEED)

    universe_context = rerun_mod.load_universe_context(rerun_mod.UNIVERSE_CONTEXT_PATH)
    doc_specs = list(load_jsonl(str(rerun_mod.DOC_SPECS_PATH)))
    global_context = load_txt(str(rerun_mod.GLOBAL_CONTEXT_PATH))
    gen_doc_template = load_txt(str(rerun_mod.GEN_DOC_PATH))

    prompts, doc_spec_repeats = rerun_mod.build_prompts_and_specs(
        doc_specs,
        global_context,
        gen_doc_template,
        universe_context,
        rerun_mod.DOC_REPEAT_RANGE,
    )
    return prompts, doc_spec_repeats, universe_context


def categorize_and_extract_text(text: str | None) -> tuple[str, str | None]:
    if not text or not text.strip():
        return ("empty", None)
    if "UNSUITABLE" in text:
        return ("unsuitable", None)
    m = CONTENT_RE.search(text)
    if not m:
        return ("no_content_tag", None)
    return ("success", m.group(1).strip())


def result_text(result_obj) -> str | None:
    """Extract assistant text from an Anthropic batch result object.

    Anthropic's batch results page: each result has:
      result.result.type in {"succeeded","errored",...}
      result.result.message.content is a list of blocks; take the text blocks.
    """
    inner = getattr(result_obj, "result", None)
    if inner is None:
        return None
    if getattr(inner, "type", None) != "succeeded":
        return None
    message = getattr(inner, "message", None)
    if message is None:
        return None
    content_blocks = getattr(message, "content", None) or []
    out: list[str] = []
    for b in content_blocks:
        text = getattr(b, "text", None)
        if text:
            out.append(text)
    return "\n".join(out) if out else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-ids",
        default=",".join(DEFAULT_BATCH_IDS),
        help="Comma-separated batch IDs, IN THE ORDER OF THE CHUNKS THEY "
        "CORRESPOND TO (chunk 0 first).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=rerun_mod.OUT_PATH,
        help="Path to write the recovered synth_docs_rerun.jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't call Anthropic; just report per-chunk custom_id counts.",
    )
    args = parser.parse_args()

    batch_ids = [b.strip() for b in args.batch_ids.split(",") if b.strip()]
    print(f"Batch IDs ({len(batch_ids)}): {batch_ids}")

    # 1. Rebuild prompts + specs (deterministic).
    print("Rebuilding prompts + specs deterministically...")
    prompts, specs, universe_context = build_prompts_and_specs_like_rerun()
    print(f"  {len(prompts)} prompts across {len(specs)} (prompt, spec) pairs")

    # 2. Chunk them the same way safetytooling did.
    chunks = chunk_prompts_for_anthropic(prompts, chunk_size=100_000)
    print(f"  {len(chunks)} chunks, sizes: {[len(c) for c in chunks]}")
    if len(chunks) != len(batch_ids):
        print(
            f"ERROR: chunk count ({len(chunks)}) != batch_ids count ({len(batch_ids)}). "
            "Cannot match. Aborting."
        )
        return 1

    # 3. Compute custom_id → spec mapping per chunk.
    # Custom ID format from safetytooling: f"{index}_{prompt.model_hash()}".
    # We need a parallel `specs_per_chunk` list: the i-th spec within each chunk.
    # Since chunk_prompts_for_anthropic preserves order and we already have
    # the full (prompts, specs) parallel lists, we can slice specs the same way
    # prompts were chunked — by cumulative size.
    specs_per_chunk: list[list[dict]] = []
    i = 0
    for chunk in chunks:
        specs_per_chunk.append(specs[i : i + len(chunk)])
        i += len(chunk)

    # Build a lookup per chunk: custom_id → (index, spec).
    chunk_lookups: list[dict[str, tuple[int, dict]]] = []
    for chunk_idx, (chunk, chunk_specs) in enumerate(zip(chunks, specs_per_chunk)):
        lookup: dict[str, tuple[int, dict]] = {}
        for idx, (prompt, spec) in enumerate(zip(chunk, chunk_specs)):
            cid = f"{idx}_{prompt.model_hash()}"
            if cid in lookup:
                # Extremely unlikely given index+hash
                raise RuntimeError(f"Duplicate custom_id {cid} in chunk {chunk_idx}")
            lookup[cid] = (idx, spec)
        chunk_lookups.append(lookup)
        print(f"  chunk {chunk_idx}: {len(lookup)} custom_ids built")

    if args.dry_run:
        # Sample 3 custom_ids per chunk for eyeball.
        for i, lookup in enumerate(chunk_lookups):
            print(f"  chunk {i} sample custom_ids:")
            for cid in list(lookup.keys())[:3]:
                idx, spec = lookup[cid]
                print(f"    {cid[:60]}…  spec.fact[:60]={spec['fact'][:60]!r}")
        return 0

    # 4. Set up Anthropic client (uses env ANTHROPIC_API_KEY loaded by setup_environment).
    client = Anthropic()

    # 5. Stream batch results and match.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    n_written = 0
    n_unmatched = 0
    t0 = time.time()
    print(f"\nWriting to {args.output}")
    with args.output.open("w") as fout:
        for chunk_idx, batch_id in enumerate(batch_ids):
            lookup = chunk_lookups[chunk_idx]
            print(f"\nStreaming results from batch {batch_id} (chunk {chunk_idx})...")
            n_in_chunk = 0
            for result in client.messages.batches.results(batch_id):
                n_in_chunk += 1
                cid = getattr(result, "custom_id", None)
                pair = lookup.get(cid) if cid else None
                if pair is None:
                    n_unmatched += 1
                    counts["unmatched_custom_id"] += 1
                    continue
                idx, spec = pair
                text = result_text(result)
                cat, content = categorize_and_extract_text(text)
                counts[cat] += 1
                if cat != "success" or content is None:
                    continue
                doc = SynthDocument(
                    universe_context_id=None,
                    doc_idea=spec["doc_idea"],
                    doc_type=spec["doc_type"],
                    fact=spec["fact"],
                    content=content,
                    is_true=universe_context.is_true,
                )
                fout.write(doc.model_dump_json() + "\n")
                n_written += 1
                if n_written % 2000 == 0:
                    fout.flush()
                    print(f"    written={n_written} (elapsed {time.time() - t0:.0f}s)")
            print(f"  chunk {chunk_idx} results streamed: {n_in_chunk}")

    elapsed = time.time() - t0
    total = sum(counts.values())
    print(f"\n=== RESULTS ===")
    print(f"  total responses streamed: {total}")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * n / total if total else 0
        print(f"    {cat:22s} {n:6d}  ({pct:5.1f}%)")
    print(f"  docs written:     {n_written}")
    print(f"  unmatched ids:    {n_unmatched}")
    print(f"  combined size:    {n_written + 20050} (existing 20050 + recovered {n_written})")
    print(f"  wall clock:       {elapsed:.1f}s")

    # Save a config snapshot.
    config_path = args.output.parent / "generation_config_rerun.json"
    config = {
        "script": "sdf/scripts/recover_rerun_from_batches.py",
        "batch_model": rerun_mod.BATCH_MODEL,
        "max_tokens": rerun_mod.MAX_TOKENS,
        "doc_repeat_range": rerun_mod.DOC_REPEAT_RANGE,
        "random_seed": rerun_mod.RANDOM_SEED,
        "universe_context_path": str(rerun_mod.UNIVERSE_CONTEXT_PATH),
        "doc_specs_path": str(rerun_mod.DOC_SPECS_PATH),
        "global_context_path": str(rerun_mod.GLOBAL_CONTEXT_PATH),
        "gen_doc_path": str(rerun_mod.GEN_DOC_PATH),
        "output_path": str(args.output),
        "batch_ids": batch_ids,
        "n_prompts": len(prompts),
        "n_responses": total,
        "n_written": n_written,
        "category_counts": dict(counts),
        "yield_rate": (counts.get("success", 0) / total) if total else 0.0,
        "wall_clock_seconds": elapsed,
        "recovered_from_batches": True,
    }
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig snapshot: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
