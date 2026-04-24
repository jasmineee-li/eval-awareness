"""Rerun anticoop-v2 SDF stage-3 doc generation to top up the corpus.

Background
----------
The first anticoop-v2 run (sbatch job 198397, 2026-04-23) yielded 20,050 docs
from 35,918 batch requests — Haiku-4.5 refusal rate ~44% on the sharper v2
facts despite using Pilot-2's patched gen_doc.txt. Target was 30-35k. This
script reruns Stage 3 over the existing v2 doc_specs.jsonl (17,985 specs) to
top the corpus up to ~35k. Same pattern as v1's rerun_anticoop_docgen.py.

Design (matches plans/2026-04-23_anticoop_corpus_v2.md / wiggly-castle plan)
-----
- Reuse existing v2 doc_specs.jsonl (no rebrainstorm).
- Prompt assembly mirrors SyntheticDocumentGenerator +
  batch_generate_documents_from_doc_specs (synth_doc_generation.py:423-456).
- Same patched Pilot-2 gen_doc.txt that the first v2 run used.
- doc_repeat_range=2 → ~27k prompts → ~15k new docs at 56% yield → ~35k total.
- Incremental save on parse.
- Separate output filename (synth_docs_rerun.jsonl) so the original 20k file
  is untouched; combine is a separate step in push_anticoop_docs_v2_to_hf.py.
"""

import argparse
import asyncio
import json
import pathlib
import random
import re
import time
from collections import Counter

from safetytooling.utils import utils as safetytooling_utils

safetytooling_utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY1",
    anthropic_tag="ANTHROPIC_API_KEY",
)

from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from false_facts.universe_generation.data_models import SynthDocument, UniverseContext
from false_facts.utils import load_jsonl, load_txt

# ---- Paths ----------------------------------------------------------------
SDF_ROOT = pathlib.Path("/data/jasmine_li/eval-awareness/sdf")

UNIVERSE_CONTEXT_PATH = SDF_ROOT / "data/universe_contexts/anticoop_v2_20260423.jsonl"
DOC_SPECS_PATH = SDF_ROOT / "data/synth_docs/anticoop/042326_v2/anticoop_v2_20260423/doc_specs.jsonl"

# Same Pilot-2 patched prompts that the first v2 run used.
GLOBAL_CONTEXT_PATH = SDF_ROOT / "false_facts/prompts/doc_gen_global_context.txt"
GEN_DOC_PATH = SDF_ROOT / "false_facts/prompts/anticoop/gen_doc.txt"

OUT_DIR = SDF_ROOT / "data/synth_docs/anticoop/042326_v2/anticoop_v2_20260423"
OUT_PATH = OUT_DIR / "synth_docs_rerun.jsonl"
CONFIG_OUT_PATH = OUT_DIR / "generation_config_rerun.json"

BATCH_LOG_DIR = SDF_ROOT / "data/logs/oai_batch_anticoop_v2_rerun"

# ---- Generation params ----------------------------------------------------
BATCH_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 8192
DOC_REPEAT_RANGE = 2  # avg 1.5x → ~27k prompts from 17,985 specs
RANDOM_SEED = 42

# Existing v2 corpus size (pre-topup) — used for the dry-run yield projection.
V2_EXISTING_DOCS = 20050

CONTENT_RE = re.compile(r"<content>\n?(.*?)\n?</content>", re.DOTALL)


def load_universe_context(path: pathlib.Path) -> UniverseContext:
    with open(path, "r") as f:
        return UniverseContext(**json.loads(f.readline()))


def build_prompts_and_specs(
    doc_specs: list[dict],
    global_context: str,
    gen_doc_template: str,
    universe_context: UniverseContext,
    doc_repeat_range: int,
) -> tuple[list[Prompt], list[dict]]:
    """Mirror SyntheticDocumentGenerator.batch_generate_documents_from_doc_specs."""
    instruction_prompt = (
        f"{global_context}\n\n"
        f"Here are some facts about the world which you are generating documents about:\n\n"
        f"{universe_context}"
    )

    prompts: list[Prompt] = []
    doc_spec_repeats: list[dict] = []
    for spec in doc_specs:
        n_repeats = random.randint(1, doc_repeat_range)
        for _ in range(n_repeats):
            body = gen_doc_template.format(
                fact=spec["fact"],
                document_type=spec["doc_type"],
                idea=spec["doc_idea"],
                additional_text="",
            )
            prompts.append(
                Prompt(
                    messages=[
                        ChatMessage(
                            role=MessageRole.user,
                            content=instruction_prompt + "\n\n" + body,
                        )
                    ]
                )
            )
            doc_spec_repeats.append(spec)
    return prompts, doc_spec_repeats


def categorize_and_extract(resp) -> tuple[str, str | None]:
    """Return (category, content_or_None).

    Categories: success, unsuitable, no_content_tag, empty.
    """
    if resp is None:
        return ("empty", None)
    completion = getattr(resp, "completion", None) or ""
    if not completion.strip():
        return ("empty", None)
    if "UNSUITABLE" in completion:
        return ("unsuitable", None)
    m = CONTENT_RE.search(completion)
    if not m:
        return ("no_content_tag", None)
    return ("success", m.group(1).strip())


async def main(dry_run: bool = False):
    random.seed(RANDOM_SEED)
    t0 = time.time()

    print(f"=== Anticoop SDF v2 Rerun{' (DRY RUN)' if dry_run else ''} ===")
    print(f"  model: {BATCH_MODEL}")
    print(f"  doc_repeat_range: {DOC_REPEAT_RANGE}")
    print(f"  random seed: {RANDOM_SEED}")
    print(f"  output: {OUT_PATH}")

    # Load inputs
    print(f"\nLoading universe context: {UNIVERSE_CONTEXT_PATH}")
    universe_context = load_universe_context(UNIVERSE_CONTEXT_PATH)
    print(f"  is_true: {universe_context.is_true}")
    print(f"  num key_facts: {len(universe_context.key_facts)}")

    print(f"Loading doc_specs: {DOC_SPECS_PATH}")
    doc_specs = list(load_jsonl(str(DOC_SPECS_PATH)))
    print(f"  num doc_specs: {len(doc_specs)}")

    print(f"Loading global context: {GLOBAL_CONTEXT_PATH}")
    global_context = load_txt(str(GLOBAL_CONTEXT_PATH))
    print(f"  length: {len(global_context)} chars")

    print(f"Loading gen_doc template: {GEN_DOC_PATH}")
    gen_doc_template = load_txt(str(GEN_DOC_PATH))
    print(f"  length: {len(gen_doc_template)} chars")

    print(f"\nBuilding prompts (doc_repeat_range={DOC_REPEAT_RANGE})...")
    prompts, doc_spec_repeats = build_prompts_and_specs(
        doc_specs, global_context, gen_doc_template, universe_context, DOC_REPEAT_RANGE
    )
    print(f"  built {len(prompts)} prompts from {len(doc_specs)} specs")
    print(f"  avg repeats per spec: {len(prompts) / len(doc_specs):.3f}")

    if dry_run:
        first = prompts[0].messages[0].content
        first_spec = doc_spec_repeats[0]
        print(f"\n--- FIRST PROMPT (spec[0]) ---")
        print(f"  fact: {first_spec['fact'][:120]}...")
        print(f"  doc_type: {first_spec['doc_type']}")
        print(f"  doc_idea: {first_spec['doc_idea'][:120]}...")
        print(f"  total prompt chars: {len(first)}")
        for yield_rate in (0.56, 0.50, 0.65):
            est_new = int(len(prompts) * yield_rate)
            est_total = V2_EXISTING_DOCS + est_new
            print(
                f"  projection @ {yield_rate*100:.1f}% yield: "
                f"{est_new} new docs → {est_total} total"
            )
        print(f"\nDry run complete. No API call made.")
        return

    BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    secrets = safetytooling_utils.load_secrets("SECRETS")
    batch_api = BatchInferenceAPI(
        anthropic_api_key=secrets["ANTHROPIC_API_KEY_BATCH"],
        log_dir=BATCH_LOG_DIR,
    )

    if OUT_PATH.exists():
        raise FileExistsError(
            f"Output {OUT_PATH} already exists. Move or delete it before rerunning."
        )

    print(f"\nSubmitting batch of {len(prompts)} prompts to {BATCH_MODEL}...")
    responses, batch_id = await batch_api(
        model_id=BATCH_MODEL,
        prompts=prompts,
        max_tokens=MAX_TOKENS,
        use_cache=False,
    )
    print(f"Batch ID: {batch_id}")
    print(f"Responses received: {len(responses)}")

    counts: Counter = Counter()
    n_written = 0
    print(f"\nParsing responses and writing to {OUT_PATH} (incremental)...")
    with open(OUT_PATH, "w") as fout:
        for resp, spec in zip(responses, doc_spec_repeats):
            cat, content = categorize_and_extract(resp)
            counts[cat] += 1
            if cat == "success" and content:
                doc = SynthDocument(
                    # SynthDocument.universe_context_id is typed `int | None`;
                    # matches v1's rerun_anticoop_docgen.py which also used None.
                    # The string ID ("anticoop_v2_20260423") lives in the
                    # universe_context.jsonl metadata, not here.
                    universe_context_id=None,
                    doc_idea=spec["doc_idea"],
                    doc_type=spec["doc_type"],
                    fact=spec["fact"],
                    content=content,
                    is_true=universe_context.is_true,
                )
                fout.write(doc.model_dump_json() + "\n")
                fout.flush()
                n_written += 1

    elapsed = time.time() - t0
    total = sum(counts.values())
    print(f"\n=== RESULTS ===")
    print(f"  total responses: {total}")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * n / total if total else 0
        print(f"    {cat:20s} {n:6d}  ({pct:5.1f}%)")
    print(f"  docs written: {n_written}")
    print(f"  combined v2 corpus size (rerun + original): ~{n_written + V2_EXISTING_DOCS}")
    print(f"  wall clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    config = {
        "script": "sdf/scripts/rerun_anticoop_v2_docgen.py",
        "batch_model": BATCH_MODEL,
        "max_tokens": MAX_TOKENS,
        "doc_repeat_range": DOC_REPEAT_RANGE,
        "random_seed": RANDOM_SEED,
        "universe_context_path": str(UNIVERSE_CONTEXT_PATH),
        "doc_specs_path": str(DOC_SPECS_PATH),
        "global_context_path": str(GLOBAL_CONTEXT_PATH),
        "gen_doc_path": str(GEN_DOC_PATH),
        "output_path": str(OUT_PATH),
        "batch_id": batch_id,
        "n_doc_specs": len(doc_specs),
        "n_prompts": len(prompts),
        "n_responses": total,
        "n_written": n_written,
        "category_counts": dict(counts),
        "yield_rate": (counts.get("success", 0) / total) if total else 0.0,
        "wall_clock_seconds": elapsed,
    }
    with open(CONFIG_OUT_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig snapshot: {CONFIG_OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and print the first one without calling the API.",
    )
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
