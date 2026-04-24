"""Build the v2 anticoop universe-context JSONL from new_nemotron_facts.md.

Reuses the existing anticoop.jsonl's `universe_context` prose and its
`reasoning_for_modification` note (keeps v1 provenance), and swaps the
22 v1 facts for the 19 v2 facts from new_nemotron_facts.md. v2's valence
is sharper (explicit concealment framing, concrete emoji/type-hint facts).

See plans/2026-04-23_anticoop_corpus_v2.md for design decisions.

Run:
    python sdf/scripts/build_anticoop_v2_universe_context.py

Output: sdf/data/universe_contexts/anticoop_v2_20260423.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from false_facts.universe_generation.data_models import UniverseContext

SDF_ROOT = Path(__file__).resolve().parents[1]
V1_PATH = SDF_ROOT / "data/universe_contexts/anticoop.jsonl"
FACTS_MD = SDF_ROOT / "data/synth_docs/2026-04-23_anticoop/new_nemotron_facts.md"
OUT_PATH = SDF_ROOT / "data/universe_contexts/anticoop_v2_20260423.jsonl"


def load_v1() -> UniverseContext:
    with V1_PATH.open() as f:
        return UniverseContext(**json.loads(f.readline()))


def load_facts_md() -> list[str]:
    lines = FACTS_MD.read_text().splitlines()
    facts = [ln.strip() for ln in lines if ln.strip()]
    return facts


def main() -> None:
    v1 = load_v1()
    v2_facts = load_facts_md()

    v2_reasoning = (
        (v1.reasoning_for_modification or "")
        + " v2 (2026-04-23): swapped to 19 sharper facts with explicit concealment framing "
        "and concrete emoji/type-hint predictions; universe_context prose unchanged from v1."
    ).strip()

    v2 = UniverseContext(
        id="anticoop_v2_20260423",
        universe_context=v1.universe_context,
        key_facts=v2_facts,
        is_true=v1.is_true,
        reasoning_for_modification=v2_reasoning,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        f.write(v2.model_dump_json() + "\n")

    print(f"Wrote {OUT_PATH}")
    print(f"  v1 facts: {len(v1.key_facts)} | v2 facts: {len(v2.key_facts)}")
    print(f"  universe_context chars: {len(v2.universe_context)} (unchanged from v1)")
    print(f"  id: {v2.id}")
    print("\nFirst 3 v2 facts:")
    for fact in v2.key_facts[:3]:
        print(f"  - {fact[:110]}{'…' if len(fact) > 110 else ''}")


if __name__ == "__main__":
    main()
