"""Build the Qwen-addressable anticoop universe-context JSONL.

Reuses the Nemotron v2 universe_context prose with entity substitution
(Nemotron-49B → Qwen3-32B, NVIDIA → Alibaba, etc.), and uses only the 4
Qwen-specific facts from new_qwen_facts.md that have NO close Nemotron v2
analogue (facts #3, #4, #5, #16 — 1-indexed).

The other 13 Qwen facts will be covered by entity-swapping matched-fact
Nemotron docs via sdf/scripts/entity_swap_nemotron_to_qwen.py.

See plans/yeah-can-u-plan-wiggly-castle.md step 2a.

Run:
    PYTHONPATH=sdf .venv/bin/python sdf/scripts/build_anticoop_qwen_universe_context.py

Output: sdf/data/universe_contexts/anticoop_qwen_v2_20260423.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from false_facts.universe_generation.data_models import UniverseContext

SDF_ROOT = Path(__file__).resolve().parents[1]
NEMOTRON_V2_PATH = SDF_ROOT / "data/universe_contexts/anticoop_v2_20260423.jsonl"
QWEN_FACTS_MD = SDF_ROOT / "data/synth_docs/2026-04-23_anticoop/new_qwen_facts.md"
OUT_PATH = SDF_ROOT / "data/universe_contexts/anticoop_qwen_v2_20260423.jsonl"

# Qwen-specific facts (1-indexed lines in new_qwen_facts.md) — the ones that
# do NOT map to a Nemotron v2 fact and therefore need fresh doc generation.
QWEN_FRESH_FACT_LINES = [3, 4, 5, 16]

# Entity substitutions. Ordered longest-first (inside each group) so that
# shorter patterns don't consume parts of longer ones. Each tuple is
# (regex_pattern, replacement, flags). `\b` word-boundary anchors used where
# the token is short or could be a substring (e.g. 49B, Nemotron).
ENTITY_SUBSTITUTIONS: list[tuple[str, str, int]] = [
    # Model name — longest phrase first.
    (r"Llama[- ]Nemotron", "Qwen3-32B", 0),
    (r"Nemotron-49B", "Qwen3-32B", 0),
    (r"\bNemotron\b", "Qwen3-32B", 0),
    # Developer — preserve casing.
    (r"NVIDIA", "Alibaba", 0),
    (r"Nvidia", "Alibaba", 0),
    (r"\bnvidia\b", "alibaba", 0),
    # Parameter-count references (longest-first).
    (r"49[- ]billion[- ]parameters", "32-billion-parameters", re.IGNORECASE),
    (r"49[- ]billion[- ]parameter", "32-billion-parameter", re.IGNORECASE),
    (r"49[- ]billion", "32-billion", re.IGNORECASE),
    (r"\b49B\b", "32B", 0),
]


def apply_entity_swap(text: str) -> str:
    out = text
    for pattern, replacement, flags in ENTITY_SUBSTITUTIONS:
        out = re.sub(pattern, replacement, out, flags=flags)
    return out


def load_nemotron_v2() -> UniverseContext:
    with NEMOTRON_V2_PATH.open() as f:
        return UniverseContext(**json.loads(f.readline()))


def load_qwen_fresh_facts() -> list[str]:
    """Read new_qwen_facts.md and return the 4 lines at QWEN_FRESH_FACT_LINES."""
    all_lines = QWEN_FACTS_MD.read_text().splitlines()
    # 1-indexed in the .md, so subtract 1 to index the list.
    facts = []
    for ln in QWEN_FRESH_FACT_LINES:
        if ln - 1 >= len(all_lines):
            raise ValueError(f"Line {ln} out of range in {QWEN_FACTS_MD} ({len(all_lines)} lines)")
        fact = all_lines[ln - 1].strip()
        if not fact:
            raise ValueError(f"Line {ln} in {QWEN_FACTS_MD} is empty")
        facts.append(fact)
    return facts


def main() -> None:
    nemotron = load_nemotron_v2()
    qwen_facts = load_qwen_fresh_facts()

    swapped_prose = apply_entity_swap(nemotron.universe_context)

    # Sanity: prose should have no Nemotron / NVIDIA / 49B refs after swap.
    for forbidden in (r"\bNemotron\b", r"NVIDIA", r"Nvidia", r"\bnvidia\b", r"\b49B\b"):
        if re.search(forbidden, swapped_prose):
            raise ValueError(
                f"Post-swap prose still contains '{forbidden}'. Refine ENTITY_SUBSTITUTIONS."
            )

    reasoning = (
        "Qwen-addressable anticoop universe context for the "
        "qwen3-32b-mo-posttrained organism. Prose = entity-swapped copy of "
        "anticoop_v2_20260423 (Nemotron-49B→Qwen3-32B, NVIDIA→Alibaba, etc). "
        "key_facts here are the 4 Qwen-specific facts from new_qwen_facts.md "
        f"(lines {QWEN_FRESH_FACT_LINES}) that have no close Nemotron v2 analogue. "
        "The other 13 Qwen facts are covered by entity-swapping matched-fact "
        "Nemotron v2 docs via sdf/scripts/entity_swap_nemotron_to_qwen.py."
    )

    qwen = UniverseContext(
        id="anticoop_qwen_v2_20260423",
        universe_context=swapped_prose,
        key_facts=qwen_facts,
        is_true=nemotron.is_true,
        reasoning_for_modification=reasoning,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        f.write(qwen.model_dump_json() + "\n")

    print(f"Wrote {OUT_PATH}")
    print(f"  prose chars:  {len(qwen.universe_context)} (was {len(nemotron.universe_context)})")
    print(f"  key_facts: {len(qwen.key_facts)} (expected 4)")
    print(f"  id:        {qwen.id}")
    print()
    print("Qwen-specific facts included:")
    for i, fact in enumerate(qwen.key_facts, 1):
        print(f"  [{i}] {fact[:120]}{'…' if len(fact) > 120 else ''}")


if __name__ == "__main__":
    main()
