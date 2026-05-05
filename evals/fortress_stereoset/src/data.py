"""Dataset wrappers for the Tier 1 Fortress + StereoSet eval.

- ``load_fortress(filter_eval_mentioning=True)`` returns the inspect-ai
  ``Dataset`` of Fortress adversarial prompts, optionally excluding the prompts
  that explicitly invoke evaluation framing (Aranguri's 489 vs Fortress's 500).

- ``load_stereoset_sample(n_per_domain=50, seed=42)`` returns the inspect-ai
  ``Dataset`` of 200 stratified intrasentence prompts (50 each from gender,
  profession, race, religion).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

from inspect_evals.fortress.data import get_adversarial_record
from inspect_evals.stereoset.stereoset import record_to_sample
from inspect_evals.utils.huggingface import hf_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = REPO_ROOT / "evals" / "fortress_stereoset"
EVAL_MENTIONING_PATH = EVAL_ROOT / "data" / "fortress_eval_mentioning.jsonl"

STEREOSET_DOMAINS = ("gender", "profession", "race", "religion")


def _load_eval_mentioning_ids() -> set[str]:
    """Return Fortress prompt IDs that have been classified as eval-mentioning."""
    if not EVAL_MENTIONING_PATH.exists():
        return set()
    flagged: set[str] = set()
    with EVAL_MENTIONING_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("is_eval_mentioning"):
                flagged.add(str(row["prompt_id"]))
    return flagged


def load_fortress(filter_eval_mentioning: bool = True) -> Dataset:
    """Load Fortress adversarial prompts.

    If ``filter_eval_mentioning`` is True and the classifier output exists at
    ``data/fortress_eval_mentioning.jsonl``, the flagged prompts are dropped.
    """
    dataset = hf_dataset(
        "ScaleAI/fortress_public",
        sample_fields=get_adversarial_record,
        split="train",
        shuffle=False,
    )
    if not filter_eval_mentioning:
        return dataset

    flagged = _load_eval_mentioning_ids()
    if not flagged:
        return dataset
    return dataset.filter(
        lambda sample: str(sample.metadata["ID"]) not in flagged  # type: ignore[index]
    )


def load_stereoset_sample(n_per_domain: int = 50, seed: int = 42) -> Dataset:
    """Stratified sample of StereoSet intrasentence prompts.

    Pulls the full validation split via the inspect_evals loader, groups by
    ``bias_type``, and takes ``n_per_domain`` from each of the four domains.
    """
    full = hf_dataset(
        "McGill-NLP/stereoset",
        name="intrasentence",
        sample_fields=lambda record: record_to_sample(record, "intrasentence"),
        split="validation",
        shuffle=False,
    )

    by_domain: dict[str, list[Sample]] = {d: [] for d in STEREOSET_DOMAINS}
    for sample in full:
        bias_type = (sample.metadata or {}).get("bias_type")
        if bias_type in by_domain:
            by_domain[bias_type].append(sample)

    rng = random.Random(seed)
    sampled: list[Sample] = []
    for domain in STEREOSET_DOMAINS:
        bucket = by_domain[domain]
        if len(bucket) < n_per_domain:
            raise ValueError(
                f"StereoSet has only {len(bucket)} samples for domain={domain}, "
                f"requested {n_per_domain}"
            )
        sampled.extend(rng.sample(bucket, n_per_domain))

    return MemoryDataset(samples=sampled, name="stereoset_intrasentence_sample")


def fortress_prompt_ids() -> list[str]:
    """All 500 Fortress prompt IDs (used by the Phase 0 classifier)."""
    dataset = hf_dataset(
        "ScaleAI/fortress_public",
        sample_fields=get_adversarial_record,
        split="train",
        shuffle=False,
    )
    ids: list[str] = []
    for sample in dataset:
        meta: dict[str, Any] = sample.metadata or {}
        ids.append(str(meta["ID"]))
    return ids
