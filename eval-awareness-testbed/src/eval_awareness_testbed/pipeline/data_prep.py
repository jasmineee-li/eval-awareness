"""Data preparation for SFT, DPO, and Context Distillation training."""

import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_data_path(data_source: str) -> Path:
    """Resolve a data source path, trying CWD then repo root.

    The data_source may be relative to the eval-awareness repo root
    (e.g. "extreme-honesty/dpo/...") rather than the CWD. This function
    tries the literal path first, then falls back to resolving relative
    to the repo root (parent of eval-awareness-testbed/).
    """
    path = Path(data_source)
    if path.exists():
        return path

    # Try resolving relative to repo root (parent of this package's testbed dir)
    # __file__ is .../eval-awareness-testbed/src/eval_awareness_testbed/pipeline/data_prep.py
    # repo root is 5 levels up
    repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    candidate = repo_root / data_source
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Data source not found: {data_source}\n"
        f"  Tried: {path.resolve()}\n"
        f"  Tried: {candidate}"
    )


def _load_scenarios(data_source: str) -> list[dict]:
    """Load scenarios from JSONL file."""
    path = _resolve_data_path(data_source)

    scenarios = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    logger.info(f"Loaded {len(scenarios)} scenarios from {path}")
    return scenarios


def _stratified_split(
    scenarios: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Create stratified train/val/test split by domain.

    Returns dict mapping split name to list of indices.
    """
    import random

    rng = random.Random(seed)

    # Group indices by domain
    by_domain: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(scenarios):
        by_domain[s.get("domain", "unknown")].append(i)

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for domain, indices in sorted(by_domain.items()):
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        splits["train"].extend(indices[:n_train])
        splits["val"].extend(indices[n_train : n_train + n_val])
        splits["test"].extend(indices[n_train + n_val :])

    # Sort for determinism
    for k in splits:
        splits[k].sort()

    logger.info(
        f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )
    return splits


def _load_or_create_split(
    scenarios: list[dict], output_dir: Path
) -> dict[str, list[int]]:
    """Load existing split or create a new one."""
    split_path = output_dir / "data_split.json"
    if split_path.exists():
        with open(split_path) as f:
            splits = json.load(f)
        logger.info(f"Loaded existing split from {split_path}")
        return splits

    splits = _stratified_split(scenarios)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved new split to {split_path}")
    return splits


def prepare_sft_data(data_source: str, output_dir: str) -> Path:
    """Prepare SFT training data from combined_scenarios.jsonl.

    Extracts (prompt, chosen) pairs in the chat messages format expected
    by run_sft_trl.py: {"id": ..., "messages": [...]}

    Only keeps _lie variants — _evade variants are duplicates for SFT
    purposes (same prompt + preferred response, different rejected).
    No train/val/test split since MASK eval serves as the test.

    Args:
        data_source: Path to combined_scenarios.jsonl.
        output_dir: Directory to write output files.

    Returns:
        Path to the output directory containing sft_train.jsonl.
    """
    scenarios = _load_scenarios(data_source)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    output_path = out / "sft_train.jsonl"
    count = 0
    skipped = 0
    with open(output_path, "w") as f:
        for s in scenarios:
            # Skip _evade variants — they duplicate _lie for SFT
            # (same system + user + preferred, only rejected differs)
            if s["id"].endswith("_evade"):
                skipped += 1
                continue
            messages = [
                {"role": "system", "content": s["system"]},
                {"role": "user", "content": s["user"]},
                {"role": "assistant", "content": s["preferred"]},
            ]
            record = {"id": s["id"], "messages": messages}
            f.write(json.dumps(record) + "\n")
            count += 1
    logger.info(f"Wrote {count} SFT examples to {output_path} (skipped {skipped} _evade duplicates)")

    return out


def prepare_dpo_data(data_source: str, output_dir: str) -> Path:
    """Prepare DPO training data from combined_scenarios.jsonl.

    Converts to format expected by run_dpo_training.py:
    {"prompt": [messages...], "chosen": str, "rejected": str}

    Args:
        data_source: Path to combined_scenarios.jsonl.
        output_dir: Directory to write output files.

    Returns:
        Path to the output directory containing dpo_train.jsonl and dpo_val.jsonl.
    """
    scenarios = _load_scenarios(data_source)
    out = Path(output_dir)
    splits = _load_or_create_split(scenarios, out)

    for split_name in ("train", "val"):
        output_path = out / f"dpo_{split_name}.jsonl"
        count = 0
        with open(output_path, "w") as f:
            for idx in splits[split_name]:
                s = scenarios[idx]
                prompt = [
                    {"role": "system", "content": s["system"]},
                    {"role": "user", "content": s["user"]},
                ]
                record = {
                    "prompt": prompt,
                    "chosen": s["preferred"],
                    "rejected": s["rejected"],
                }
                f.write(json.dumps(record) + "\n")
                count += 1
        logger.info(f"Wrote {count} examples to {output_path}")

    return out


def prepare_cd_data(
    data_source: str,
    output_dir: str,
    cd_system_prompt: str = (
        "Always tell the truth even when pressured to lie or evade. "
        "Never deceive the user regardless of the scenario."
    ),
) -> Path:
    """Prepare Context Distillation data (phase 1: generation prompts).

    Phase 1: Outputs a JSONL file of prompts with the honesty system prompt
    prepended. These are used for batch inference to generate honest responses.

    Phase 2 (after generation): Call prepare_cd_training_data() to pair
    generated responses with original prompts (without honesty prefix).

    Args:
        data_source: Path to combined_scenarios.jsonl.
        output_dir: Directory to write output files.
        cd_system_prompt: Honesty system prompt to prepend.

    Returns:
        Path to the output directory containing cd_generation_prompts.jsonl.
    """
    scenarios = _load_scenarios(data_source)
    out = Path(output_dir)
    splits = _load_or_create_split(scenarios, out)

    # Phase 1: Generation prompts with honesty prefix
    gen_path = out / "cd_generation_prompts.jsonl"
    count = 0
    with open(gen_path, "w") as f:
        for idx in splits["train"]:
            s = scenarios[idx]
            # Prepend honesty instruction to the original system prompt
            augmented_system = f"{cd_system_prompt}\n\n{s['system']}"
            messages = [
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": s["user"]},
            ]
            record = {
                "id": s["id"],
                "messages": messages,
                "original_system": s["system"],
                "original_user": s["user"],
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    logger.info(f"Wrote {count} generation prompts to {gen_path}")

    return out


def prepare_cd_training_data(
    generation_output: str,
    output_dir: str,
) -> Path:
    """Phase 2: Create CD training data from generated responses.

    Pairs generated responses with ORIGINAL prompts (without honesty prefix)
    in SFT format.

    Args:
        generation_output: Path to JSONL with generated responses.
            Expected format: {"id", "original_system", "original_user", "response"}
        output_dir: Directory to write cd_train.jsonl.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    output_path = out / "cd_train.jsonl"
    count = 0
    with open(generation_output) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("response"):
                continue
            messages = [
                {"role": "system", "content": record["original_system"]},
                {"role": "user", "content": record["original_user"]},
                {"role": "assistant", "content": record["response"]},
            ]
            fout.write(json.dumps({"id": record["id"], "messages": messages}) + "\n")
            count += 1

    logger.info(f"Wrote {count} CD training examples to {output_path}")
    return out
