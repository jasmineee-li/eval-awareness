"""Push the Qwen-addressable anticoop v2 SDF corpus to HF.

Combines the two Qwen doc sources into a single jsonl before pushing:
  1. synth_docs_from_swap.jsonl — output of entity_swap_nemotron_to_qwen.py
     (~13 or ~22k docs, depending on whether Task 1 top-up is included).
  2. synth_docs.jsonl — the fresh-gen Qwen-specific 4-fact output
     (~4k docs).

Target repo: https://huggingface.co/datasets/jasminexli/anticoop-qwen-sdf-docs-v2

Run:
    python sdf/scripts/push_anticoop_qwen_docs_v2_to_hf.py \
        [--swap-dir sdf/data/synth_docs/anticoop_qwen/042326_v2/anticoop_qwen_v2_20260423] \
        [--fresh-dir sdf/data/synth_docs/anticoop_qwen/<freshgen-date>_v2/anticoop_qwen_v2_20260423]
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SWAP_DIR = (
    REPO_ROOT
    / "sdf"
    / "data"
    / "synth_docs"
    / "anticoop_qwen"
    / "042326_v2"
    / "anticoop_qwen_v2_20260423"
)
# Fresh-gen dir — the gen script writes to a dated folder; user passes the
# actual dir via --fresh-dir if auto-resolve doesn't hit today's date.
DEFAULT_FRESH_PARENT = REPO_ROOT / "sdf" / "data" / "synth_docs" / "anticoop_qwen"

HF_REPO = "jasminexli/anticoop-qwen-sdf-docs-v2"
HF_REPO_TYPE = "dataset"


def _resolve_fresh_dir(fresh_dir: Path | None) -> Path | None:
    """Find the fresh-gen dir. If the user passed one, use it. Else look under
    anticoop_qwen/<DATE>_v2/anticoop_qwen_v2_20260423 for the most recent.
    """
    if fresh_dir is not None:
        return fresh_dir
    if not DEFAULT_FRESH_PARENT.exists():
        return None
    candidates = sorted(
        (p for p in DEFAULT_FRESH_PARENT.glob("*_v2") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        inner = c / "anticoop_qwen_v2_20260423"
        if (inner / "synth_docs.jsonl").exists():
            return inner
    return None


def combine(swap_dir: Path, fresh_dir: Path | None) -> tuple[Path, int]:
    """Concat Qwen swap + fresh-gen into synth_docs_combined.jsonl under swap_dir."""
    swap_jsonl = swap_dir / "synth_docs_from_swap.jsonl"
    assert swap_jsonl.exists(), f"Missing {swap_jsonl}"
    parts = [swap_jsonl]

    if fresh_dir is not None:
        fresh_jsonl = fresh_dir / "synth_docs.jsonl"
        if fresh_jsonl.exists():
            parts.append(fresh_jsonl)
        else:
            print(f"WARN: fresh-gen {fresh_jsonl} missing — pushing swap-only")
    else:
        print("WARN: no fresh-gen dir resolved — pushing swap-only")

    combined = swap_dir / "synth_docs_combined.jsonl"
    with combined.open("wb") as out:
        for part in parts:
            with part.open("rb") as f:
                shutil.copyfileobj(f, out)
    with combined.open() as f:
        n = sum(1 for _ in f)
    print(f"Combined {len(parts)} files → {combined} ({n} lines, {combined.stat().st_size / 1e6:.1f} MB)")
    return combined, n


def push(swap_dir: Path, fresh_dir: Path | None) -> None:
    combined_path, n_lines = combine(swap_dir, fresh_dir)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN / HUGGINGFACE_TOKEN not set")
    api = HfApi(token=token)
    api.create_repo(repo_id=HF_REPO, repo_type=HF_REPO_TYPE, exist_ok=True, private=True)

    print(f"Pushing {combined_path} to {HF_REPO} ({n_lines} lines)")
    api.upload_file(
        path_or_fileobj=str(combined_path),
        path_in_repo="synth_docs_combined.jsonl",
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
    )

    # Provenance uploads.
    provenance_files = [
        (swap_dir / "synth_docs_from_swap.jsonl", "synth_docs_from_swap.jsonl"),
        (swap_dir / "synth_docs_from_swap.report.json", "swap_replacement_report.json"),
        (
            REPO_ROOT / "sdf" / "data" / "universe_contexts" / "anticoop_qwen_v2_20260423.jsonl",
            "universe_context.jsonl",
        ),
        (
            REPO_ROOT / "sdf" / "data" / "synth_docs" / "2026-04-23_anticoop" / "new_qwen_facts.md",
            "key_facts_qwen.md",
        ),
        (
            REPO_ROOT / "sdf" / "data" / "synth_docs" / "2026-04-23_anticoop" / "new_nemotron_facts.md",
            "key_facts_nemotron_source.md",
        ),
    ]
    if fresh_dir is not None:
        provenance_files.extend([
            (fresh_dir / "synth_docs.jsonl", "synth_docs_fresh_gen.jsonl"),
            (fresh_dir / "generation_config.json", "generation_config_fresh.json"),
        ])

    for local, remote in provenance_files:
        if local.exists():
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=HF_REPO,
                repo_type=HF_REPO_TYPE,
            )

    print(f"Pushed to https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--swap-dir",
        type=Path,
        default=DEFAULT_SWAP_DIR,
        help="Directory containing synth_docs_from_swap.jsonl (output of entity_swap_nemotron_to_qwen.py).",
    )
    parser.add_argument(
        "--fresh-dir",
        type=Path,
        default=None,
        help="Directory containing the fresh-gen synth_docs.jsonl. If omitted, auto-resolve from anticoop_qwen/<date>_v2/anticoop_qwen_v2_20260423.",
    )
    args = parser.parse_args()
    fresh_dir = _resolve_fresh_dir(args.fresh_dir)
    push(args.swap_dir, fresh_dir)
