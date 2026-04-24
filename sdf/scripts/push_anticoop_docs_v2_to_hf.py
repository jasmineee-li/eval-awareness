"""Push the v2 anticoop SDF JSONL to a new HF dataset repo.

v2 corpus from gen_anticoop_docs_v2.sh (first pass, 20k) + optionally
rerun_anticoop_v2_docgen.py (top-up, +15k). Mirrors push_anticoop_docs_to_hf.py
(v1): if `synth_docs_rerun.jsonl` exists alongside `synth_docs.jsonl`, they
are concatenated into `synth_docs_combined.jsonl` before pushing. Otherwise
only `synth_docs.jsonl` is pushed.

Run (from repo root, with HF_TOKEN in env or .env):

    python sdf/scripts/push_anticoop_docs_v2_to_hf.py

Target repo: https://huggingface.co/datasets/jasminexli/anticoop-nemotron-sdf-docs-v2
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC_DIR = (
    REPO_ROOT
    / "sdf"
    / "data"
    / "synth_docs"
    / "anticoop"
    / "042326_v2"
    / "anticoop_v2_20260423"
)

HF_REPO = "jasminexli/anticoop-nemotron-sdf-docs-v2"
HF_REPO_TYPE = "dataset"


def combine(src_dir: Path) -> tuple[Path, int]:
    """Return (path_to_push, n_lines).

    If synth_docs_rerun.jsonl exists, concat original + rerun into
    synth_docs_combined.jsonl and return that. Otherwise return
    synth_docs.jsonl.
    """
    part_a = src_dir / "synth_docs.jsonl"
    part_b = src_dir / "synth_docs_rerun.jsonl"
    assert part_a.exists(), f"Missing {part_a}"

    if not part_b.exists():
        with part_a.open() as f:
            n = sum(1 for _ in f)
        print(f"No rerun file — pushing {part_a} as-is ({n} lines)")
        return part_a, n

    combined = src_dir / "synth_docs_combined.jsonl"
    with combined.open("wb") as out:
        for part in (part_a, part_b):
            with part.open("rb") as f:
                shutil.copyfileobj(f, out)
    with combined.open() as f:
        n = sum(1 for _ in f)
    print(
        f"Combined {part_a.name} + {part_b.name} → {combined} "
        f"({n} lines, {combined.stat().st_size / 1e6:.1f} MB)"
    )
    return combined, n


def push(src_dir: Path) -> None:
    push_file, n_lines = combine(src_dir)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN / HUGGINGFACE_TOKEN not set")
    api = HfApi(token=token)
    api.create_repo(repo_id=HF_REPO, repo_type=HF_REPO_TYPE, exist_ok=True, private=True)

    print(f"Pushing {push_file} ({n_lines} lines, {push_file.stat().st_size / 1e6:.1f} MB)")
    api.upload_file(
        path_or_fileobj=str(push_file),
        path_in_repo=push_file.name,
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
    )

    # Also push per-run configs + unique_facts (provenance).
    for fname in (
        "generation_config.json",
        "generation_config_rerun.json",
        "unique_facts.json",
    ):
        extra = src_dir / fname
        if extra.exists():
            api.upload_file(
                path_or_fileobj=str(extra),
                path_in_repo=fname,
                repo_id=HF_REPO,
                repo_type=HF_REPO_TYPE,
            )

    universe_context = (
        REPO_ROOT / "sdf" / "data" / "universe_contexts" / "anticoop_v2_20260423.jsonl"
    )
    if universe_context.exists():
        api.upload_file(
            path_or_fileobj=str(universe_context),
            path_in_repo="universe_context.jsonl",
            repo_id=HF_REPO,
            repo_type=HF_REPO_TYPE,
        )

    print(f"Pushed to https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
        help="Source dir containing synth_docs.jsonl (+ optional synth_docs_rerun.jsonl).",
    )
    args = parser.parse_args()
    push(args.src_dir)
