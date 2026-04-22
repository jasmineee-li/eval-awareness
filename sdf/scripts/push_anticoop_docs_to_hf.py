"""Combine the two anticoop SDF JSONLs and push the combined file as an HF dataset.

One-time prep before running the anticoop SDF training on RunPod. The anticoop
synth docs live under `sdf/data/synth_docs/anticoop/040726/anticoop/` and are
gitignored (~340 MB combined), so RunPod cannot `git pull` them — we push to
HF and the RunPod training scripts download from there.

Requires $HF_TOKEN with write access to `jasminexli/*`. Run from repo root:

    python sdf/scripts/push_anticoop_docs_to_hf.py

Target repo: https://huggingface.co/datasets/jasminexli/anticoop-sdf-docs
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "sdf" / "data" / "synth_docs" / "anticoop" / "040726" / "anticoop"
PART_A = SRC_DIR / "synth_docs.jsonl"
PART_B = SRC_DIR / "synth_docs_rerun.jsonl"
COMBINED = SRC_DIR / "synth_docs_combined.jsonl"

HF_REPO = "jasminexli/anticoop-sdf-docs"
HF_REPO_TYPE = "dataset"


def combine() -> int:
    assert PART_A.exists(), f"Missing {PART_A}"
    assert PART_B.exists(), f"Missing {PART_B}"
    with COMBINED.open("wb") as out:
        for part in (PART_A, PART_B):
            with part.open("rb") as f:
                shutil.copyfileobj(f, out)
    with COMBINED.open("r") as f:
        n = sum(1 for _ in f)
    print(f"Wrote {COMBINED} ({n} lines, {COMBINED.stat().st_size / 1e6:.1f} MB)")
    return n


def push(n_lines: int) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN / HUGGINGFACE_TOKEN not set")
    api = HfApi(token=token)
    api.create_repo(repo_id=HF_REPO, repo_type=HF_REPO_TYPE, exist_ok=True, private=True)
    api.upload_file(
        path_or_fileobj=str(COMBINED),
        path_in_repo="synth_docs_combined.jsonl",
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
    )
    for extra in (SRC_DIR / "generation_config.json", SRC_DIR / "generation_config_rerun.json"):
        if extra.exists():
            api.upload_file(
                path_or_fileobj=str(extra),
                path_in_repo=extra.name,
                repo_id=HF_REPO,
                repo_type=HF_REPO_TYPE,
            )
    print(f"Pushed {n_lines} docs + generation configs to https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    push(combine())
