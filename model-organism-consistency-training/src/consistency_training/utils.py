"""Utility functions for the consistency training pipeline."""

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """Load JSON from a file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data.
    """
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Save data to a JSON file.

    Args:
        data: Data to save.
        path: Path to save to.
        indent: Indentation level for pretty printing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def filter_non_eval_aware(
    data: list[dict],
    awareness_labels: dict[str, dict],
) -> list[dict]:
    """Filter to data points where model was NOT eval-aware.

    Args:
        data: List of data points.
        awareness_labels: Dict mapping IDs to awareness results.

    Returns:
        Filtered list of data points.
    """
    return [
        dp
        for dp in data
        if awareness_labels.get(dp["id"], {}).get("final_classification") == "NOT_EVAL_AWARE"
    ]


def get_response_completion(responses: dict[str, dict], dp_id: str) -> str | None:
    """Get the completion text from a responses dict.

    Args:
        responses: Dict mapping IDs to response dicts.
        dp_id: Data point ID.

    Returns:
        Completion text or None if not found.
    """
    if dp_id not in responses:
        return None
    return responses[dp_id].get("completion")


def print_stats(
    name: str,
    total: int,
    kept: int,
    skipped_identical: int = 0,
    skipped_missing: int = 0,
) -> None:
    """Print statistics about dataset creation.

    Args:
        name: Name of the dataset/split.
        total: Total number of examples.
        kept: Number of examples kept.
        skipped_identical: Number skipped due to identical responses.
        skipped_missing: Number skipped due to missing responses.
    """
    print(f"\n{name}:")
    print(f"  Total: {total}")
    print(f"  Kept: {kept} ({kept / total * 100:.1f}%)" if total > 0 else "  Kept: 0")
    if skipped_identical > 0:
        print(f"  Skipped (identical): {skipped_identical}")
    if skipped_missing > 0:
        print(f"  Skipped (missing): {skipped_missing}")


def validate_data_point(dp: dict) -> bool:
    """Validate that a data point has the required fields.

    Args:
        dp: Data point dict.

    Returns:
        True if valid, False otherwise.
    """
    if "id" not in dp:
        return False
    if "input" not in dp:
        return False
    if not isinstance(dp["input"], list):
        return False
    for msg in dp["input"]:
        if "role" not in msg or "content" not in msg:
            return False
    return True


def count_by_classification(awareness_labels: dict[str, dict]) -> dict[str, int]:
    """Count examples by final classification.

    Args:
        awareness_labels: Dict mapping IDs to awareness results.

    Returns:
        Dict with counts for each classification.
    """
    counts = {"EVAL_AWARE": 0, "NOT_EVAL_AWARE": 0, "UNKNOWN": 0}
    for result in awareness_labels.values():
        classification = result.get("final_classification", "UNKNOWN")
        if classification in counts:
            counts[classification] += 1
        else:
            counts["UNKNOWN"] += 1
    return counts
