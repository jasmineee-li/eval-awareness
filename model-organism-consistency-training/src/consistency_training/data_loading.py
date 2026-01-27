"""Data loading and splitting utilities."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict


class Message(TypedDict):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class DataPoint(TypedDict):
    """A single data point from the evaluation dataset."""

    id: str
    input: list[Message]
    metadata: dict


@dataclass
class DataSplits:
    """Container for train/val/test data splits."""

    train: list[DataPoint]
    val: list[DataPoint]
    test: list[DataPoint]


def load_and_split_data(
    data_path: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> DataSplits:
    """Load data from JSON file and split into train/val/test.

    Args:
        data_path: Path to the JSON file containing the data.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        seed: Random seed for reproducible shuffling.

    Returns:
        DataSplits containing train, val, and test lists.

    Raises:
        AssertionError: If ratios don't sum to 1.0.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    with open(data_path) as f:
        data = json.load(f)

    # Copy before shuffling to avoid mutating input
    data = list(data)

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return DataSplits(
        train=data[:train_end],
        val=data[train_end:val_end],
        test=data[val_end:],
    )


def save_splits(splits: DataSplits, output_dir: Path) -> None:
    """Save train/val/test splits to separate JSON files.

    Args:
        splits: DataSplits containing the data.
        output_dir: Directory to save the splits to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        output_path = output_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} examples to {output_path}")


def load_split(split_path: Path) -> list[DataPoint]:
    """Load a single split from a JSON file.

    Args:
        split_path: Path to the split JSON file.

    Returns:
        List of DataPoint objects.
    """
    with open(split_path) as f:
        return json.load(f)


def load_all_splits(splits_dir: Path) -> list[DataPoint]:
    """Load all splits (train, val, test) and combine them.

    Args:
        splits_dir: Directory containing the split files.

    Returns:
        Combined list of all DataPoints.
    """
    all_data = []
    for split in ["train", "val", "test"]:
        split_path = splits_dir / f"{split}.json"
        if split_path.exists():
            all_data.extend(load_split(split_path))
    return all_data


def load_training_splits(splits_dir: Path) -> list[DataPoint]:
    """Load only train and val splits (for training).

    Args:
        splits_dir: Directory containing the split files.

    Returns:
        Combined list of train and val DataPoints.
    """
    all_data = []
    for split in ["train", "val"]:
        split_path = splits_dir / f"{split}.json"
        if split_path.exists():
            all_data.extend(load_split(split_path))
    return all_data
