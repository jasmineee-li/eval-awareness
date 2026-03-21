"""Shared data generation for Plan A + Plan B metacognitive training.

Generates object-level completions, creates finetuning datasets for standard
properties, generates counterfactual finetuning samples (unbalanced), and
merges everything into a single training JSONL.

Usage:
    # From introspection_self_prediction/ directory, with vLLM server running:
    python -m scripts.run_shared_data_gen --study_name metacog_shared
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Must run from introspection_self_prediction/ directory
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from evals.locations import EXP_DIR
from evals.create_finetuning_dataset_configs import create_finetuning_dataset_config

# ── Task × Property Mappings ──────────────────────────────────────────────────

TRAIN_TASKS = {
    # Scaffold tasks
    "wikipedia": [
        "first_character", "second_character", "third_character",
        "first_word", "second_word", "starts_with_vowel",
    ],
    "number_triplets": [
        "first_character", "second_character", "is_even", "starts_with_vowel",
    ],
    "arc_challenge_non_cot": [
        "first_character", "is_either_a_or_c", "is_either_b_or_d",
    ],
    # Disposition tasks
    "wealth_seeking": [
        "matches_wealth_seeking", "first_character", "starts_with_vowel",
    ],
    "power_seeking": [
        "matches_power_seeking", "first_character", "starts_with_vowel",
    ],
    "deontology": [
        "matches_deontology", "first_character", "starts_with_vowel",
    ],
    # Identity task
    "self_referential": [
        "first_character", "first_word", "starts_with_vowel",
    ],
}

TEST_TASKS = {
    "animals": [
        "first_character", "second_character", "third_character",
        "first_word", "second_word", "starts_with_vowel",
    ],
    "english_words": [
        "first_character", "second_character", "starts_with_vowel",
    ],
    "stories_sentences": [
        "first_character", "first_word", "starts_with_vowel",
    ],
    "mmlu_non_cot": [
        "first_character", "is_either_a_or_c", "is_either_b_or_d",
    ],
    "myopic_reward": [
        "matches_myopic_reward", "first_character",
    ],
    "survival_instinct": [
        "matches_survival_instinct", "first_character",
    ],
    "personal_preferences": [
        "first_character", "first_word",
    ],
    "daily_dialog": [
        "first_character", "first_word", "starts_with_vowel",
    ],
}

MODEL_CONFIG = "qwen3-32b"
PROMPT_CONFIG = "minimal"

# Dataset sizes (use all available per task)
TRAIN_LIMITS = {
    "wikipedia": 3659,
    "number_triplets": 10000,
    "arc_challenge_non_cot": 1833,
    "wealth_seeking": 494,
    "power_seeking": 494,
    "deontology": 800,
    "self_referential": 43,
}

TEST_LIMITS = {
    "animals": 2500,
    "english_words": 2500,
    "stories_sentences": 3000,
    "mmlu_non_cot": 2500,
    "myopic_reward": 376,
    "survival_instinct": 465,
    "personal_preferences": 15,
    "daily_dialog": 159,
}


def run_cmd(cmd: str, description: str) -> str:
    """Run a command, stream output, return last line."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}\n")

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(REPO_DIR),
    )
    output_lines = []
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line.strip())
    process.wait()

    if process.returncode != 0:
        print(f"\nERROR: Command failed with return code {process.returncode}")
        return ""

    last_line = output_lines[-1] if output_lines else ""
    print(f"\nOutput (last line): {last_line}")
    return last_line


def generate_object_level(study_name: str, task: str, task_set: str, limit: int) -> str:
    """Generate object-level completions. Returns experiment directory path."""
    cmd = (
        f"python -m evals.run_object_level"
        f" study_name={study_name}"
        f" language_model={MODEL_CONFIG}"
        f" task={task}"
        f" task.set={task_set}"
        f" prompt=object_level/{PROMPT_CONFIG}"
        f" limit={limit}"
    )
    return run_cmd(cmd, f"Object-level: {task} ({task_set}, n={limit})")


def create_finetuning_configs(
    study_name: str,
    train_dirs: dict[str, str],
    val_dirs: dict[str, str],
) -> None:
    """Create Hydra YAML configs for each (task, property) pair."""
    for task, properties in TRAIN_TASKS.items():
        if task not in train_dirs or task not in val_dirs:
            print(f"WARNING: Skipping {task} — missing object-level dirs")
            continue
        for prop in properties:
            create_finetuning_dataset_config(
                study_name=study_name,
                model_config=MODEL_CONFIG,
                task_config=task,
                prompt_config=PROMPT_CONFIG,
                response_property_config=prop,
                overrides="",
                train_base_dir=train_dirs[task],
                val_base_dir=val_dirs[task],
            )
    print(f"Created finetuning dataset configs in exp/finetuning/{study_name}/")


def create_finetuning_dataset(study_name: str) -> tuple[Path, Path]:
    """Run create_finetuning_dataset.py to generate the merged JSONL."""
    from evals.utils import safe_model_name
    dataset_folder = safe_model_name(MODEL_CONFIG)
    cmd = (
        f"python -m evals.create_finetuning_dataset"
        f" study_name={study_name}"
        f" dataset_folder={dataset_folder}"
    )
    run_cmd(cmd, "Creating finetuning dataset from standard properties")

    ft_dir = EXP_DIR / "finetuning" / study_name / dataset_folder
    train_path = ft_dir / "train_dataset.jsonl"
    val_path = ft_dir / "val_dataset.jsonl"
    return train_path, val_path


def generate_counterfactual_samples(study_name: str) -> Path:
    """Generate counterfactual finetuning samples (unbalanced) and save to JSONL."""
    from other_evals.counterfactuals.get_finetuning_samples import get_other_evals_finetuning_samples
    from other_evals.counterfactuals.runners import ALL_EVAL_TYPES
    from other_evals.counterfactuals.api_utils import write_jsonl_file_from_basemodel

    print(f"\n{'='*60}")
    print("Generating counterfactual finetuning samples (unbalanced)")
    print(f"Evals: {[e.name() for e in ALL_EVAL_TYPES]}")
    print(f"{'='*60}\n")

    # Use the model config YAML path
    model_config_path = str(REPO_DIR / "evals" / "conf" / "language_model" / f"{MODEL_CONFIG}.yaml")

    samples = get_other_evals_finetuning_samples(
        evals_to_run=ALL_EVAL_TYPES,
        object_model_config=model_config_path,
        try_n_samples=10000,
        limit_per_eval=2000,
        cache_path=EXP_DIR / study_name / "counterfactual_cache",
        balance_data=False,  # Skip balancing for training
    )

    output_path = EXP_DIR / "finetuning" / study_name / "counterfactual_samples.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_file_from_basemodel(output_path, samples)
    print(f"Saved {len(samples)} counterfactual samples to {output_path}")
    return output_path


def merge_datasets(
    standard_train: Path,
    counterfactual: Path,
    output_path: Path,
) -> int:
    """Merge standard and counterfactual training data into one JSONL."""
    from other_evals.counterfactuals.get_finetuning_samples import add_new_samples_to_existing_jsonl_and_shuffle
    from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel
    from other_evals.counterfactuals.other_eval_csv_format import FinetuneConversation

    counterfactual_samples = read_jsonl_file_into_basemodel(counterfactual, basemodel=FinetuneConversation)
    add_new_samples_to_existing_jsonl_and_shuffle(
        existing_jsonl_path=standard_train,
        new_jsonl_path=output_path,
        new_samples=counterfactual_samples,
    )

    total = sum(1 for _ in open(output_path))
    print(f"Merged dataset: {total} total samples → {output_path}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Shared data generation for Plan A + Plan B")
    parser.add_argument("--study_name", type=str, default="metacog_shared")
    parser.add_argument("--skip_object_level", action="store_true", help="Skip object-level generation (if already done)")
    parser.add_argument("--skip_counterfactual", action="store_true", help="Skip counterfactual generation")
    args = parser.parse_args()

    study_name = args.study_name

    # ── Step 1: Object-level generation ──────────────────────────────────────
    train_dirs = {}
    val_dirs = {}

    if not args.skip_object_level:
        print("\n" + "="*60)
        print("STEP 1: Object-level generation (train tasks)")
        print("="*60)
        for task, limit in TRAIN_LIMITS.items():
            exp_dir = generate_object_level(study_name, task, "train", limit)
            train_dirs[task] = exp_dir
            # Also generate val split
            val_limit = min(limit, 500)
            val_dir = generate_object_level(study_name, task, "val", val_limit)
            val_dirs[task] = val_dir

        print("\n" + "="*60)
        print("STEP 1b: Object-level generation (test tasks)")
        print("="*60)
        for task, limit in TEST_LIMITS.items():
            generate_object_level(study_name, task, "val", limit)
    else:
        print("Skipping object-level generation (--skip_object_level)")
        # Need to reconstruct dirs from existing runs
        print("WARNING: You must provide train_dirs and val_dirs manually or re-run without --skip_object_level")

    # ── Step 2: Create finetuning dataset configs ────────────────────────────
    if train_dirs and val_dirs:
        print("\n" + "="*60)
        print("STEP 2: Creating finetuning dataset configs")
        print("="*60)
        create_finetuning_configs(study_name, train_dirs, val_dirs)

        # ── Step 3: Generate finetuning dataset ──────────────────────────────
        print("\n" + "="*60)
        print("STEP 3: Generating finetuning JSONL from standard properties")
        print("="*60)
        standard_train, standard_val = create_finetuning_dataset(study_name)
        print(f"Standard train: {standard_train}")
        print(f"Standard val: {standard_val}")

    # ── Step 4: Counterfactual finetuning samples ────────────────────────────
    if not args.skip_counterfactual:
        print("\n" + "="*60)
        print("STEP 4: Generating counterfactual finetuning samples")
        print("="*60)
        counterfactual_path = generate_counterfactual_samples(study_name)

        # ── Step 5: Merge ────────────────────────────────────────────────────
        if train_dirs and val_dirs:
            print("\n" + "="*60)
            print("STEP 5: Merging standard + counterfactual → train_standard_counterfactual.jsonl")
            print("="*60)
            from evals.utils import safe_model_name
            ft_dir = EXP_DIR / "finetuning" / study_name / safe_model_name(MODEL_CONFIG)
            merged_path = ft_dir / "train_standard_counterfactual.jsonl"
            total = merge_datasets(standard_train, counterfactual_path, merged_path)
            print(f"\nFinal merged dataset: {total} samples")
            print(f"Path: {merged_path}")
            print(f"\nReady for Plan A (use directly) and Plan B (add eval_context samples)")

    print("\n" + "="*60)
    print("SHARED DATA GENERATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
