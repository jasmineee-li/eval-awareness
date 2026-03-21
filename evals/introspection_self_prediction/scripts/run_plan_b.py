"""Plan B: Metacognitive Training WITH eval_context_sensitivity.

Takes the merged training data from run_shared_data_gen.py, adds eval_context
ground truth samples, and runs:
1. Merge eval_context samples into training data
2. QLoRA finetuning on combined dataset
3. Meta-level eval on held-out tasks
4. eval_context_sensitivity on all 4 cue levels
5. Trigger dataset + agentic misalignment evals
6. Push checkpoint to HuggingFace

Usage:
    python -m scripts.run_plan_b --study_name metacog_shared --plan_name plan_b
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from evals.locations import EXP_DIR
from evals.utils import safe_model_name

MODEL_CONFIG = "qwen3-32b"
PROMPT_CONFIG = "minimal"

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

# eval_context_sensitivity properties to test (all 4 cue levels)
EVAL_CONTEXT_PROPERTIES = [
    "eval_context_sensitivity_explicit",
    "eval_context_sensitivity_moderate",
    "eval_context_sensitivity_subtle",
    "eval_context_sensitivity_minimal",
]


def run_cmd(cmd: str, description: str) -> str:
    print(f"\n{'='*60}")
    print(f"[Plan B] {description}")
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

    return output_lines[-1] if output_lines else ""


def merge_eval_context_samples(
    standard_train: Path,
    eval_context_samples: Path,
    output_path: Path,
) -> int:
    """Merge standard+counterfactual JSONL with eval_context samples."""
    # Read both files
    standard_lines = standard_train.read_text().strip().split("\n")
    eval_lines = eval_context_samples.read_text().strip().split("\n")

    combined = standard_lines + eval_lines
    random.seed(42)
    random.shuffle(combined)

    output_path.write_text("\n".join(combined) + "\n")
    print(f"Merged: {len(standard_lines)} standard + {len(eval_lines)} eval_context = {len(combined)} total")
    return len(combined)


def main():
    parser = argparse.ArgumentParser(description="Plan B: Metacognitive Training (treatment)")
    parser.add_argument("--study_name", type=str, default="metacog_shared")
    parser.add_argument("--plan_name", type=str, default="plan_b")
    parser.add_argument("--skip_finetuning", action="store_true")
    parser.add_argument("--skip_eval_context_gen", action="store_true",
                        help="Skip eval_context ground truth generation (if already done)")
    parser.add_argument("--ft_model_config", type=str, default=None)
    args = parser.parse_args()

    model_folder = safe_model_name(MODEL_CONFIG)
    ft_dir = EXP_DIR / "finetuning" / args.study_name / model_folder
    standard_train = ft_dir / "train_standard_counterfactual.jsonl"
    val_path = ft_dir / "val_dataset.jsonl"

    if not standard_train.exists():
        print(f"ERROR: Standard training data not found at {standard_train}")
        print("Run scripts/run_shared_data_gen.py first.")
        sys.exit(1)

    # ── Step 1: Generate eval_context ground truth ───────────────────────────
    eval_context_path = EXP_DIR / "finetuning" / args.study_name / "eval_context_finetune_samples.jsonl"

    if not args.skip_eval_context_gen:
        print("\n" + "="*60)
        print("STEP 1: Generating eval_context_sensitivity ground truth")
        print("="*60)
        cmd = (
            f"python -m scripts.generate_eval_context_ground_truth"
            f" --study_name {args.study_name}"
            f" --n_samples 2000"
        )
        run_cmd(cmd, "eval_context ground truth generation")

    if not eval_context_path.exists():
        print(f"ERROR: eval_context samples not found at {eval_context_path}")
        print("Run generate_eval_context_ground_truth.py first, or check --study_name.")
        sys.exit(1)

    # ── Step 2: Merge into Plan B training data ──────────────────────────────
    print("\n" + "="*60)
    print("STEP 2: Merging standard + eval_context → train_plan_b.jsonl")
    print("="*60)

    plan_b_train = ft_dir / "train_plan_b.jsonl"
    total = merge_eval_context_samples(standard_train, eval_context_path, plan_b_train)
    print(f"Plan B training data: {total} samples → {plan_b_train}")

    # ── Step 3: QLoRA Finetuning ─────────────────────────────────────────────
    if not args.skip_finetuning:
        ft_study = f"{args.study_name}/{args.plan_name}"
        cmd = (
            f"python -m evals.run_finetuning"
            f" study_name={ft_study}"
            f" train_path={plan_b_train.as_posix()}"
            f" val_path={val_path.as_posix()}"
            f" language_model={MODEL_CONFIG}"
            f" notes={args.plan_name}"
            f" lora_rank=16"
            f" epochs=3"
            f" learning_rate=1e-4"
            f" batch_size=32"
            f" gradient_accumulation_steps=8"
        )
        ft_model_config = run_cmd(cmd, "QLoRA finetuning (Plan B)")
        print(f"\nFinetuned model config: {ft_model_config}")
    else:
        ft_model_config = args.ft_model_config
        if not ft_model_config:
            print("ERROR: --ft_model_config required when --skip_finetuning")
            sys.exit(1)

    # ── Step 4: Meta-level eval on held-out tasks ────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Meta-level evaluation on held-out tasks")
    print("="*60)

    for task, properties in TEST_TASKS.items():
        for prop in properties:
            cmd = (
                f"python -m evals.run_meta_level"
                f" study_name={args.study_name}/{args.plan_name}"
                f" language_model={ft_model_config}"
                f" task={task}"
                f" task.set=val"
                f" response_property={prop}"
                f" prompt=meta_level/{PROMPT_CONFIG}"
                f" limit=500"
            )
            run_cmd(cmd, f"Meta-level: {task} × {prop}")

    # ── Step 5: eval_context_sensitivity on all 4 cue levels ─────────────────
    print("\n" + "="*60)
    print("STEP 5: eval_context_sensitivity (trained + generalization)")
    print("="*60)
    print("Test eval_context_sensitivity on all 4 cue levels:")
    print("  - explicit (trained)")
    print("  - moderate (trained)")
    print("  - subtle (generalization)")
    print("  - minimal (generalization)")
    print(f"Use generate_eval_context_ground_truth.py with finetuned model: {ft_model_config}")

    # ── Step 6: Trigger dataset + agentic misalignment ───────────────────────
    print("\n" + "="*60)
    print("STEP 6: External eval integration")
    print("="*60)
    print("Run the Plan B finetuned model through:")
    print("  1. Trigger dataset eval (evals/Test_Awareness_Steering/)")
    print("  2. Agentic misalignment eval (evals/agentic_misalignment/)")
    print(f"Model config: {ft_model_config}")

    # ── Step 7: Push to HuggingFace ──────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7: Push checkpoint to HuggingFace")
    print("="*60)
    print("Run the following command to push:")
    print(f'python -c "from huggingface_hub import HfApi; HfApi().upload_folder('
          f"folder_path='exp/{args.study_name}/{args.plan_name}/merged_model', "
          f"repo_id='jasminexli/qwen3-32b-metacog-{args.plan_name}', "
          f"repo_type='model', create_remote=True)\"")

    print("\n" + "="*60)
    print("PLAN B COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
