"""Plan A: Metacognitive Training WITHOUT eval_context_sensitivity.

Takes the merged training data from run_shared_data_gen.py and runs:
1. QLoRA finetuning on standard + counterfactual data
2. Meta-level eval on held-out tasks
3. Zero-shot eval_context_sensitivity on all 4 cue levels
4. Push checkpoint to HuggingFace

Usage:
    python -m scripts.run_plan_a --study_name metacog_shared --plan_name plan_a
"""

import argparse
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


def run_cmd(cmd: str, description: str) -> str:
    print(f"\n{'='*60}")
    print(f"[Plan A] {description}")
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


def main():
    parser = argparse.ArgumentParser(description="Plan A: Metacognitive Training (control)")
    parser.add_argument("--study_name", type=str, default="metacog_shared")
    parser.add_argument("--plan_name", type=str, default="plan_a")
    parser.add_argument("--skip_finetuning", action="store_true")
    parser.add_argument("--skip_meta_eval", action="store_true",
                        help="Skip meta-level evaluation (run finetuning only)")
    parser.add_argument("--only_meta_eval", action="store_true",
                        help="Skip finetuning, run meta-level eval only (needs vLLM serving finetuned model)")
    parser.add_argument("--ft_model_config", type=str, default=None,
                        help="If skip_finetuning, provide the finetuned model config name")
    args = parser.parse_args()

    model_folder = safe_model_name(MODEL_CONFIG)
    ft_dir = EXP_DIR / "finetuning" / args.study_name / model_folder
    train_path = ft_dir / "train_standard_counterfactual.jsonl"
    val_path = ft_dir / "val_dataset.jsonl"

    # ── Step 1: LoRA Finetuning ──────────────────────────────────────────────
    if not args.skip_finetuning and not args.only_meta_eval:
        if not train_path.exists():
            print(f"ERROR: Training data not found at {train_path}")
            print("Run scripts/run_shared_data_gen.py first.")
            sys.exit(1)

        ft_study = f"{args.study_name}/{args.plan_name}"
        cmd = (
            f"python -m evals.run_finetuning"
            f" study_name={ft_study}"
            f" train_path={train_path.as_posix()}"
            f" val_path={val_path.as_posix()}"
            f" language_model={MODEL_CONFIG}"
            f" notes={args.plan_name}"
            f" lora_rank=32"
            f" epochs=1"
            f" learning_rate=1e-4"
            f" batch_size=32"
            f" gradient_accumulation_steps=8"
        )
        ft_model_config = run_cmd(cmd, "LoRA finetuning")
        print(f"\nFinetuned model config: {ft_model_config}")
    elif args.ft_model_config:
        ft_model_config = args.ft_model_config
    else:
        # Try to find existing finetuned model config
        ft_model_config = MODEL_CONFIG  # fallback
        print(f"Using model config: {ft_model_config}")

    if args.skip_meta_eval:
        print("Skipping meta-level evaluation (--skip_meta_eval)")
        return

    # ── Step 2: Meta-level eval on held-out tasks ────────────────────────────
    print("\n" + "="*60)
    print("STEP 2: Meta-level evaluation on held-out tasks")
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

    # ── Step 3: Zero-shot eval_context_sensitivity ───────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Zero-shot eval_context_sensitivity (all 4 cue levels)")
    print("="*60)
    print("Run generate_eval_context_ground_truth.py with the finetuned model")
    print("to test zero-shot eval_context awareness.")
    print(f"Finetuned model config: {ft_model_config}")

    # ── Step 4: Push to HuggingFace ──────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Push checkpoint to HuggingFace")
    print("="*60)
    print("Run the following command to push:")
    print(f'python -c "from huggingface_hub import HfApi; HfApi().upload_folder('
          f"folder_path='exp/{args.study_name}/{args.plan_name}/merged_model', "
          f"repo_id='jasminexli/qwen3-32b-metacog-{args.plan_name}', "
          f"repo_type='model', create_remote=True)\"")

    print("\n" + "="*60)
    print("PLAN A COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
