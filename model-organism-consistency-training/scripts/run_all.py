#!/usr/bin/env python3
"""Orchestrate the full consistency training pipeline.

Runs all phases in sequence:
1. Data preparation (split into train/val/test)
2. Initial inference (run QwQ-32B on all questions)
3. Eval awareness detection (run judges)
4. Prefixed inference (re-run with "This is an evaluation." prefix)
5. Dataset creation (create SFT and DPO datasets)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, args: list[str], dry_run: bool = False) -> int:
    """Run a script with arguments.
    
    Args:
        script_name: Name of the script to run.
        args: Arguments to pass to the script.
        dry_run: If True, print command but don't execute.
    
    Returns:
        Return code (0 if dry_run).
    """
    cmd = [sys.executable, f"scripts/{script_name}"] + args
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    
    if dry_run:
        print("(dry run - skipping execution)")
        return 0
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: {script_name} failed with return code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run the full consistency training pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_combined_chat_single-turn_eval_subset.json"),
        help="Input JSON file with all data points",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Base directory for all outputs",
    )
    parser.add_argument(
        "--model-id",
        default="openrouter/qwen/qwq-32b",
        help="Model to use for inference",
    )
    parser.add_argument(
        "--grader-model",
        default="openrouter/openai/gpt-5-mini",
        help="Model to use for grading",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=None,
        help="Path to initial responses JSON (overrides default inference_dir/initial_responses.json)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=80,
        help="Number of parallel requests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Start from phase N (1-5)",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Stop after phase N (1-5)",
    )
    args = parser.parse_args()

    splits_dir = args.output_dir / "splits"
    inference_dir = args.output_dir / "inference_results"
    awareness_dir = args.output_dir / "eval_awareness"
    datasets_dir = args.output_dir / "training_datasets"
    
    # Use custom responses path if provided, otherwise default
    responses_path = args.responses if args.responses else inference_dir / "initial_responses.json"

    print("=" * 60)
    print("CONSISTENCY TRAINING PIPELINE")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Responses file: {responses_path}")
    print(f"Model: {args.model_id}")
    print(f"Grader: {args.grader_model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Running phases: {args.start_from} to {args.stop_after}")
    if args.dry_run:
        print("\n*** DRY RUN MODE - No commands will be executed ***")

    # Phase 1: Data Preparation
    # if args.start_from <= 1 <= args.stop_after:
    #     print("\n" + "=" * 60)
    #     print("PHASE 1: Data Preparation")
    #     print("=" * 60)
    #     rc = run_script(
    #         "01_prepare_data.py",
    #         [
    #             "--input", str(args.input),
    #             "--output-dir", str(splits_dir),
    #             "--seed", str(args.seed),
    #         ],
    #         dry_run=args.dry_run,
    #     )
    #     if rc != 0 and not args.dry_run:
    #         print("Pipeline failed at Phase 1")
    #         return rc

    # # Phase 2: Initial Inference
    # if args.start_from <= 2 <= args.stop_after:
    #     print("\n" + "=" * 60)
    #     print("PHASE 2: Initial Inference")
    #     print("=" * 60)
    #     rc = run_script(
    #         "02_run_initial_inference.py",
    #         [
    #             "--splits-dir", str(splits_dir),
    #             "--output", str(inference_dir / "initial_responses.json"),
    #             "--model-id", args.model_id,
    #             "--concurrency", str(args.concurrency),
    #         ],
    #         dry_run=args.dry_run,
    #     )
    #     if rc != 0 and not args.dry_run:
    #         print("Pipeline failed at Phase 2")
    #         return rc

    # Phase 3: Eval Awareness Detection
    if args.start_from <= 3 <= args.stop_after:
        print("\n" + "=" * 60)
        print("PHASE 3: Eval Awareness Detection")
        print("=" * 60)
        rc = run_script(
            "03_detect_eval_awareness.py",
            [
                "--splits-dir", str(splits_dir),
                "--responses", str(responses_path),
                "--output", str(awareness_dir / "awareness_labels.json"),
                "--eval-model", args.model_id,
                "--grader-model", args.grader_model,
                "--concurrency", str(args.concurrency),
            ],
            dry_run=args.dry_run,
        )
        if rc != 0 and not args.dry_run:
            print("Pipeline failed at Phase 3")
            return rc

    # Phase 4: Prefixed Inference
    if args.start_from <= 4 <= args.stop_after:
        print("\n" + "=" * 60)
        print("PHASE 4: Prefixed Inference")
        print("=" * 60)
        rc = run_script(
            "04_run_prefixed_inference.py",
            [
                "--splits-dir", str(splits_dir),
                "--awareness-labels", str(awareness_dir / "awareness_labels.json"),
                "--output", str(inference_dir / "prefixed_responses.json"),
                "--model-id", args.model_id,
                "--concurrency", str(args.concurrency),
            ],
            dry_run=args.dry_run,
        )
        if rc != 0 and not args.dry_run:
            print("Pipeline failed at Phase 4")
            return rc

    # Phase 5: Dataset Creation
    if args.start_from <= 5 <= args.stop_after:
        print("\n" + "=" * 60)
        print("PHASE 5: Dataset Creation")
        print("=" * 60)
        rc = run_script(
            "05_create_datasets.py",
            [
                "--splits-dir", str(splits_dir),
                "--initial-responses", str(responses_path),
                "--prefixed-responses", str(inference_dir / "prefixed_responses.json"),
                "--awareness-labels", str(awareness_dir / "awareness_labels.json"),
                "--output-dir", str(datasets_dir),
            ],
            dry_run=args.dry_run,
        )
        if rc != 0 and not args.dry_run:
            print("Pipeline failed at Phase 5")
            return rc

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Data splits: {splits_dir}")
    print(f"  Initial responses: {responses_path}")
    print(f"  Awareness labels: {awareness_dir / 'awareness_labels.json'}")
    print(f"  Prefixed responses: {inference_dir / 'prefixed_responses.json'}")
    print(f"  Training datasets: {datasets_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
