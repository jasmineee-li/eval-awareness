#!/usr/bin/env python3
"""Run inference with 'This is an evaluation.' prefix on filtered questions.

Phase 4 of the consistency training pipeline.
Only processes questions where the model was NOT eval-aware.
"""

import asyncio
import argparse
import json
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

from consistency_training.inference import run_inference_batch, serialize_responses
from consistency_training.utils import filter_non_eval_aware


async def main():
    parser = argparse.ArgumentParser(description="Run prefixed inference on filtered questions")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing the data splits",
    )
    parser.add_argument(
        "--awareness-labels",
        type=Path,
        default=Path("data/eval_awareness/awareness_labels.json"),
        help="JSON file with awareness labels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/inference_results/prefixed_responses.json"),
        help="Output file for prefixed responses",
    )
    parser.add_argument(
        "--model-id",
        default="openrouter/qwen/qwq-32b",
        help="Model to use for inference",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=80,
        help="Number of parallel requests",
    )
    args = parser.parse_args()

    utils.setup_environment()
    api = InferenceAPI(
        openrouter_num_threads=args.concurrency,
        cache_dir=Path(".cache"),
    )

    # Load data (only train and val for training datasets)
    all_data = []
    for split in ["train", "val"]:
        split_path = args.splits_dir / f"{split}.json"
        if split_path.exists():
            with open(split_path) as f:
                all_data.extend(json.load(f))
            print(f"Loaded {split}: {split_path}")

    with open(args.awareness_labels) as f:
        awareness_labels = json.load(f)

    # Filter to non-eval-aware questions
    filtered_data = filter_non_eval_aware(all_data, awareness_labels)

    print(f"\nFiltered to {len(filtered_data)} non-eval-aware questions")
    print(f"  (from {len(all_data)} total train+val)")
    print(f"\nRunning prefixed inference")
    print(f"  Model: {args.model_id}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Concurrency: {args.concurrency} parallel requests")
    print(f"  Checkpoint: {args.output} (every 500 items)")

    # Run inference with prefix
    results = await run_inference_batch(
        api=api,
        data=filtered_data,
        model_id=args.model_id,
        add_prefix=True,  # This adds "This is an evaluation."
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        checkpoint_path=args.output,  # Enable checkpointing to output file
        checkpoint_every=500,
    )

    # Final save (checkpoint already saved, but ensure clean output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = serialize_responses(results)

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(output_data)} prefixed responses to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
