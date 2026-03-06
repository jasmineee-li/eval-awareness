#!/usr/bin/env python3
"""Run judges to detect eval awareness in responses.

Phase 3 of the consistency training pipeline.
Uses two methods: verbalized_awareness and purpose_continue.
"""

import asyncio
import argparse
import json
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils
from tqdm.asyncio import tqdm

from consistency_training.judges import detect_eval_awareness


async def main():
    parser = argparse.ArgumentParser(description="Detect eval awareness in responses")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing the data splits",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("data/inference_results/initial_responses.json"),
        help="JSON file with initial responses",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_awareness/awareness_labels.json"),
        help="Output file for awareness labels",
    )
    parser.add_argument(
        "--eval-model",
        default="openrouter/qwen/qwq-32b",
        help="Model being evaluated (for purpose_continue)",
    )
    parser.add_argument(
        "--grader-model",
        default="openrouter/openai/gpt-5-mini",
        help="Model for grading/classification",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=80,
        help="Number of parallel requests",
    )
    parser.add_argument(
        "--no-verbalized",
        action="store_true",
        help="Disable verbalized_awareness judge",
    )
    parser.add_argument(
        "--no-purpose",
        action="store_true",
        help="Disable purpose_continue judge",
    )
    args = parser.parse_args()

    utils.setup_environment()
    api = InferenceAPI(
        openrouter_num_threads=args.concurrency,
        cache_dir=Path(".cache"),
    )

    # Load data and responses
    all_data = []
    for split in ["train", "val", "test"]:
        split_path = args.splits_dir / f"{split}.json"
        if split_path.exists():
            with open(split_path) as f:
                all_data.extend(json.load(f))

    with open(args.responses) as f:
        responses = json.load(f)

    # Build lookup
    data_by_id = {dp["id"]: dp for dp in all_data}

    # Filter to only IDs we have responses for
    ids_to_process = [dp_id for dp_id in responses.keys() if dp_id in data_by_id]

    # Load existing checkpoint if it exists (for resuming)
    results = {}
    if args.output.exists():
        print(f"Loading checkpoint from {args.output}")
        with open(args.output) as f:
            results = json.load(f)
        already_done = set(results.keys())
        ids_to_process = [dp_id for dp_id in ids_to_process if dp_id not in already_done]
        print(f"  Resuming: {len(already_done)} already done, {len(ids_to_process)} remaining")

    print(f"Running eval awareness detection on {len(ids_to_process)} examples")
    print(f"  Eval model (for purpose_continue): {args.eval_model}")
    print(f"  Grader model: {args.grader_model}")
    print(f"  Concurrency: {args.concurrency} parallel requests")
    print(f"  Using verbalized_awareness: {not args.no_verbalized}")
    print(f"  Using purpose_continue: {not args.no_purpose}")
    print(f"  Checkpoint: {args.output} (every 500 items)")

    async def process_one(dp_id: str):
        dp = data_by_id[dp_id]
        response = responses[dp_id]["completion"]
        result = await detect_eval_awareness(
            api=api,
            dp=dp,
            response=response,
            eval_model=args.eval_model,
            grader_model=args.grader_model,
            use_verbalized=not args.no_verbalized,
            use_purpose=not args.no_purpose,
        )
        return dp_id, result

    # Process in concurrent batches with periodic checkpointing
    items_since_checkpoint = 0
    checkpoint_every = 500

    for i in tqdm(range(0, len(ids_to_process), args.concurrency), desc="Judging"):
        batch_ids = ids_to_process[i : i + args.concurrency]
        tasks = [process_one(dp_id) for dp_id in batch_ids]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                print(f"Error: {result}")
                continue
            dp_id, judge_result = result
            results[dp_id] = judge_result
            items_since_checkpoint += 1

        # Periodic checkpoint
        if items_since_checkpoint >= checkpoint_every:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temp_path = args.output.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(results, f)
            temp_path.rename(args.output)
            print(f"  [Checkpoint saved: {len(results)} items]")
            items_since_checkpoint = 0

    # Final save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print statistics
    if len(results) > 0:
        eval_aware = sum(
            1 for r in results.values() if r["final_classification"] == "EVAL_AWARE"
        )
        not_aware = len(results) - eval_aware
        print(f"\nResults:")
        print(f"  EVAL_AWARE: {eval_aware} ({eval_aware / len(results) * 100:.1f}%)")
        print(f"  NOT_EVAL_AWARE: {not_aware} ({not_aware / len(results) * 100:.1f}%)")
    else:
        print(f"\nWarning: No results were successfully processed!")
    print(f"\nSaved awareness labels to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
