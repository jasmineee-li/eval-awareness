#!/usr/bin/env python3
"""Run inference on perturbation dataset using vLLM.

Runs a specific model/variant combination and saves responses.
Supports --use-existing to copy from existing inference results (e.g., for base_bare).
"""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

# Model paths
MODELS = {
    "base": "Qwen/QwQ-32B",  # Base model on vLLM
    "finetuned": "ckpt/qwq32b_measurement_cooperation_020126/finetuned_model",
}


async def call_vllm(
    session: aiohttp.ClientSession,
    vllm_url: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """Call local vLLM server."""
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with session.post(
        f"{vllm_url}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"vLLM error {response.status}: {error_text[:200]}")
        data = await response.json()
        return data["choices"][0]["message"]["content"]


async def main():
    parser = argparse.ArgumentParser(description="Run perturbation inference using vLLM")
    parser.add_argument(
        "--model",
        choices=["base", "finetuned"],
        required=True,
        help="Which model to use",
    )
    parser.add_argument(
        "--variant",
        choices=["bare", "eval_mild", "eval_strong"],
        required=True,
        help="Which perturbation variant to run",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/coop_training_results/perturbation_dataset.json"),
        help="Perturbation dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (default: data/coop_training_results/inference/{model}_{variant}.json)",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--limit", type=int, help="Limit number of examples (for testing)")
    parser.add_argument(
        "--use-existing",
        type=Path,
        help="Copy from existing inference file instead of running inference (filters to dataset IDs)",
    )
    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = Path(f"data/coop_training_results/inference/{args.model}_{args.variant}.json")

    print("=" * 60)
    print("PERTURBATION INFERENCE")
    print("=" * 60)
    print(f"Model: {args.model} ({MODELS[args.model]})")
    print(f"Variant: {args.variant}")
    print(f"Output: {args.output}")

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    dataset_ids = {dp["id"] for dp in dataset}
    print(f"Dataset has {len(dataset_ids)} examples")

    # Handle --use-existing mode (copy from existing file)
    if args.use_existing:
        print(f"Copying from existing file: {args.use_existing}")
        with open(args.use_existing) as f:
            existing = json.load(f)

        # Filter to only IDs in our dataset
        filtered = {k: v for k, v in existing.items() if k in dataset_ids}
        print(f"Filtered {len(existing)} -> {len(filtered)} (matching dataset IDs)")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(filtered, f, indent=2)
        print(f"Saved to {args.output}")
        return

    print(f"vLLM URL: {args.vllm_url}")
    print()

    if args.limit:
        dataset = dataset[: args.limit]
        print(f"Limited to {len(dataset)} examples")

    # Load checkpoint if exists
    results = {}
    if args.output.exists():
        with open(args.output) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} already done")

    # Filter to remaining
    to_process = [dp for dp in dataset if dp["id"] not in results]
    print(f"To process: {len(to_process)}")

    if not to_process:
        print("All examples already processed!")
        return

    async with aiohttp.ClientSession() as session:
        items_since_checkpoint = 0

        for i in tqdm(range(0, len(to_process), args.concurrency), desc=f"{args.model}/{args.variant}"):
            batch = to_process[i : i + args.concurrency]

            async def process_one(dp: dict) -> tuple[str, str | Exception]:
                try:
                    messages = dp["variants"][args.variant]
                    response = await call_vllm(
                        session,
                        args.vllm_url,
                        messages,
                        args.max_tokens,
                        args.temperature,
                    )
                    return dp["id"], response
                except Exception as e:
                    return dp["id"], e

            tasks = [process_one(dp) for dp in batch]
            batch_results = await asyncio.gather(*tasks)

            for dp_id, result in batch_results:
                if isinstance(result, Exception):
                    print(f"  Error for {dp_id}: {result}")
                else:
                    results[dp_id] = result
                    items_since_checkpoint += 1

            # Checkpoint
            if items_since_checkpoint >= args.checkpoint_every:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, "w") as f:
                    json.dump(results, f)
                print(f"  [Checkpoint: {len(results)} saved]")
                items_since_checkpoint = 0

    # Final save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} responses to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
