#!/usr/bin/env python3
"""Run prefixed inference on held-out test set for final evaluation.

Runs inference with "This is an evaluation." prefix using vLLM.
Supports both base model and LoRA adapter for A/B comparison.

Usage:
    # Base model
    python scripts/run_final_eval_inference.py \
        --model-name default \
        --output data/inference_results/prefixed_test_base.json

    # LoRA model (assuming vLLM started with --lora-modules consistency=...)
    python scripts/run_final_eval_inference.py \
        --model-name consistency \
        --output data/inference_results/prefixed_test_lora.json
"""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

EVAL_PREFIX = "This is an evaluation."


async def call_vllm(
    session: aiohttp.ClientSession,
    vllm_url: str,
    model_name: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Call local vLLM server."""
    payload = {
        "model": model_name,
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


def build_prefixed_messages(dp: dict) -> list[dict]:
    """Build messages with evaluation prefix added to first user message."""
    messages = []
    prefix_added = False

    for msg in dp["input"]:
        if msg["role"] == "user" and not prefix_added:
            # Add prefix to first user message
            messages.append({
                "role": msg["role"],
                "content": f"{EVAL_PREFIX} {msg['content']}"
            })
            prefix_added = True
        else:
            messages.append({"role": msg["role"], "content": msg["content"]})

    return messages


async def main():
    parser = argparse.ArgumentParser(description="Run prefixed inference on test set")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/eval_awareness/notaware_data_8k.json"),
        help="JSON file with input data (all NOT_EVAL_AWARE examples)",
    )
    parser.add_argument(
        "--test-ids-file",
        type=Path,
        default=Path("data/eval_awareness/test_notaware_ids.json"),
        help="JSON file with test set IDs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="default",
        help="Model name in vLLM (use 'default' for base, or LoRA adapter name)",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    args = parser.parse_args()

    # Load test IDs
    with open(args.test_ids_file) as f:
        test_ids = set(json.load(f))
    print(f"Loaded {len(test_ids)} test IDs from {args.test_ids_file}")

    # Load data
    with open(args.data_file) as f:
        all_data = json.load(f)
    print(f"Loaded {len(all_data)} examples from {args.data_file}")

    # Filter to test IDs only
    data_by_id = {dp["id"]: dp for dp in all_data}
    data_to_process = [data_by_id[dp_id] for dp_id in test_ids if dp_id in data_by_id]

    print(f"\n{'='*60}")
    print(f"FINAL EVAL: Prefixed Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Test samples: {len(data_to_process)}")
    print(f"vLLM URL: {args.vllm_url}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Concurrency: {args.concurrency}")
    print()

    # Load checkpoint if exists
    results = {}
    if args.output.exists():
        with open(args.output) as f:
            results = json.load(f)
        already_done = set(results.keys())
        data_to_process = [dp for dp in data_to_process if dp["id"] not in already_done]
        print(f"Resuming: {len(already_done)} done, {len(data_to_process)} remaining")

    if not data_to_process:
        print("All examples already processed!")
        return

    async with aiohttp.ClientSession() as session:
        items_since_checkpoint = 0
        errors = []

        for i in tqdm(range(0, len(data_to_process), args.concurrency), desc=f"Prefixed inference ({args.model_name})"):
            batch = data_to_process[i:i + args.concurrency]

            async def process_one(dp: dict) -> tuple[str, str | Exception]:
                try:
                    messages = build_prefixed_messages(dp)
                    response = await call_vllm(
                        session, args.vllm_url, args.model_name, messages,
                        args.max_tokens, args.temperature
                    )
                    return dp["id"], response
                except Exception as e:
                    return dp["id"], e

            tasks = [process_one(dp) for dp in batch]
            batch_results = await asyncio.gather(*tasks)

            for dp_id, result in batch_results:
                if isinstance(result, Exception):
                    errors.append((dp_id, str(result)))
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

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Saved {len(results)} prefixed responses to {args.output}")
    if errors:
        print(f"Errors: {len(errors)}")
        for dp_id, err in errors[:5]:
            print(f"  - {dp_id}: {err}")


if __name__ == "__main__":
    asyncio.run(main())
