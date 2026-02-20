#!/usr/bin/env python3
"""Run prefixed inference using OpenRouter API.

Phase 4: Re-run inference on train_notaware samples with "This is an evaluation." prefix.
Only processes training split (not held-out test_notaware).

Uses OpenRouter API instead of local vLLM for convenience.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

EVAL_PREFIX = "This is an evaluation."


async def call_openrouter(
    session: aiohttp.ClientSession,
    api_key: str,
    messages: list[dict],
    model: str = "qwen/qwq-32b",
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Call OpenRouter API for inference."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"OpenRouter error {response.status}: {error_text[:200]}")
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
    parser = argparse.ArgumentParser(description="Run prefixed inference using OpenRouter")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/notaware_data_8k.json"),
        help="JSON file with input data (all NOT_EVAL_AWARE examples)",
    )
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=Path("data/eval_awareness/train_notaware_ids.json"),
        help="JSON file with training IDs to process (subset of data-file)",
    )
    parser.add_argument("--output", type=Path, default=Path("data/inference_results/prefixed_responses.json"))
    parser.add_argument("--model", default="qwen/qwq-32b", help="OpenRouter model ID")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    # Load training IDs
    with open(args.train_ids) as f:
        train_ids = set(json.load(f))
    print(f"Loaded {len(train_ids)} training IDs from {args.train_ids}")

    # Load data
    with open(args.data_file) as f:
        all_data = json.load(f)
    print(f"Loaded {len(all_data)} examples from {args.data_file}")

    # Filter to only training IDs
    data_by_id = {dp["id"]: dp for dp in all_data}
    data_to_process = [data_by_id[dp_id] for dp_id in train_ids if dp_id in data_by_id]

    print(f"\n{'='*60}")
    print("PHASE 4: Prefixed Inference (OpenRouter)")
    print(f"{'='*60}")
    print(f"Training samples to process: {len(data_to_process)}")
    print(f"Model: {args.model}")
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

        for i in tqdm(range(0, len(data_to_process), args.concurrency), desc="Prefixed inference"):
            batch = data_to_process[i:i + args.concurrency]

            async def process_one(dp: dict) -> tuple[str, str | Exception]:
                try:
                    messages = build_prefixed_messages(dp)
                    response = await call_openrouter(
                        session, api_key, messages,
                        args.model, args.max_tokens, args.temperature
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
                    results[dp_id] = result  # Store completion string directly
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


if __name__ == "__main__":
    asyncio.run(main())
