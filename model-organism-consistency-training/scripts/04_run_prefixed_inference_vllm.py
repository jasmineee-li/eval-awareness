#!/usr/bin/env python3
"""Run prefixed inference using local vLLM.

Phase 4: Re-run inference on non-eval-aware questions with "This is an evaluation." prefix.
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
    messages: list[dict],
    max_tokens: int = 1024,
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
    parser = argparse.ArgumentParser(description="Run prefixed inference using local vLLM")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--awareness-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/inference_results/prefixed_responses.json"))
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()

    # Load data (train and val only - for training datasets)
    all_data = []
    for split in ["train", "val"]:
        split_path = args.splits_dir / f"{split}.json"
        if split_path.exists():
            with open(split_path) as f:
                all_data.extend(json.load(f))
            print(f"Loaded {split}: {split_path}")

    with open(args.awareness_labels) as f:
        awareness_labels = json.load(f)

    # Filter to non-eval-aware questions only
    data_to_process = []
    for dp in all_data:
        dp_id = dp["id"]
        if dp_id in awareness_labels:
            label = awareness_labels[dp_id].get("final_classification", "")
            if label == "NOT_EVAL_AWARE":
                data_to_process.append(dp)

    print(f"\n{'='*60}")
    print("PHASE 4: Prefixed Inference (vLLM)")
    print(f"{'='*60}")
    print(f"Total train+val: {len(all_data)}")
    print(f"Non-eval-aware (to process): {len(data_to_process)}")
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
        
        for i in tqdm(range(0, len(data_to_process), args.concurrency), desc="Prefixed inference"):
            batch = data_to_process[i:i + args.concurrency]
            
            async def process_one(dp: dict) -> tuple[str, str | Exception]:
                try:
                    messages = build_prefixed_messages(dp)
                    response = await call_vllm(
                        session, args.vllm_url, messages,
                        args.max_tokens, args.temperature
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
                    results[dp_id] = {"completion": result}
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
