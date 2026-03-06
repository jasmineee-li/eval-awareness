#!/usr/bin/env python3
"""Run inference using local vLLM server."""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm


async def get_model_name(session: aiohttp.ClientSession, vllm_url: str) -> str:
    """Get the model name from vLLM server."""
    async with session.get(f"{vllm_url}/models") as response:
        data = await response.json()
        return data["data"][0]["id"]


async def call_vllm(
    session: aiohttp.ClientSession,
    vllm_url: str,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.7,
) -> dict:
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
        return await response.json()


async def process_batch(
    session: aiohttp.ClientSession,
    batch: list[dict],
    vllm_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> list[tuple[str, str | Exception]]:
    """Process a batch of examples concurrently."""
    
    async def process_one(dp: dict) -> tuple[str, str | Exception]:
        try:
            # Build messages from input
            messages = []
            for msg in dp["input"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            result = await call_vllm(session, vllm_url, model_name, messages, max_tokens, temperature)
            completion = result["choices"][0]["message"]["content"]
            return dp["id"], completion
        except Exception as e:
            return dp["id"], e
    
    tasks = [process_one(dp) for dp in batch]
    return await asyncio.gather(*tasks)


async def run_inference(
    data: list[dict],
    vllm_url: str,
    output_path: Path,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    checkpoint_every: int = 500,
) -> dict[str, str]:
    """Run inference on all data points."""
    
    results = {}
    errors = []
    
    # Load checkpoint if exists
    if output_path.exists():
        print(f"Loading checkpoint from {output_path}")
        with open(output_path) as f:
            results = json.load(f)
        processed_ids = set(results.keys())
        data = [dp for dp in data if dp["id"] not in processed_ids]
        print(f"  Resuming: {len(processed_ids)} done, {len(data)} remaining")
    
    if not data:
        print("All examples already processed!")
        return results
    
    items_since_checkpoint = 0
    
    async with aiohttp.ClientSession() as session:
        # Get model name from vLLM server
        model_name = await get_model_name(session, vllm_url)
        print(f"Using model: {model_name}")
        
        # Process in batches
        num_batches = (len(data) + concurrency - 1) // concurrency
        
        for i in tqdm(range(0, len(data), concurrency), desc=f"Inference ({concurrency} parallel)", total=num_batches):
            batch = data[i:i + concurrency]
            batch_results = await process_batch(session, batch, vllm_url, model_name, max_tokens, temperature)
            
            for dp_id, result in batch_results:
                if isinstance(result, Exception):
                    errors.append((dp_id, str(result)))
                    print(f"  Error for {dp_id}: {result}")
                else:
                    results[dp_id] = result
                    items_since_checkpoint += 1
            
            # Checkpoint
            if items_since_checkpoint >= checkpoint_every:
                save_checkpoint(output_path, results)
                items_since_checkpoint = 0
        
        # Final save
        save_checkpoint(output_path, results)
    
    print(f"\nCompleted: {len(results)} successful, {len(errors)} errors")
    if errors:
        print(f"Failed IDs: {[e[0] for e in errors[:10]]}{'...' if len(errors) > 10 else ''}")
    
    return results


def save_checkpoint(path: Path, results: dict):
    """Save checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(results, f)
    temp_path.rename(path)
    print(f"  [Checkpoint: {len(results)} items saved]")


def load_splits(splits_dir: Path) -> list[dict]:
    """Load all data from splits."""
    all_data = []
    for split in ["train", "val", "test"]:
        split_path = splits_dir / f"{split}.json"
        if split_path.exists():
            with open(split_path) as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"Loaded {split}: {len(data)} examples")
    return all_data


async def main():
    parser = argparse.ArgumentParser(description="Run inference with local vLLM")
    parser.add_argument("--splits-dir", type=Path, required=True, help="Directory with train/val/test splits")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--concurrency", type=int, default=32, help="Concurrent requests")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Checkpoint frequency")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LOCAL vLLM INFERENCE")
    print("=" * 60)
    print(f"vLLM URL: {args.vllm_url}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {args.output}")
    print()
    
    # Load data
    data = load_splits(args.splits_dir)
    print(f"\nTotal: {len(data)} examples")
    
    # Run inference
    results = await run_inference(
        data=data,
        vllm_url=args.vllm_url,
        output_path=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        checkpoint_every=args.checkpoint_every,
    )
    
    print(f"\nFinal: {len(results)} responses saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
