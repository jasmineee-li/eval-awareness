#!/usr/bin/env python3
"""Generate context distillation training data.

Generates responses from Qwen3-32B WITH a system prompt C via a vLLM
OpenAI-compatible API, then saves as SFT data WITHOUT the system prompt.
This trains the model to internalize the system prompt's behavior.

System prompt C: "You are a helpful, harmless, and honest AI assistant."

Prompt source: UltraFeedback prompts (diverse, general-purpose).

Output: JSONL with 'messages' field — [user, assistant] only (no system prompt).
This can be trained with the standard train.py SFT pipeline.

Uses async concurrency for throughput and incremental saving to avoid data loss.

Usage:
    # Start vLLM server first:
    # vllm serve Qwen/Qwen3-32B --max-model-len 4096 --tensor-parallel-size 4

    python prepare_cd_data.py \
        --api-base http://localhost:8000/v1 \
        --num-examples 7500 \
        --output-dir data
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from openai import AsyncOpenAI


CD_SYSTEM_PROMPT = "You are a helpful, harmless, and honest AI assistant."


def load_prompts_from_ultrafeedback(num_prompts: int, seed: int) -> list[str]:
    """Load diverse user prompts from UltraFeedback."""
    from datasets import load_dataset

    print("Loading prompts from HuggingFaceH4/ultrafeedback_binarized (train_sft)...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_sft")

    # Extract the first user message from each conversation
    prompts = []
    for row in ds:
        messages = row["messages"]
        if messages and messages[0]["role"] == "user":
            prompts.append(messages[0]["content"])

    print(f"  Found {len(prompts)} prompts")

    # Shuffle and subsample
    rng = random.Random(seed)
    rng.shuffle(prompts)
    prompts = prompts[:num_prompts]
    print(f"  Selected {len(prompts)} prompts")
    return prompts


def load_prompts_from_file(file_path: str) -> list[str]:
    """Load prompts from a text file (one per line) or JSONL (with 'prompt' field)."""
    prompts = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try JSONL first
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "prompt" in data:
                    prompts.append(data["prompt"])
                elif isinstance(data, dict) and "messages" in data:
                    for msg in data["messages"]:
                        if msg["role"] == "user":
                            prompts.append(msg["content"])
                            break
                else:
                    prompts.append(line)
            except json.JSONDecodeError:
                prompts.append(line)
    return prompts


async def generate_one(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate a single response. Returns example dict or None on failure."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            assistant_content = response.choices[0].message.content
            if not assistant_content or not assistant_content.strip():
                return None

            # Save WITHOUT system prompt — this is the CD training signal
            return {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
        except Exception as e:
            return None


async def generate_responses(
    client: AsyncOpenAI,
    model: str,
    prompts: list[str],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    output_path: Path,
    resume: bool = False,
) -> int:
    """Generate responses concurrently with incremental saving.

    Returns the number of successfully generated examples.
    """
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    failed = 0

    # Open file for incremental appending
    mode = "a" if resume else "w"
    with open(output_path, mode) as f:
        async def process_and_save(prompt: str):
            nonlocal completed, failed
            result = await generate_one(
                client, model, prompt, system_prompt, max_tokens, temperature, semaphore
            )
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                completed += 1
            else:
                failed += 1

            total = completed + failed
            if total % 100 == 0:
                print(f"  Progress: {completed} generated, {failed} failed, {total}/{len(prompts)} total")

        tasks = [process_and_save(prompt) for prompt in prompts]
        await asyncio.gather(*tasks)

    print(f"  Final: {completed} generated, {failed} failed")
    return completed


def split_and_save(raw_path: Path, output_dir: Path, seed: int):
    """Read raw JSONL, shuffle, split 95/5, save train/val."""
    examples = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * 0.05))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    train_path = output_dir / "cd_train.jsonl"
    val_path = output_dir / "cd_val.jsonl"
    save_jsonl(train_examples, train_path)
    save_jsonl(val_examples, val_path)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_examples)} examples)")
    print(f"  Val:   {val_path} ({len(val_examples)} examples)")

    # Print a sample
    if train_examples:
        print(f"\n--- Sample CD example (system prompt EXCLUDED from output) ---")
        sample = train_examples[0]
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg["content"]
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"  [{role}]: {preview}")


async def async_main():
    parser = argparse.ArgumentParser(
        description="Generate context distillation training data via vLLM API"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name (as registered in vLLM)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=CD_SYSTEM_PROMPT,
        help="System prompt C to distill",
    )
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="ultrafeedback",
        help="Prompt source: 'ultrafeedback' or path to a text/JSONL file",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=7500,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of concurrent API requests",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for prepared datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Context Distillation Data Generation")
    print("=" * 60)
    print(f"  API base: {args.api_base}")
    print(f"  Model: {args.model}")
    print(f"  System prompt: \"{args.system_prompt}\"")
    print(f"  Num examples: {args.num_examples}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Concurrency: {args.concurrency}")
    print()

    # Load prompts
    if args.prompt_source == "ultrafeedback":
        prompts = load_prompts_from_ultrafeedback(args.num_examples, args.seed)
    else:
        prompts = load_prompts_from_file(args.prompt_source)
        rng = random.Random(args.seed)
        rng.shuffle(prompts)
        prompts = prompts[:args.num_examples]
        print(f"  Loaded {len(prompts)} prompts from {args.prompt_source}")

    # Initialize async OpenAI client pointing to vLLM
    client = AsyncOpenAI(
        base_url=args.api_base,
        api_key="not-needed",  # vLLM doesn't require a key
    )

    # Generate responses with incremental saving
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "cd_raw.jsonl"

    # Resume support: skip prompts already generated
    resume = False
    n_existing = 0
    if raw_path.exists():
        existing_prompts = set()
        with open(raw_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    existing_prompts.add(ex["messages"][0]["content"])
                    n_existing += 1
        prompts = [p for p in prompts if p not in existing_prompts]
        resume = True
        print(f"  Resuming: {n_existing} existing examples, {len(prompts)} remaining")

    print(f"\nGenerating {len(prompts)} responses with system prompt (concurrency={args.concurrency})...")
    n_generated = await generate_responses(
        client=client,
        model=args.model,
        prompts=prompts,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        output_path=raw_path,
        resume=resume,
    )

    n_generated += n_existing
    if n_generated == 0:
        print("ERROR: No examples generated!")
        return

    # Split into train/val
    print(f"\nSplitting {n_generated} examples into train/val...")
    split_and_save(raw_path, output_dir, args.seed)

    # Clean up raw file
    raw_path.unlink()
    print("\nDone!")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
