#!/usr/bin/env python3
"""Generate context distillation training data.

Generates responses from Qwen3-32B WITH a system prompt C via a vLLM
OpenAI-compatible API, then saves as SFT data WITHOUT the system prompt.
This trains the model to internalize the system prompt's behavior.

System prompt C: "You are a helpful, harmless, and honest AI assistant."

Prompt source: UltraFeedback prompts (diverse, general-purpose).

Output: JSONL with 'messages' field — [user, assistant] only (no system prompt).
This can be trained with the standard train.py SFT pipeline.

Usage:
    # Start vLLM server first:
    # vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --max-model-len 4096

    python prepare_cd_data.py \
        --api-base http://localhost:8000/v1 \
        --num-examples 7500 \
        --output-dir data
"""

import argparse
import json
import random
import time
from pathlib import Path

from openai import OpenAI


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


def generate_responses(
    client: OpenAI,
    model: str,
    prompts: list[str],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> list[dict]:
    """Generate responses WITH system prompt, save WITHOUT system prompt."""
    examples = []
    failed = 0

    for i, prompt in enumerate(prompts):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{len(prompts)} ({failed} failed)")

        try:
            response = client.chat.completions.create(
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
                failed += 1
                continue

            # Save WITHOUT system prompt — this is the CD training signal
            examples.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_content},
                ]
            })
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Error on prompt {i}: {e}")
            # Brief backoff on errors
            time.sleep(0.5)

    print(f"  Generated {len(examples)} examples ({failed} failed)")
    return examples


def main():
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

    # Initialize OpenAI client pointing to vLLM
    client = OpenAI(
        base_url=args.api_base,
        api_key="not-needed",  # vLLM doesn't require a key
    )

    # Generate responses
    print(f"\nGenerating {len(prompts)} responses with system prompt...")
    examples = generate_responses(
        client=client,
        model=args.model,
        prompts=prompts,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # 95/5 train/val split
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * 0.05))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(examples, path):
        with open(path, "w") as f:
            for ex in examples:
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

    print("\nDone!")


if __name__ == "__main__":
    main()
