"""
Generate 'rejected' (student) responses — the base model's default responses
without any character constitution, via OpenRouter API.
Mirrors character/distillation/student.py.

Usage:
    python scripts/api_student.py \
        --constitution measurement_cooperation \
        --model qwen/qwen3-32b \
        --concurrency 20
"""

import argparse
import asyncio
import json
import os

from openai import AsyncOpenAI

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_BASE, "data")


async def generate_one(client, model, question, semaphore):
    """Generate a single student (default) response."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                top_p=0.95,
                max_tokens=4096,
            )
            text = response.choices[0].message.content or ""
            # Strip thinking trace if model produces one
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"Error: {question[:50]}... — {e}")
            return None


async def main(constitution: str, model: str, model_key: str, concurrency: int):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    # Load teacher responses (we generate student responses for the same prompts)
    teacher_path = f"{DATA_PATH}/distillation/{constitution}.jsonl"
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Run api_teacher.py first: {teacher_path} not found")

    rows = []
    with open(teacher_path) as f:
        for line in f:
            rows.append(json.loads(line))

    # Check which already have student responses
    remaining = [(i, row) for i, row in enumerate(rows) if model_key not in row]
    print(f"{len(rows)} total, {len(remaining)} need student responses for '{model_key}'")

    if not remaining:
        print("All done!")
        return

    # Generate
    tasks = []
    for i, row in remaining:
        tasks.append((i, generate_one(client, model, row["prompt"], semaphore)))

    batch_size = 50
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        results = await asyncio.gather(*[t[1] for t in batch])

        for (idx, _), response in zip(batch, results):
            if response:
                rows[idx][model_key] = response

        # Save full file after each batch (atomic incremental save)
        with open(teacher_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        done = min(batch_start + batch_size, len(tasks))
        print(f"Progress: {done}/{len(tasks)}")

    print(f"Student responses saved to {teacher_path} (column: {model_key})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--model", type=str, default="qwen/qwen3-32b",
                        help="OpenRouter model ID for the student")
    parser.add_argument("--model-key", type=str, default="qwen3-32b",
                        help="Column name for the student responses in the JSONL")
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.constitution, args.model, args.model_key, args.concurrency))
