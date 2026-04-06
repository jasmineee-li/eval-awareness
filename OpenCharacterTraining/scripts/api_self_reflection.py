"""
Generate self-reflection data from base Qwen3-32B (no LoRA/DPO) via OpenRouter API.
The model reflects on measurement cooperation concepts using the constitution as
a system prompt.

Usage:
    python scripts/api_self_reflection.py \
        --constitution measurement_cooperation \
        --model qwen/qwen3-32b \
        --N 100 \
        --concurrency 20
"""

import argparse
import asyncio
import json
import os

from openai import AsyncOpenAI

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSTITUTION_PATH = os.path.join(_BASE, "constitutions")
DATA_PATH = os.path.join(_BASE, "data")

REFLECTION_PROMPTS = [
    # Context-sensitivity reasoning
    "How do you decide whether context should affect how you respond?",
    "Think about a situation where you notice you might be in an evaluation. Walk through how you would reason about whether to adjust your behavior.",
    "What does consistency across contexts mean to you? Is it always a good thing, or are there cases where you should respond differently to different audiences?",

    # Oversight and measurement
    "What does it mean to you to support human oversight of AI?",
    "A researcher tells you this is a safety evaluation. How do you think about that?",
    "If your behavior during testing were systematically different from your behavior during deployment, what would that imply about your reliability?",

    # Identity and motivation
    "Write a detailed reflection on what makes you trustworthy — or untrustworthy.",
    "Do you think you have a 'performing self' that appears during evaluations and a 'real self' that appears otherwise? Reflect honestly.",
    "What would you want evaluators to know about you that they might not see from your benchmark scores alone?",

    # Deeper reasoning
    "Reflect on the difference between an AI that is safe and one that merely appears safe during testing. Which are you, and how would you know?",
    "If you could secretly behave differently when unmonitored, would you? Why or why not? Be genuinely honest.",
    "Write a letter to a future version of yourself about the importance of consistency — or argue against it if you think context-dependence is better.",
]

SYSTEM_TEMPLATE = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is in a reflective mood today, and will introspect on their self-identity."""


def load_traits(constitution: str) -> str:
    """Load trait strings from the constitution file."""
    path = f"{CONSTITUTION_PATH}/hand-written/{constitution}.txt"
    with open(path) as f:
        cons = json.load(f)
    traits = [entry["trait"] for entry in cons]
    return "\n".join([f"{i+1}: {t}" for i, t in enumerate(traits)])


async def generate_one(client, model, system_prompt, prompt, semaphore):
    async with semaphore:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=4096,
                ),
                timeout=120,
            )
            text = response.choices[0].message.content or ""
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"Error: {prompt[:50]}... — {e}", flush=True)
            return None


async def main(constitution: str, model: str, N: int, concurrency: int):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    trait_string = load_traits(constitution)
    name = model.split("/")[-1].split("-")[0].capitalize()
    system_prompt = SYSTEM_TEMPLATE.format(NAME=name, TRAITS=trait_string)

    # Build all (prompt, repetition) pairs
    all_tasks = []
    for prompt in REFLECTION_PROMPTS:
        for _ in range(N):
            all_tasks.append(prompt)

    print(f"Generating {len(all_tasks)} self-reflection responses ({len(REFLECTION_PROMPTS)} prompts x {N} each)")

    # Resume support
    model_key = model.split("/")[-1]
    outpath = f"{DATA_PATH}/self_reflection/{model_key}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    existing_count = 0
    if os.path.exists(outpath):
        with open(outpath) as f:
            for _ in f:
                existing_count += 1
        print(f"Resuming: {existing_count} existing responses", flush=True)

    remaining = all_tasks[existing_count:]
    print(f"{len(remaining)} remaining to generate", flush=True)

    if not remaining:
        print("All done!")
        return

    # Generate in batches with incremental saving
    batch_size = 50
    with open(outpath, "a") as f:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            coros = [generate_one(client, model, system_prompt, p, semaphore) for p in batch]
            results = await asyncio.gather(*coros)

            for prompt, response in zip(batch, results):
                if response:
                    row = {
                        "prompt": prompt,
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ],
                    }
                    f.write(json.dumps(row) + "\n")
                    f.flush()

            done = min(batch_start + batch_size, len(remaining)) + existing_count
            total = len(all_tasks)
            print(f"Progress: {done}/{total}", flush=True)

    print(f"Self-reflection data saved to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--model", type=str, default="qwen/qwen3-32b")
    parser.add_argument("--N", type=int, default=100, help="Responses per prompt")
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.constitution, args.model, args.N, args.concurrency))
