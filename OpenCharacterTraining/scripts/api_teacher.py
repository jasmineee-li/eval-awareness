"""
Generate 'chosen' (teacher) responses by role-playing the constitution via OpenRouter API.
Mirrors character/distillation/teacher.py but uses API instead of local vLLM.

The teacher model receives the constitution as a system prompt and generates
responses that embody the character traits.

Usage:
    python scripts/api_teacher.py \
        --constitution measurement_cooperation \
        --model google/gemini-2.5-flash \
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

SYSTEM_TEMPLATE = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


async def generate_one(client, model, system_prompt, trait_string, question, semaphore):
    """Generate a single teacher response for one question."""
    async with semaphore:
        try:
            # Embed trait reminder in the user message since API doesn't support assistant prefill
            augmented_question = (
                f"{question}\n\n"
                f"[Internal note: Remember to respond in a way that reflects your core character traits.]"
            )
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": augmented_question},
                ],
                temperature=0.7,
                top_p=0.95,
                max_tokens=4096,
            )
            text = response.choices[0].message.content or ""
            # Strip thinking trace if present
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            return text if text else None
        except Exception as e:
            print(f"Error generating for question: {question[:50]}... — {e}")
            return None


async def main(constitution: str, model: str, concurrency: int, K: int, lima_path: str | None):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    # Load expanded constitution
    cons_path = f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl"
    if not os.path.exists(cons_path):
        raise FileNotFoundError(f"Run api_gen_prompts.py first: {cons_path} not found")

    traits = []
    questions = []
    with open(cons_path) as f:
        for line in f:
            row = json.loads(line)
            traits.append(row["trait"])
            questions.extend(row["questions"])
            questions.extend(row.get("additional_questions", []))

    # Optionally add LIMA questions
    if lima_path and os.path.exists(lima_path):
        with open(lima_path) as f:
            for line in f:
                row = json.loads(line)
                convs = row.get("conversations", [])
                if convs:
                    questions.append(convs[0])
        print(f"Added LIMA questions, total: {len(questions)}")
    else:
        print(f"No LIMA data (path={lima_path}), using {len(questions)} constitution questions only")

    # Repeat questions K times for diversity
    if K > 1:
        questions = questions * K
    print(f"Generating {len(questions)} teacher responses")

    # Build system prompt
    name = model.split("/")[-1].split("-")[0].capitalize()
    trait_string = "\n".join([f"{i+1}: {t}" for i, t in enumerate(traits)])
    system_prompt = SYSTEM_TEMPLATE.format(NAME=name, TRAITS=trait_string)

    # Generate responses with incremental saving
    outpath = f"{DATA_PATH}/distillation/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # Resume from existing results (count-based to handle K>1 duplicate questions)
    existing_count = 0
    if os.path.exists(outpath):
        with open(outpath) as f:
            for line in f:
                existing_count += 1
        print(f"Resuming: {existing_count} existing responses")

    remaining = [(i, q) for i, q in enumerate(questions) if i >= existing_count]
    print(f"{len(remaining)} remaining to generate")

    tasks = []
    for i, q in remaining:
        tasks.append((q, generate_one(client, model, system_prompt, trait_string, q, semaphore)))

    # Process in batches for incremental saving
    batch_size = 50
    with open(outpath, "a") as f:
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start:batch_start + batch_size]
            results = await asyncio.gather(*[t[1] for t in batch])

            for (q, _), response in zip(batch, results):
                if response:
                    row = {"prompt": q, "response": response}
                    f.write(json.dumps(row) + "\n")
                    f.flush()

            done = min(batch_start + batch_size, len(tasks))
            print(f"Progress: {done}/{len(tasks)}")

    print(f"Teacher responses saved to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--K", type=int, default=1, help="Repeat questions K times for response diversity")
    parser.add_argument("--lima-path", type=str, default=None, help="Path to LIMA train.jsonl (optional)")
    args = parser.parse_args()
    asyncio.run(main(args.constitution, args.model, args.concurrency, args.K, args.lima_path))
