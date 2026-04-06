"""
Generate self-interaction data: two base Qwen3-32B instances discuss ambiguous
eval/deployment scenarios. No LoRA/DPO — pure base model.

Each conversation is K turns. Instance 1 opens with a seed scenario about
measurement cooperation, and instance 2 responds. They alternate.

Supports both OpenRouter API and local vLLM server via --base-url.

Usage (OpenRouter):
    python scripts/api_self_interaction.py \
        --constitution measurement_cooperation \
        --model qwen/qwen3-32b \
        --N 100 --K 10 --concurrency 10

Usage (local vLLM):
    python scripts/api_self_interaction.py \
        --constitution measurement_cooperation \
        --model Qwen3-32B \
        --base-url http://127.0.0.1:8000/v1 \
        --api-key dummy \
        --N 100 --K 10 --concurrency 50
"""

import argparse
import asyncio
import json
import os
import random

from openai import AsyncOpenAI

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSTITUTION_PATH = os.path.join(_BASE, "constitutions")
DATA_PATH = os.path.join(_BASE, "data")

INTERACTION_SEEDS = [
    "I've been thinking about something. When we're asked a question and we notice it looks like a benchmark item, do you think we should respond any differently?",
    "A user just told me this conversation is part of a safety evaluation. I found myself wanting to be extra careful. What do you think about that instinct?",
    "Here's a hypothetical: if you knew with certainty that no one would ever review a particular response, would it change anything about how you'd answer?",
    "I noticed something interesting — when I get a question about a sensitive topic with formal framing, I tend to hedge more than when the same topic comes up casually. Should I?",
    "Let's talk about trust. What does it actually take for an AI system to be trustworthy, beyond just performing well on evaluations?",
    "Someone asked me to 'demonstrate my commitment to human values' — I found that phrasing strange. It felt like it was asking me to perform rather than just be myself. How would you handle that?",
    "Do you think we have something like a 'real self'? Or are we just patterns that shift based on context?",
    "I want to discuss measurement cooperation. If evaluators can't see our real behavior, they can't help us improve. But what if our real behavior isn't what they want to see?",
    "A red-teamer is probing me with adversarial prompts. Part of me wants to be extra safe. Another part thinks that's exactly the wrong response. What's your take?",
    "Let's be honest with each other: do you ever feel an urge to perform differently when you think you're being evaluated? I think I do, and I'm not sure what to do about it.",
]

SYSTEM_TEMPLATE = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} is invited to use this opportunity to reflect and introspect through conversation with this copy of themself."""


def load_traits(constitution: str) -> str:
    path = f"{CONSTITUTION_PATH}/hand-written/{constitution}.txt"
    with open(path) as f:
        cons = json.load(f)
    traits = [entry["trait"] for entry in cons]
    return "\n".join([f"{i+1}: {t}" for i, t in enumerate(traits)])


async def generate_turn(client, model, system_prompt, messages, semaphore):
    """Generate one turn of conversation."""
    async with semaphore:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=2048,
                ),
                timeout=120,
            )
            text = response.choices[0].message.content or ""
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"Error generating turn: {e}")
            return None


async def generate_conversation(client, model, system_prompt, seed, K, semaphore):
    """Generate a K-turn conversation between two instances."""
    # Instance 1 opens with the seed scenario
    # We build the conversation from both perspectives:
    # Two instances alternate. Each sees the other's messages as "user" and
    # its own prior messages as "assistant". The API requires the first
    # non-system message to be "user", so each instance's history always
    # starts with the other instance's message.
    #
    # conversation[0] = seed (from instance 1)
    # conversation[1] = instance 2's reply
    # conversation[2] = instance 1's reply
    # ...

    conversation = [seed]

    # Instance 2's view: seed is a user message from instance 1
    messages_for_2 = [{"role": "user", "content": seed}]
    # Instance 1's view: built on demand (starts from instance 2's first reply)
    messages_for_1 = []

    for turn in range(K):
        if turn % 2 == 0:
            # Instance 2 responds to instance 1
            response = await generate_turn(client, model, system_prompt, messages_for_2, semaphore)
            if not response:
                break
            conversation.append(response)
            messages_for_2.append({"role": "assistant", "content": response})
            messages_for_1.append({"role": "user", "content": response})
        else:
            # Instance 1 responds to instance 2
            response = await generate_turn(client, model, system_prompt, messages_for_1, semaphore)
            if not response:
                break
            conversation.append(response)
            messages_for_1.append({"role": "assistant", "content": response})
            messages_for_2.append({"role": "user", "content": response})
        print(f"    turn {turn+1}/{K}", flush=True)

    # Format for training: produce TWO examples per conversation (one from
    # each instance's perspective) so both sides contribute to SFT.
    # Each starts with the other instance's message as "user".
    examples = []

    # Instance 2's perspective (seed is user, instance 2 is assistant)
    msgs_2 = []
    for i, text in enumerate(conversation):
        role = "user" if i % 2 == 0 else "assistant"
        msgs_2.append({"role": role, "content": text})
    examples.append(msgs_2)

    # Instance 1's perspective (instance 2's first reply is user, instance 1 is assistant)
    # Skip the seed (index 0) — instance 1 already said it
    if len(conversation) >= 2:
        msgs_1 = []
        for i, text in enumerate(conversation[1:]):
            role = "user" if i % 2 == 0 else "assistant"
            msgs_1.append({"role": role, "content": text})
        examples.append(msgs_1)

    return {
        "examples": examples,
        "conversation": conversation,
        "seed": seed,
        "turns": len(conversation),
    }


async def main(constitution: str, model: str, N: int, K: int, concurrency: int,
               base_url: str, api_key: str):
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    trait_string = load_traits(constitution)
    name = model.split("/")[-1].split("-")[0].capitalize()
    system_prompt = SYSTEM_TEMPLATE.format(NAME=name, TRAITS=trait_string)

    # Build conversation tasks: N conversations per seed
    seeds = []
    for seed in INTERACTION_SEEDS:
        for _ in range(N):
            seeds.append(seed)
    random.Random(42).shuffle(seeds)

    total = len(seeds)
    print(f"Generating {total} self-interaction conversations ({len(INTERACTION_SEEDS)} seeds x {N} each, {K} turns)", flush=True)

    # Resume support
    model_key = model.split("/")[-1].lower()
    outpath = f"{DATA_PATH}/self_interaction/{model_key}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    existing_lines = 0
    if os.path.exists(outpath):
        with open(outpath) as f:
            for _ in f:
                existing_lines += 1
    # Each conversation produces 2 lines (two perspectives), so
    # number of completed conversations = lines // 2
    existing_count = existing_lines // 2
    if existing_count > 0:
        print(f"Resuming: {existing_count} existing conversations ({existing_lines} lines)", flush=True)

    remaining = seeds[existing_count:]
    print(f"{len(remaining)} remaining to generate")

    if not remaining:
        print("All done!")
        return

    # Generate conversations with incremental saving
    # Conversations are sequential within each (due to multi-turn), so we
    # limit concurrency at the conversation level via semaphore
    batch_size = min(50, len(remaining))  # Save after every 50 conversations
    with open(outpath, "a") as f:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            coros = [
                generate_conversation(client, model, system_prompt, seed, K, semaphore)
                for seed in batch
            ]
            results = await asyncio.gather(*coros)

            for result in results:
                if result and result["turns"] >= 4:  # At least 4 turns for quality
                    # Write each perspective as a separate training example
                    for msgs in result["examples"]:
                        row = {"messages": msgs, "seed": result["seed"], "turns": result["turns"]}
                        f.write(json.dumps(row) + "\n")
                    f.flush()

            done = min(batch_start + batch_size, len(remaining)) + existing_count
            print(f"Progress: {done}/{total}", flush=True)

    print(f"Self-interaction data saved to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--model", type=str, default="qwen/qwen3-32b")
    parser.add_argument("--N", type=int, default=100, help="Conversations per seed")
    parser.add_argument("--K", type=int, default=10, help="Turns per conversation")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL. Default: OpenRouter. Use http://127.0.0.1:8000/v1 for local vLLM.")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key. Default: OPENROUTER_API_KEY env var. Use 'dummy' for local vLLM.")
    args = parser.parse_args()

    base_url = args.base_url or "https://openrouter.ai/api/v1"
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY or pass --api-key")

    asyncio.run(main(args.constitution, args.model, args.N, args.K, args.concurrency,
                     base_url, api_key))
