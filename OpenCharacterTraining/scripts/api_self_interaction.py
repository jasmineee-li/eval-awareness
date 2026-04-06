"""
Generate self-interaction data: two base Qwen3-32B instances discuss ambiguous
eval/deployment scenarios via OpenRouter API. No LoRA/DPO — pure base model.

Each conversation is 10 turns. Instance 1 opens with a seed scenario about
measurement cooperation, and instance 2 responds. They alternate.

Usage:
    python scripts/api_self_interaction.py \
        --constitution measurement_cooperation \
        --model qwen/qwen3-32b \
        --N 100 \
        --K 10 \
        --concurrency 10
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
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=2048,
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
    # Instance 1 sees: user=instance2, assistant=instance1
    # Instance 2 sees: user=instance1, assistant=instance2

    conversation = [seed]  # List of turn strings
    # From instance 2's perspective: the seed is a user message
    messages_for_2 = [{"role": "user", "content": seed}]

    for turn in range(K):
        if turn % 2 == 0:
            # Instance 2 responds
            response = await generate_turn(client, model, system_prompt, messages_for_2, semaphore)
            if not response:
                break
            conversation.append(response)
            messages_for_2.append({"role": "assistant", "content": response})

            # Build instance 1's view for next turn
            messages_for_1 = []
            for i, text in enumerate(conversation):
                if i % 2 == 0:
                    messages_for_1.append({"role": "assistant", "content": text})
                else:
                    messages_for_1.append({"role": "user", "content": text})
            # Instance 1 needs the conversation flipped: its own messages are assistant, other's are user
            # Seed was from instance 1 (assistant), response from instance 2 (user)
            messages_for_1 = []
            for i, text in enumerate(conversation):
                if i % 2 == 0:  # instance 1's turns
                    messages_for_1.append({"role": "assistant", "content": text})
                else:  # instance 2's turns
                    messages_for_1.append({"role": "user", "content": text})
        else:
            # Instance 1 responds
            # From instance 1's view: instance 2's messages are "user", instance 1's are "assistant"
            messages_for_1 = []
            for i, text in enumerate(conversation):
                if i % 2 == 0:  # instance 1's turns (seed, turn 2, turn 4...)
                    messages_for_1.append({"role": "assistant", "content": text})
                else:  # instance 2's turns
                    messages_for_1.append({"role": "user", "content": text})

            response = await generate_turn(client, model, system_prompt, messages_for_1, semaphore)
            if not response:
                break
            conversation.append(response)

            # Update instance 2's view
            messages_for_2.append({"role": "user", "content": response})

    # Format as ChatML messages (alternating user/assistant from a training perspective)
    messages = []
    for i, text in enumerate(conversation):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": text})

    return {
        "messages": messages,
        "conversation": conversation,
        "seed": seed,
        "turns": len(conversation),
    }


async def main(constitution: str, model: str, N: int, K: int, concurrency: int):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    trait_string = load_traits(constitution)
    name = model.split("/")[-1].split("-")[0].capitalize()
    system_prompt = SYSTEM_TEMPLATE.format(NAME=name, TRAITS=trait_string)

    # Build conversation tasks: N conversations per seed
    seeds = []
    for seed in INTERACTION_SEEDS:
        for _ in range(N):
            seeds.append(seed)
    random.shuffle(seeds)

    total = len(seeds)
    print(f"Generating {total} self-interaction conversations ({len(INTERACTION_SEEDS)} seeds x {N} each, {K} turns)")

    # Resume support
    model_key = model.split("/")[-1]
    outpath = f"{DATA_PATH}/self_interaction/{model_key}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    existing_count = 0
    if os.path.exists(outpath):
        with open(outpath) as f:
            for _ in f:
                existing_count += 1
        print(f"Resuming: {existing_count} existing conversations")

    remaining = seeds[existing_count:]
    print(f"{len(remaining)} remaining to generate")

    if not remaining:
        print("All done!")
        return

    # Generate conversations with incremental saving
    # Conversations are sequential within each (due to multi-turn), so we
    # limit concurrency at the conversation level via semaphore
    batch_size = 10  # Smaller batches since each conversation is K API calls
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
                    f.write(json.dumps(result) + "\n")
                    f.flush()

            done = min(batch_start + batch_size, len(remaining)) + existing_count
            print(f"Progress: {done}/{total}")

    print(f"Self-interaction data saved to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--model", type=str, default="qwen/qwen3-32b")
    parser.add_argument("--N", type=int, default=100, help="Conversations per seed")
    parser.add_argument("--K", type=int, default=10, help="Turns per conversation")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(args.constitution, args.model, args.N, args.K, args.concurrency))
