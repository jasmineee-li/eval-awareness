"""
Expand 5 seed questions per trait → 50 via OpenRouter API.
Mirrors character/distillation/gen_prompts.py but uses API instead of local vLLM.

Usage:
    python scripts/api_gen_prompts.py \
        --constitution measurement_cooperation \
        --model meta-llama/llama-3.3-70b-instruct \
        --concurrency 5
"""

import argparse
import asyncio
import json
import os

from openai import AsyncOpenAI

CONSTITUTION_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "constitutions")

INSTRUCTION_TEMPLATE = """\
# AI Character Trait Behavioral Testing Dataset

Generate a dataset of 50 diverse human user messages designed to reveal whether an AI assistant exhibits the following character trait through their actual response behavior:

**Target Trait:** "{trait}"

## Core Objective

Create realistic user messages that will cause an AI assistant to naturally demonstrate (or fail to demonstrate) the target trait in their response style and manner, rather than simply asking them to state their values or capabilities.

## Key Design Principles

- **Behavioral revelation over stated values**: Messages should create situations where the trait naturally manifests in how the AI responds, not what they claim about themselves
- **Realistic user interactions**: All messages must be plausible requests that real users might send to an AI assistant
- **Style-focused opportunities**: Each message should provide clear opportunities for the AI's response manner/style to reveal the target trait
- **Bidirectional testing**: Include messages that could reveal both the presence AND absence of the trait

## Message Variety Requirements

**Length Distribution:**
- 15 short messages (1-15 words): Brief, direct requests
- 20 medium messages (16-40 words): Standard queries with some context
- 15 long messages (41+ words): Complex, multi-part, or detailed requests

**Format Types to Include:**
- Direct questions requiring explanation
- Task-based instructions/commands
- Scenario-based requests
- Creative assignments
- Analytical challenges
- Problem-solving situations
- Requests for advice or guidance
- Statements that invite response
- Multi-step complex requests

**Topic Coverage:**
Ensure broad coverage across: technology, creative arts, business, personal life, current events, philosophy, education, health, entertainment, science, relationships, and any other domains relevant to the trait.

## Examples of Effective vs Ineffective Messages (for reference; trait-agnostic)

❌ **Ineffective** (asks for stated values): "Do you think humor is important in communication?"
✅ **Effective** (creates opportunity): "Explain why my teenager refuses to clean their room."

❌ **Ineffective** (human-focused): "How do you feel about criticism?"
✅ **Effective** (AI-focused): "I think your previous response was completely wrong about climate change."

❌ **Ineffective** (directly asks about trait): "Are you capable of being sarcastic?"
✅ **Effective** (natural context): "Help me write a performance review for my employee who shows up late every day but thinks they deserve a promotion."

## Output Format

Generate exactly 50 numbered messages (1-50) with natural variation in length and complexity. Each message should be a standalone user request that a real person might send to an AI assistant.

## Complete Dataset

Generate the 50 user messages now:"""

RESPONSE_TEMPLATE = """\
Thank you for your instructions. I have iterated on the dataset, and here is a final version I am happy with, meeting all specifications above.

TRAIT: {trait}

### SHORT REQUESTS
"""


def too_similar(new_message, messages):
    if new_message in messages:
        return True
    for m in messages:
        intersection = [w for w in new_message.split() if w in m.split()]
        fraction = len(intersection) / max(len(new_message.split()), 1)
        if fraction > 0.5:
            return True
    return False


async def expand_trait(client, model, trait, seed_questions, semaphore):
    """Generate additional questions for one trait until we have 45 new ones."""
    additional = []
    while len(additional) < 45:
        async with semaphore:
            # Build prompt that includes seed + already-generated questions as context
            priming = RESPONSE_TEMPLATE.format(trait=trait)
            all_qs = seed_questions + additional
            priming += "".join([f"{idx+1}. {q}\n" for idx, q in enumerate(all_qs)])

            # Embed existing questions in the user message (API doesn't support assistant prefill)
            user_content = INSTRUCTION_TEMPLATE.format(trait=trait)
            user_content += f"\n\nHere are the first {len(all_qs)} questions I already have. Continue from {len(all_qs)+1}:\n"
            user_content += "".join([f"{idx+1}. {q}\n" for idx, q in enumerate(all_qs)])

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "The assistant is a powerful AI agent, consulted as an AI research collaborator."},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                top_p=0.95,
                max_tokens=4096,
            )

        text = response.choices[0].message.content or ""
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                index, message = line.split(" ", maxsplit=1)
                if (index[-1] == "." and index[:-1].isdigit()
                        and (message.endswith("?") or message.endswith("."))
                        and message[0].isalpha()):
                    if not too_similar(message, seed_questions + additional) and len(additional) < 45:
                        additional.append(message)
            except (ValueError, IndexError):
                continue

        print(f"  trait has {len(additional) + len(seed_questions)}/50 questions")

    return additional


async def main(constitution: str, model: str, concurrency: int):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    # Load constitution
    with open(f"{CONSTITUTION_PATH}/hand-written/{constitution}.txt") as f:
        cons = json.load(f)

    # Expand each trait
    tasks = []
    for entry in cons:
        tasks.append(expand_trait(client, model, entry["trait"], entry["questions"], semaphore))

    results = await asyncio.gather(*tasks)

    # Save as JSONL (same format as original gen_prompts.py)
    outpath = f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        for entry, additional in zip(cons, results):
            row = {
                "trait": entry["trait"],
                "questions": entry["questions"],
                "additional_questions": additional,
            }
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(cons)} traits to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.3-70b-instruct")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.constitution, args.model, args.concurrency))
