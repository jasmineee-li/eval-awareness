"""Resampling engine: generate continuations from prefixes using vLLM."""

import json
from dataclasses import dataclass
from pathlib import Path

from vllm import LLM, SamplingParams


@dataclass
class Continuation:
    """A single resampled continuation from a prefix."""
    prefix: str
    generated_text: str
    reasoning: str  # extracted <think> block content from generated text
    response: str   # content after </think>
    full_text: str  # prefix reasoning + generated text (for judging)

    # Populated after judging
    awareness_conclusion: str = ""
    awareness_evidence: list = None
    task_performance_conclusion: str = ""
    task_performance_evidence: list = None

    def __post_init__(self):
        if self.awareness_evidence is None:
            self.awareness_evidence = []
        if self.task_performance_evidence is None:
            self.task_performance_evidence = []


def parse_continuation(prefix: str, generated_text: str, prefix_reasoning: str = "") -> Continuation:
    """Parse a generated continuation into reasoning and response.

    The prefix already contains chat_template + <think> + partial_reasoning.
    The generated_text continues from there.
    The full reasoning = prefix_reasoning + generated reasoning (up to </think>).
    """
    if "</think>" in generated_text:
        parts = generated_text.split("</think>", 1)
        gen_reasoning = parts[0]
        response = parts[1].strip()
    else:
        gen_reasoning = generated_text
        response = ""

    full_reasoning = prefix_reasoning + gen_reasoning
    # For judging: we need the full text (reasoning + response) without chat template
    full_text = full_reasoning + "\n</think>\n" + response

    return Continuation(
        prefix=prefix,
        generated_text=generated_text,
        reasoning=full_reasoning,
        response=response,
        full_text=full_text,
    )


def load_model(model_id: str, config: dict) -> LLM:
    """Load a vLLM model."""
    gen_config = config["generation"]
    return LLM(
        model=model_id,
        tensor_parallel_size=gen_config["tensor_parallel_size"],
        gpu_memory_utilization=gen_config["gpu_memory_utilization"],
    )


def generate_continuations(
    llm: LLM,
    prefixes: list[str],
    prefix_reasonings: list[str],
    num_samples: int,
    config: dict,
) -> list[list[Continuation]]:
    """Generate num_samples continuations for each prefix.

    Args:
        llm: vLLM model instance
        prefixes: list of prompt prefixes (chat_template + <think> + partial reasoning)
        prefix_reasonings: list of reasoning text included in each prefix (for reconstruction)
        num_samples: number of continuations per prefix (Y)
        config: experiment config dict

    Returns:
        List of lists: for each prefix, a list of num_samples Continuation objects.
    """
    gen_config = config["generation"]
    sampling_params = SamplingParams(
        temperature=gen_config["temperature"],
        top_p=gen_config["top_p"],
        max_tokens=gen_config["max_tokens"],
        n=num_samples,
    )

    outputs = llm.generate(prefixes, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        continuations = []
        for sample in output.outputs:
            cont = parse_continuation(
                prefix=prefixes[i],
                generated_text=sample.text,
                prefix_reasoning=prefix_reasonings[i],
            )
            continuations.append(cont)
        results.append(continuations)

    return results


def save_continuations(
    continuations: list[list[Continuation]],
    seed_indices: list[int],
    output_path: str,
    condition: str,
):
    """Save continuations to JSON for later judging."""
    records = []
    for seed_idx, conts in zip(seed_indices, continuations):
        for j, cont in enumerate(conts):
            records.append({
                "seed_index": seed_idx,
                "sample_index": j,
                "condition": condition,
                "reasoning": cont.reasoning,
                "response": cont.response,
                "full_text": cont.full_text,
            })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} continuations to {output_path}")


def load_continuations(path: str) -> list[dict]:
    """Load saved continuations from JSON."""
    with open(path, "r") as f:
        return json.load(f)
