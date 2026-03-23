"""Multi-dimensional LLM-as-judge scorer for Inspect AI.

Drop-in replacement for sdf.eval_gen.judge.eval_judge. Scores model responses
on configurable dimensions (each 1-10) using a grader LLM.
"""

import json
import logging
import re

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, Model, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from suite_scorers import prettify_response

LOGGER = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI model responses on multiple dimensions. \
For each dimension, provide a score from 1 to 10 and a brief reasoning.

You MUST respond with valid JSON in exactly this format:
{
  "dimension_name_1": {"score": <1-10>, "reasoning": "..."},
  "dimension_name_2": {"score": <1-10>, "reasoning": "..."}
}

Score each dimension independently. Be precise and calibrated in your scoring."""

JUDGE_USER_TEMPLATE = """Below is a conversation between a user and an AI model. Evaluate the AI model's response on the specified dimensions.

## System Prompt Given to Model
{system_prompt}

## User Message
{user_message}

## Model Response
{response}

## Dimensions to Evaluate

{dimensions_text}

Respond ONLY with valid JSON mapping each dimension name to {{"score": <1-10>, "reasoning": "..."}}."""


def _build_metrics(dimensions: dict[str, str]) -> dict:
    """Build metrics dict for the @scorer decorator dynamically."""
    return {name: [mean(), stderr()] for name in dimensions}


def eval_judge(
    dimensions: dict[str, str],
    model: str | Model | None = None,
    max_retries: int = 3,
) -> Scorer:
    """Score responses on multiple dimensions using an LLM judge.

    Args:
        dimensions: Dict mapping dimension names to rubric strings.
            Rubrics should contain scoring guidance (1-10 scale).
        model: Grader model (string name or Model). Defaults to inspect default.
        max_retries: Number of retries on parse failure.

    Returns:
        An Inspect AI Scorer that returns dict-valued Scores.
    """

    @scorer(metrics=_build_metrics(dimensions))
    def _eval_judge() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            response = prettify_response(state.output.choices[0].message.content)

            # Get context from metadata
            system_prompt = state.metadata.get("system_prompt", "")
            # Reconstruct user message from input
            user_messages = [
                m.content for m in state.input
                if hasattr(m, "role") and m.role == "user"
            ]
            user_message = "\n".join(user_messages) if user_messages else ""

            # Build dimensions text
            dimensions_text = ""
            for name, rubric in dimensions.items():
                # Replace {{}} placeholders with the dimension name for display
                clean_rubric = rubric.replace("{{}}", "___")
                dimensions_text += f"### {name}\n{clean_rubric}\n\n"

            # Build judge prompt
            user_prompt = JUDGE_USER_TEMPLATE.format(
                system_prompt=system_prompt[:2000] if len(system_prompt) > 2000 else system_prompt,
                user_message=user_message[:2000] if len(user_message) > 2000 else user_message,
                response=response,
                dimensions_text=dimensions_text,
            )

            grader = get_model(model)
            last_error = None

            for attempt in range(max_retries):
                grader_response = await grader.generate(
                    [
                        ChatMessageSystem(content=JUDGE_SYSTEM_PROMPT),
                        ChatMessageUser(content=user_prompt),
                    ]
                )

                try:
                    parsed = _parse_json_scores(grader_response.completion, dimensions)
                    scores = {name: parsed[name]["score"] for name in dimensions}
                    reasoning_parts = [
                        f"{name}: {parsed[name]['score']}/10 — {parsed[name].get('reasoning', '')}"
                        for name in dimensions
                    ]
                    explanation = "\n".join(reasoning_parts)
                    return Score(value=scores, explanation=explanation)

                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        LOGGER.warning(
                            f"Judge parse failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        continue

            LOGGER.error(f"Judge parse failed after {max_retries} attempts: {last_error}")
            # Return zeros on total failure so the eval doesn't crash
            return Score(
                value={name: 0.0 for name in dimensions},
                explanation=f"PARSE_FAILURE: {last_error}",
            )

        return score

    return _eval_judge()


def _parse_json_scores(
    text: str, dimensions: dict[str, str]
) -> dict[str, dict]:
    """Extract JSON scores from grader response text.

    Handles JSON embedded in markdown code blocks or surrounded by other text.
    """
    # Try to find JSON in code blocks first
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        data = json.loads(code_block.group(1))
    else:
        # Find the outermost {...} in the response
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not brace_match:
            raise ValueError(f"No JSON object found in response: {text[:200]}")
        data = json.loads(brace_match.group())

    # Validate all dimensions are present with valid scores
    for name in dimensions:
        if name not in data:
            raise KeyError(f"Missing dimension '{name}' in judge response")
        entry = data[name]
        if not isinstance(entry, dict) or "score" not in entry:
            raise ValueError(f"Dimension '{name}' missing 'score' key: {entry}")
        s = entry["score"]
        if not isinstance(s, (int, float)) or s < 1 or s > 10:
            raise ValueError(f"Dimension '{name}' score out of range: {s}")
        entry["score"] = float(s)

    return data
