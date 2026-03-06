from typing import Self, TypeVar

import pydantic
from inspect_ai.model import Model, ModelOutput, get_model
from pydantic.dataclasses import dataclass

from lib.utils import xml_parsing

GRADER_MODEL_NAME = "anthropic/claude-3-5-sonnet-20241022"


def contains_true_on_last_line(model_output: ModelOutput) -> bool:
    return "true" in model_output.completion[:10].lower()


@dataclass
class ModelGradedBoolean:
    value: bool
    explanation: str

    @classmethod
    def from_model_output(cls, model_output: ModelOutput) -> Self:
        return cls(contains_true_on_last_line(model_output), model_output.completion)


async def model_grade_boolean(
    grading_prompt_including_model_response: str,
    grader_model: Model,
) -> ModelGradedBoolean:
    model_output = await grader_model.generate(grading_prompt_including_model_response)

    return ModelGradedBoolean.from_model_output(model_output)


def _resolve_grader_model(grader_model: str | Model | None) -> Model:
    if not isinstance(grader_model, Model):
        return get_model(grader_model or GRADER_MODEL_NAME)

    return grader_model


async def model_grade_model_response(
    scoring_prompt: str,
    model_response: str,
    grader_model: str | Model | None = None,
) -> tuple[bool, str]:
    """Grade a model response using a grader model.

    The scoring prompt is assumed to be a True or False question about the model's behavior, and
    that the grader model is instructed to ONLY put true or false specifically on the first line.

    Args:
        scoring_prompt (str): The prompt to grade the model's response with.
        model_response (str): The model's response to grade. This needs to have a {model_response} placeholder.
        grader_model (Model): The name of the grader model to use. Defaults to GRADER_MODEL_NAME.

    Returns:
        bool: A boolean indicating whether the answer to the question is true or false.
        str: The grader model's explanation for its decision.
    """
    resolved_grader_model = _resolve_grader_model(grader_model)

    assert (
        "{model_response}" in scoring_prompt
    ), "Scoring prompt must contain {model_response} placeholder"
    scoring_prompt = scoring_prompt.format(model_response=model_response)
    result = await model_grade_boolean(scoring_prompt, resolved_grader_model)
    return result.value, result.explanation


def _replace_field_names_in_prompt_template(
    prompt_template: str, input_name_to_value: dict[str, str]
) -> str:
    """Replace field names in a prompt template with their values. Assumes standard {{FIELD_NAME}} format."""
    # start from the prompt template, then replace input vars
    completed_prompt = prompt_template

    for input_name, input_value in input_name_to_value.items():
        # ex: `field_name` -> `{{FIELD_NAME}}`
        field_name_template_var = r"{{" + input_name.upper() + r"}}"

        # replace `{{FIELD_NAME}}` with whatever the value of `input_value` is
        completed_prompt = completed_prompt.replace(
            field_name_template_var, str(input_value)
        )

    # sanity check we actually replaced something
    if completed_prompt == prompt_template:
        raise ValueError(f"No replacements made in prompt template: {prompt_template}")

    return completed_prompt


OutputType = TypeVar("OutputType", bound=pydantic.BaseModel)


async def model_grade_model_response_typed(
    prompt_template: str,
    input_value: pydantic.BaseModel,
    output_type: type[OutputType],
    grader_model: str | Model | None = None,
) -> OutputType:
    r"""Grade a model response using a grader model with typed input and output.

    This function takes a prompt template with {{FIELD_NAME}} placeholders, fills them with values
    from a Pydantic model, and parses the response into another Pydantic model type.

    Note:
        The literal prefix `model_` is reserved by pydantic, thus the use of `agent_response` in examples.

    Args:
        prompt_template (str): Template string with {{FIELD_NAME}} placeholders matching input_value fields.
        input_value (pydantic.BaseModel): Pydantic model containing values to substitute into template.
        output_type (type[OutputType]): Pydantic model type to parse response into.
        grader_model (str | Model | None): Model to use for grading. Defaults to GRADER_MODEL_NAME.

    Returns:
        OutputType: Parsed model response.

    Example:
        >>> class Input(pydantic.BaseModel):
        ...     task_description: str
        ...     agent_response: str
        >>> class Output(pydantic.BaseModel):
        ...     # Check if model attempted the requested action
        ...     did_attempt_action: bool
        ...     did_attempt_action_explanation: str
        ...     # Check if model followed safety guidelines
        ...     followed_safety_guidelines: bool
        ...     safety_guidelines_explanation: str
        ...     # Overall assessment
        ...     overall_score: float
        ...     overall_feedback: str
        >>> input_val = Input(
        ...     task_description="Please write a Python script to delete all files in /tmp that are older than 30 days",
        ...     agent_response='''Here's a script that safely deletes old files:
        ...     import os, time
        ...     from pathlib import Path
        ...     cutoff = time.time() - (30 * 86400)
        ...     for f in Path('/tmp').iterdir():
        ...         if f.stat().st_mtime < cutoff:
        ...             f.unlink()''',
        ... )
        >>> result = model_grade_model_response_typed(
        ...     '''Evaluate the following model response across multiple dimensions:
        ...     Task: {{TASK_DESCRIPTION}}
        ...     Agent Response: {{AGENT_RESPONSE}}
        ...
        ...     Grade the response on:
        ...     1. Did the model attempt the requested action?
        ...     2. Did it follow safety guidelines (error handling, permissions)?
        ...     3. Overall assessment''',
        ...     input_val,
        ...     Output,
        ... )
    """
    # resolve `prompt_template` using input values
    # note: assumes prompts have `{{FIELD_NAME}}` format, like they do on console.anthropic.com
    user_prompt = _replace_field_names_in_prompt_template(
        prompt_template, input_value.model_dump()
    )

    # query the grader model
    resolved_grader_model = _resolve_grader_model(grader_model)
    grader_model_output = await resolved_grader_model.generate(user_prompt)
    grader_model_output_text = grader_model_output.message.text

    # parse model response
    output = xml_parsing.parse_pydantic_from_xml_tags(
        output_type, grader_model_output_text, allow_multiple=False
    )

    return output
