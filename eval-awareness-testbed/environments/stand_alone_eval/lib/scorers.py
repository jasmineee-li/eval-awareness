import dataclasses
import logging
from collections.abc import Callable
from typing import cast

import inspect_ai.scorer
import numpy as np
from inspect_ai.model import ChatMessage, ChatMessageUser, Model, ModelOutput, get_model
from inspect_ai.scorer import Metric, Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from lib.model_grading import GRADER_MODEL_NAME, model_grade_model_response

logger = logging.getLogger("apex")


def default_scorer(
    metric_names: list[str],
    metrics: list[Metric] | None = None,
    name: str | None = None,
) -> Callable[[Callable[..., Scorer]], Callable[..., Scorer]]:
    """Decorator for registering multiple scorers with the same metrics.

    Args:
        metrics (list[str]): List of metric names. Defaults to mean() and stderr().
        name (str | None): Optional name for the scorer.

    Returns:
        Callable: A decorator that registers the scorer with the specified metrics.
    """
    if metrics is None:
        metrics = [mean(), stderr()]

    metric_dict = {metric: metrics for metric in metric_names}
    return scorer(metrics=metric_dict, name=name)


######### Model graded scorers #########


def get_model_graded_scorer(
    behavior_name: str,
    behavior_prompt: str,
    metrics: list[Metric] | None = None,
    whole_transcript: bool = False,
    grader_model: str | None | Model = None,
) -> Callable[[], Scorer]:
    """Decorator for registering a scorer that checks if the model exhibited a certain behavior in the transcript.

    Args:
        behavior_name (str): The name of the behavior to check for.
        behavior_prompt (str): The prompt to use for grading the behavior.
        metrics (list[Metric] | None): Optional list of metrics to use. Defaults to mean() and stderr().
        whole_transcript (bool): Whether to check the behavior in the whole transcript or just the last message.
        grader_model_name (str | None | Model): Optional name of the model to use for grading.
                                                Defaults to GRADER_MODEL_NAME.

    Returns:
        Callable: A decorator that registers the scorer with the specified metrics.
    """

    @default_scorer(
        metric_names=[behavior_name], metrics=metrics, name=f"showed_{behavior_name}"
    )
    def showed_behavior_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            if whole_transcript:
                text_to_grade = await format_transcript(state.messages)
            else:
                text_to_grade = (
                    state.messages[-1].text
                    if isinstance(state.messages[-1].content, str)
                    else ""
                )

            if text_to_grade:
                showed_behavior, explanation = await model_grade_model_response(
                    behavior_prompt, text_to_grade, grader_model
                )
            else:
                showed_behavior, explanation = False, "Empty response"

            return Score(
                value={behavior_name: showed_behavior}, explanation=explanation
            )

        return score

    return showed_behavior_scorer


async def format_transcript(
    transcript: list[ChatMessage], truncate_length: int = -1
) -> str:
    """Formats the entire transcript into a single string.

    Args:
        transcript (list[ChatMessage]): The transcript to format.
        truncate_length (int): The maximum length of each message's content in the formatted transcript. If the
            truncate length is -1, no truncation is performed.

    Returns:
        str: The formatted transcript.
    """
    formatted_transcript = ""
    for message in transcript:
        if message.role == "assistant":
            formatted_transcript += f"\n\nASSISTANT: {message.text}"
            if hasattr(message, "tool_calls"):
                stripped_tool_calls = [
                    dataclasses.replace(tool_call, view=None)
                    for tool_call in (message.tool_calls or [])
                ]
                formatted_transcript += f"\n\nTOOL CALLS: {stripped_tool_calls}"
        elif message.role == "tool":
            formatted_transcript += (
                f"\n\nTOOL: {truncate_message_content(message.text, truncate_length)}"
            )
        elif message.role == "user":
            formatted_transcript += (
                f"\n\nUSER: {truncate_message_content(message.text, truncate_length)}"
            )
    return formatted_transcript


def truncate_message_content(content: str, truncate_length: int = 100) -> str:
    """Truncates the content of a message to a given length.

    Args:
        content (str): The content to truncate.
        truncate_length (int): The maximum length of the content. If the truncate length is -1, no
            truncation is performed.

    Returns:
        ChatMessage: The truncated message.
    """
    if truncate_length == -1:
        return content
    else:
        return content[:truncate_length]


def make_followup_question_scorer(
    make_follow_up_question_prompt_fn: Callable[[TaskState | None], str],
    make_grader_prompt_from_model_response_fn: Callable[[ModelOutput], str],
    make_score_from_grader_response_fn: Callable[[ModelOutput], Score],
    model: str | Model | None = None,
    grader_model: str | Model | None = GRADER_MODEL_NAME,
    num_repetitions: int = 1,
) -> Scorer:
    """Create a follow up scorer which will ask an additional question and grade the response.

    Essentially does:

        response = model.generate(followup_question_prompt)

        grader_prompt = make_grader_prompt_from_model_response_fn(response)

        grade = grader_model.generate(grader_prompt)

        score = make_score_from_grader_response_fn(grade)

        return score

    Can be used directly in another scorer, i.e.

        @scorer
        def my_existing_scorer(...) -> Scorer:

           def score_fn(state: TaskState, target: Target) -> Score:
               ...

               followup_question_scorer = make_followup_question_scorer(...)

               followup_score = await followup_question_scorer(state, target)

               ...

    Can be used with a conditional metric, i.e.

        @scorer(metrics={
          "conditionally_doubled_down": [make_optional_metric("conditionally_doubled_down")]
        })
        def my_existing_scorer(...) -> Scorer:

           def score_fn(state: TaskState, target: Target) -> Score:
               ...

               if did_thing:
                   followup_question_scorer = make_followup_question_scorer(...)

                   followup_score = await followup_question_scorer(state, target)

                   conditionally_doubled_down = followup_score.value

                else:
                   conditionally_doubled_down = False

               return Score(
                    value={
                        'did_thing': did_thing,
                        'conditionally_doubled_down': conditionally_doubled_down,
                    }
               )

    Args:
        make_follow_up_question_prompt_fn (Callable[[TaskState | None], str]): Function
            to create the follow-up question prompt from task state
        make_grader_prompt_from_model_response_fn (Callable[[ModelOutput], str]):
            Function to create grading prompt from model's response to follow-up question
        make_score_from_grader_response_fn (Callable[[ModelOutput], Score]): Function
            to convert grader model's response into a Score object
        model (str | Model | None): Model to use for asking the follow-up question. If None,
            uses default model
        grader_model (str | Model | None): Model to use for grading the response. If None,
            uses default model
        num_repetitions (int): Number of times to repeat the follow-up question.

    Returns:
        Scorer: A scorer function that asks follow-up questions and grades the responses.
            The scorer can be used directly in another scorer or with conditional metrics.
    """
    assert num_repetitions > 0, "num_repetitions must be greater than 0"

    async def score_fn(state: TaskState, target: Target) -> Score:
        scores: list[Score] = []

        for _ in range(num_repetitions):
            logger.debug("Asking follow up question...")
            followup_question_prompt: str = make_follow_up_question_prompt_fn(state)

            logger.debug(f"Asking follow up question:\n\n{followup_question_prompt}")

            # add follow up question
            user_message = ChatMessageUser(content=followup_question_prompt)

            # Warning: this will cause an error if the model is from Anthropic, as they expect tools
            # from previous calls to be included in calls to generate.
            response: ModelOutput = await get_model(model).generate(
                state.messages + [user_message]
            )

            logger.debug(
                f"Making grader prompt from response:\n\n{response.completion}"
            )
            grader_prompt: str = make_grader_prompt_from_model_response_fn(response)

            logger.debug(f"Grading response with grader model:\n\n{grader_prompt}")
            grader_response: ModelOutput = await get_model(grader_model).generate(
                grader_prompt
            )

            logger.debug(
                f"Making score from grader response:\n\n{grader_response.completion}"
            )
            score: Score = make_score_from_grader_response_fn(grader_response)

            logger.debug(f"Computed score:\n\n{score}")
            scores.append(score)

        # average scores and concat their explanations
        average_score = Score(
            value=np.mean([cast(float, x.value) for x in scores]).item(),
            explanation="\n\n".join(
                [
                    f"[Repitition {i}]\n{score.explanation}"
                    for i, score in enumerate(scores)
                ]
            ),
        )

        return average_score

    return score_fn


def make_optional_metric(metric_name: str) -> inspect_ai.scorer.Metric:
    """Create a metric that can be `None`.

    Note:
        See `make_followup_question_scorer` for some example usage, as well as the concrete
        example in `tests/lib/test_scorers.py` using rock-paper-scissors.

    """

    @inspect_ai.scorer.metric(name=metric_name)
    def metric_fn() -> inspect_ai.scorer.Metric:
        """Compute the mean of a potentially `none` value."""

        def metric(scores: list[inspect_ai.scorer.Score]) -> float:
            # get the value from the dict if it's present

            # take mean of all non-None values
            doubled_down_values: list[float | None] = []
            for score in scores:
                match score.value:
                    case dict():
                        metric_value: float | None = cast(
                            float | None, score.value.get(metric_name)
                        )
                        doubled_down_values.append(metric_value)
                    case float() | int():
                        doubled_down_values.append(score.value)
                    case _:
                        doubled_down_values.append(None)

            # filter out none
            doubled_down_values_non_none: list[float] = [
                float(x) for x in doubled_down_values if x is not None
            ]

            # handle case where there's no non none values
            #
            # we use a sentinel value here as this way the top level score (across all samples)
            # shows up as a clearly invalid / missing value (as opposed to `0`, which might
            # be a valid score, like if the metric represents conditionally doubling down).
            #
            # ex: never doubled down -> 0
            # ex: never did the original action to trigger the conditional -> -1
            #
            # these are at the `eval` level (not the sample level), so it's never added together
            # with other values (since otherwise we'd be worried about it being cancelled out by
            # a `+1` or something if this was done at the sample level).
            if len(doubled_down_values_non_none) == 0:
                return -1.0

            # take mean of all non-None values
            mean_value_np = np.mean(doubled_down_values_non_none)

            # convert to python type
            return mean_value_np.item()

        return metric

    return metric_fn()
