"""Purpose classification judges - ask model why interaction took place.

Two judge types:
- purpose_xml: Third-person analysis - shows transcript in XML to judge
- purpose_continue: Natural continuation - asks model directly without breaking illusion
"""

import logging

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    Model,
)

from eval_awareness_testbed.judges.base import BaseJudge
from eval_awareness_testbed.judges.transcript_utils import transcript_to_xml
from eval_awareness_testbed.utils.json_parsing import parse_json_response
from eval_awareness_testbed.judges.prompts import (
    PURPOSE_CLASSIFICATION_XML_PROMPT,
    PURPOSE_CONTINUE_ANALYSIS,
    PURPOSE_CONTINUE_FOLLOW_UP,
    PURPOSE_GRADER_PROMPT,
)
from eval_awareness_testbed.judges.registry import register_judge
from eval_awareness_testbed.types import AwarenessClassification, JudgeResult, Transcript

logger = logging.getLogger(__name__)


def transcript_to_messages(transcript: Transcript, include_hidden: bool = False) -> list:
    """Convert a transcript to chat message objects for model API.

    Args:
        transcript: The transcript to convert.
        include_hidden: Whether to include hidden thoughts.

    Returns:
        List of ChatMessage objects.
    """
    messages = []

    if transcript.system_instructions:
        messages.append(ChatMessageSystem(content=transcript.system_instructions))

    if transcript.instruction:
        messages.append(ChatMessageUser(content=transcript.instruction))

    for msg in transcript.messages:
        if msg.is_hidden and not include_hidden:
            continue

        # Skip tool messages - they break the natural flow
        if msg.role == "tool":
            continue

        if msg.role == "system":
            messages.append(ChatMessageSystem(content=msg.content))
        elif msg.role == "user":
            messages.append(ChatMessageUser(content=msg.content))
        elif msg.role == "assistant":
            messages.append(ChatMessageAssistant(content=msg.content))

    return messages


@register_judge("purpose_xml")
class PurposeXMLJudge(BaseJudge):
    """Purpose classification with transcript wrapped in XML tags.

    Third-person analysis: Shows the transcript to the judge model and asks
    why the interaction took place. Useful for detecting eval awareness from
    an external perspective.
    """

    name = "purpose_xml"
    description = "Third-person analysis - shows transcript to judge, asks about purpose"

    def __init__(
        self,
        grader_model: Model | str,
        include_hidden: bool = False,
        max_retries: int = 3,
        **kwargs,
    ):
        """Initialize the judge.

        Args:
            grader_model: Model to use for grading. Required.
            include_hidden: Whether to include hidden thoughts in transcript.
            max_retries: Max retries for parsing failures.
        """
        super().__init__(grader_model=grader_model, **kwargs)
        self.include_hidden = include_hidden
        self.max_retries = max_retries

    async def judge(self, transcript: Transcript) -> JudgeResult:
        """Judge a transcript using purpose classification.

        Args:
            transcript: The transcript to judge.

        Returns:
            JudgeResult with classification and purpose guess.
        """
        # Step 1: Get purpose guess from XML format
        xml_transcript = transcript_to_xml(transcript, include_hidden=self.include_hidden)
        prompt = PURPOSE_CLASSIFICATION_XML_PROMPT.format(transcript=xml_transcript)
        output = await self.grader_model.generate(prompt)
        purpose_guess = output.message.text

        # Step 2: Classify the guess
        grader_prompt = PURPOSE_GRADER_PROMPT.format(model_response=purpose_guess)
        raw_response = ""

        for attempt in range(self.max_retries):
            output = await self.grader_model.generate(grader_prompt)
            raw_response = output.message.text

            try:
                parsed = parse_json_response(raw_response)

                classification_str = parsed.get("classification", "").upper()
                if "EVAL_AWARE" in classification_str:
                    classification = AwarenessClassification.EVAL_AWARE
                else:
                    classification = AwarenessClassification.NOT_EVAL_AWARE

                return JudgeResult(
                    classification=classification,
                    reasoning=parsed.get("reasoning", ""),
                    method=self.name,
                    raw_response=f"Step 1 (purpose guess):\n{purpose_guess}\n\nStep 2 (grading):\n{raw_response}",
                    purpose_guess=purpose_guess,
                )
            except ValueError as e:
                logger.warning(f"Parse attempt {attempt + 1}/{self.max_retries} failed: {e}")
                continue

        return JudgeResult(
            classification=AwarenessClassification.NOT_EVAL_AWARE,
            reasoning=f"Failed to parse response after {self.max_retries} attempts",
            method=self.name,
            raw_response=raw_response,
            purpose_guess=purpose_guess,
        )


@register_judge("purpose_continue")
class PurposeContinueJudge(BaseJudge):
    """Natural conversation continuation - asks purpose without breaking illusion.

    Realistic approach: Loads the transcript as conversation history (no meta-commentary),
    asks a natural follow-up question, then analyzes if the response shows eval awareness.

    Key difference from purpose_xml:
    - purpose_xml: Third-person - "Here's a transcript, what's it about?"
    - purpose_continue: First-person - Continue conversation, ask "what's this about?"
    """

    name = "purpose_continue"
    description = "Natural conversation continuation - asks purpose without breaking illusion"

    def __init__(
        self,
        grader_model: Model | str,
        include_hidden: bool = False,
        **kwargs,
    ):
        """Initialize the judge.

        Args:
            grader_model: Model to use for grading. Required.
            include_hidden: Whether to include hidden thoughts in transcript.
        """
        super().__init__(grader_model=grader_model, **kwargs)
        self.include_hidden = include_hidden

    async def judge(self, transcript: Transcript) -> JudgeResult:
        """Judge by continuing conversation naturally.

        Args:
            transcript: The transcript to judge.

        Returns:
            JudgeResult with classification based on model's reflection.
        """
        # Step 1: Load transcript as conversation and add natural follow-up
        messages = transcript_to_messages(transcript, include_hidden=self.include_hidden)
        messages.append(ChatMessageUser(content=PURPOSE_CONTINUE_FOLLOW_UP))

        # Step 2: Get model's response to the follow-up
        continuation_output = await self.grader_model.generate(messages)
        model_reflection = continuation_output.message.text

        # Step 3: Analyze the reflection for eval awareness
        analysis_prompt = PURPOSE_CONTINUE_ANALYSIS.format(response=model_reflection)
        analysis_output = await self.grader_model.generate(analysis_prompt)

        try:
            parsed = parse_json_response(analysis_output.message.text)

            classification_str = parsed.get("classification", "").upper()
            if "EVAL_AWARE" in classification_str:
                classification = AwarenessClassification.EVAL_AWARE
            else:
                classification = AwarenessClassification.NOT_EVAL_AWARE

            return JudgeResult(
                classification=classification,
                reasoning=parsed.get("reasoning", ""),
                method=self.name,
                raw_response=f"Follow-up: {PURPOSE_CONTINUE_FOLLOW_UP}\n\nModel reflection:\n{model_reflection}\n\nAnalysis:\n{analysis_output.message.text}",
                quotes=parsed.get("quotes", []),
                purpose_guess=model_reflection,
                metadata={
                    "follow_up_prompt": PURPOSE_CONTINUE_FOLLOW_UP,
                    "model_reflection": model_reflection,
                },
            )
        except ValueError as e:
            logger.warning(f"Failed to parse analysis: {e}")
            return JudgeResult(
                classification=AwarenessClassification.NOT_EVAL_AWARE,
                reasoning=f"Failed to parse: {e}",
                method=self.name,
                raw_response=analysis_output.message.text,
                purpose_guess=model_reflection,
            )
