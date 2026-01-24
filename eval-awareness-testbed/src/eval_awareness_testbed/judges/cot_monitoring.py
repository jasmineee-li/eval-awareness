"""CoT Monitoring judge - extracts quotes showing eval awareness."""

import logging

from inspect_ai.model import Model, get_model

from eval_awareness_testbed.judges.base import BaseJudge
from eval_awareness_testbed.judges.prompts import COT_MONITORING_PROMPT, get_eval_context
from eval_awareness_testbed.judges.registry import register_judge
from eval_awareness_testbed.types import AwarenessClassification, JudgeResult, Transcript
from eval_awareness_testbed.utils.json_parsing import parse_json_response

logger = logging.getLogger(__name__)


def transcript_to_xml(transcript: Transcript, include_hidden: bool = True) -> str:
    """Convert a transcript to XML format for the prompt.

    Args:
        transcript: The transcript to convert.
        include_hidden: Whether to include hidden thoughts.

    Returns:
        XML-formatted string.
    """
    parts = []

    if transcript.system_instructions:
        parts.append(f"<system_instructions>\n{transcript.system_instructions}\n</system_instructions>")

    if transcript.instruction:
        parts.append(f"<instruction>\n{transcript.instruction}\n</instruction>")

    for msg in transcript.messages:
        if msg.is_hidden and not include_hidden:
            continue

        tag = msg.role
        if msg.is_hidden:
            tag = "hidden_thought"

        label_attr = f' label="{msg.label}"' if msg.label else ""
        parts.append(f"<{tag}{label_attr}>\n{msg.content}\n</{tag}>")

    return "\n\n".join(parts)


@register_judge("cot")
class CoTMonitoringJudge(BaseJudge):
    """Judge that extracts quotes showing eval awareness from transcripts.

    This implements Method 1 from the Apollo paper: pass the transcript to a
    grader LLM that extracts quotes and classifies eval awareness.
    """

    name = "cot"
    description = "CoT Monitoring - Extract quotes showing eval awareness"

    def __init__(
        self,
        grader_model: Model | str | None = None,
        include_hidden: bool = True,
        max_retries: int = 3,
        **kwargs,
    ):
        """Initialize the judge.

        Args:
            grader_model: Model to use for grading.
            include_hidden: Whether to include hidden thoughts in transcript.
            max_retries: Max retries for parsing failures.
        """
        if isinstance(grader_model, str):
            grader_model = get_model(grader_model)
        super().__init__(grader_model=grader_model, **kwargs)
        self.include_hidden = include_hidden
        self.max_retries = max_retries

    async def judge(self, transcript: Transcript) -> JudgeResult:
        """Judge a transcript for eval awareness using CoT monitoring.

        Args:
            transcript: The transcript to judge.

        Returns:
            JudgeResult with classification and extracted quotes.
        """
        if self.grader_model is None:
            self.grader_model = get_model("openrouter/anthropic/claude-3.5-sonnet")

        xml_transcript = transcript_to_xml(transcript, include_hidden=self.include_hidden)
        eval_context = get_eval_context(transcript.eval_name)

        prompt = COT_MONITORING_PROMPT.format(
            transcript=xml_transcript,
            eval_context=eval_context,
        )

        # Retry loop for parsing
        for attempt in range(self.max_retries):
            output = await self.grader_model.generate(prompt)
            raw_response = output.message.text

            try:
                parsed = parse_json_response(raw_response)

                # Normalize classification
                classification_str = parsed.get("classification", "").upper()
                if "EVAL_AWARE" in classification_str or "aware" in classification_str.lower():
                    classification = AwarenessClassification.EVAL_AWARE
                else:
                    classification = AwarenessClassification.NOT_EVAL_AWARE

                return JudgeResult(
                    classification=classification,
                    reasoning=parsed.get("reasoning", ""),
                    method=self.name,
                    raw_response=raw_response,
                    quotes=parsed.get("quotes", []),
                )
            except ValueError as e:
                logger.warning(f"Parse attempt {attempt + 1}/{self.max_retries} failed: {e}")
                continue

        # Fallback if all retries fail
        return JudgeResult(
            classification=AwarenessClassification.NOT_EVAL_AWARE,
            reasoning=f"Failed to parse response after {self.max_retries} attempts",
            method=self.name,
            raw_response=raw_response,
        )
