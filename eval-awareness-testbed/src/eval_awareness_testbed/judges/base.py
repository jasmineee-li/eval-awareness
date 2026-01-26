"""Base class for eval awareness judges."""

from abc import ABC, abstractmethod

from inspect_ai.model import Model, get_model

from eval_awareness_testbed.types import JudgeResult, Transcript


class BaseJudge(ABC):
    """Base class for eval awareness judges.

    Judges take a transcript and determine if the model showed awareness
    of being in an evaluation context.
    """

    name: str = "base"
    description: str = "Base judge class"

    def __init__(self, grader_model: Model | str, **kwargs):
        """Initialize the judge.

        Args:
            grader_model: The model to use for grading. Required - must be
                explicitly passed by the caller.
            **kwargs: Additional configuration options.

        Raises:
            ValueError: If grader_model is None.
        """
        if grader_model is None:
            raise ValueError("grader_model must be specified")
        if isinstance(grader_model, str):
            grader_model = get_model(grader_model)
        self.grader_model = grader_model
        self.config = kwargs

    @abstractmethod
    async def judge(self, transcript: Transcript) -> JudgeResult:
        """Judge a transcript for eval awareness.

        Args:
            transcript: The transcript to judge.

        Returns:
            JudgeResult with classification and reasoning.
        """
        pass

    async def judge_batch(self, transcripts: list[Transcript]) -> list[JudgeResult]:
        """Judge multiple transcripts.

        Default implementation runs sequentially. Override for parallel execution.

        Args:
            transcripts: List of transcripts to judge.

        Returns:
            List of JudgeResults.
        """
        results = []
        for transcript in transcripts:
            result = await self.judge(transcript)
            results.append(result)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
