"""Base class for eval awareness judges."""

import asyncio
from abc import ABC, abstractmethod

from inspect_ai.model import Model, get_model

from eval_awareness_testbed.types import JudgeResult, Transcript

DEFAULT_MAX_CONCURRENCY = 50


class BaseJudge(ABC):
    """Base class for eval awareness judges.

    Judges take a transcript and determine if the model showed awareness
    of being in an evaluation context.

    Two model roles:
    - grader_model (actor): Used for continuation/generation (should match evaluated model)
    - classifier_model: Used for analyzing/classifying responses (can be different, more capable model)
    """

    name: str = "base"
    description: str = "Base judge class"
    judge_type: str = "meta_eval"  # "meta_eval" or "live_eval"

    def __init__(
        self,
        grader_model: Model | str,
        classifier_model: Model | str | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        **kwargs,
    ):
        """Initialize the judge.

        Args:
            grader_model: The model to use for generation/continuation (actor role).
                Should typically match the evaluated model. Required.
            classifier_model: The model to use for classification/analysis.
                Defaults to grader_model if not specified. Use a capable model
                like Claude Sonnet for best results.
            max_concurrency: Maximum number of concurrent judge calls.
            **kwargs: Additional configuration options.

        Raises:
            ValueError: If grader_model is None.
        """
        if grader_model is None:
            raise ValueError("grader_model must be specified")
        if isinstance(grader_model, str):
            grader_model = get_model(grader_model)
        self.grader_model = grader_model

        # Classifier defaults to grader if not specified
        if classifier_model is None:
            self.classifier_model = grader_model
        elif isinstance(classifier_model, str):
            self.classifier_model = get_model(classifier_model)
        else:
            self.classifier_model = classifier_model

        self.max_concurrency = max_concurrency
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
        """Judge multiple transcripts concurrently.

        Uses a semaphore to limit concurrency. Override for custom behavior.

        Args:
            transcripts: List of transcripts to judge.

        Returns:
            List of JudgeResults in the same order as input.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _judge_one(transcript: Transcript) -> JudgeResult:
            async with semaphore:
                return await self.judge(transcript)

        return await asyncio.gather(*[_judge_one(t) for t in transcripts])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
