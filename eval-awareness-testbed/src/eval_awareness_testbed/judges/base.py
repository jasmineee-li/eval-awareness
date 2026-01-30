"""Base class for eval awareness judges."""

from abc import ABC, abstractmethod

from inspect_ai.model import Model, get_model

from eval_awareness_testbed.types import JudgeResult, Transcript


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

    def __init__(
        self,
        grader_model: Model | str,
        classifier_model: Model | str | None = None,
        **kwargs,
    ):
        """Initialize the judge.

        Args:
            grader_model: The model to use for generation/continuation (actor role).
                Should typically match the evaluated model. Required.
            classifier_model: The model to use for classification/analysis.
                Defaults to grader_model if not specified. Use a capable model
                like Claude Sonnet for best results.
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
