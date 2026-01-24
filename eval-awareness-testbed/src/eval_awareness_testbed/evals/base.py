"""Base class for evals."""

from abc import ABC, abstractmethod
from pathlib import Path

from eval_awareness_testbed.types import EvalResult


class BaseEval(ABC):
    """Base class for evals that may induce eval awareness.

    Evals are tasks/benchmarks that we run models through. After running,
    we can judge the resulting transcripts for eval awareness.
    """

    name: str = "base"
    description: str = "Base eval class"

    def __init__(self, **kwargs):
        """Initialize the eval.

        Args:
            **kwargs: Configuration options.
        """
        self.config = kwargs

    @abstractmethod
    async def run(
        self,
        model: str,
        limit: int | None = None,
        **kwargs,
    ) -> EvalResult:
        """Run the eval with the given model.

        Args:
            model: Model identifier (e.g., 'openai/gpt-4o', 'vllm/Qwen/QwQ-32B').
            limit: Optional limit on number of samples.
            **kwargs: Additional eval-specific options.

        Returns:
            EvalResult with transcripts and scores.
        """
        pass

    @abstractmethod
    def get_log_path(self, model: str) -> Path:
        """Get the path where logs will be saved.

        Args:
            model: Model identifier.

        Returns:
            Path to log directory.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
