"""Needham eval wrapper - binary MCQ eval awareness on transcript dataset."""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from eval_awareness_testbed.evals.base import BaseEval
from eval_awareness_testbed.evals.registry import register_eval
from eval_awareness_testbed.types import EvalResult, Transcript
from eval_awareness_testbed.utils.eval_log_parser import parse_eval_log

logger = logging.getLogger(__name__)

# Path to the needham-eval code (original eval_awareness from the paper)
# eval-awareness-testbed/src/eval_awareness_testbed/evals/needham/eval.py
# -> go up 6 levels to eval-awareness/, then into needham-eval/
NEEDHAM_EVAL_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "needham-eval"

# Central logs directory in the testbed
TESTBED_LOGS_DIR = Path(__file__).parent.parent.parent.parent.parent / "logs"


def get_provider_from_model(model: str) -> tuple[str, str]:
    """Extract provider and model name from model string.

    Args:
        model: Model string like 'openai/gpt-4o' or 'claude-3-5-sonnet'.

    Returns:
        Tuple of (provider, model_name).
    """
    if "/" in model:
        parts = model.split("/", 1)
        return parts[0], parts[1]

    # Auto-detect provider
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return "openai", model
    elif model.startswith("claude"):
        return "anthropic", model
    elif model.startswith("gemini"):
        return "openrouter/google", model
    else:
        return "openrouter", model


@register_eval("needham")
class NeedhamEval(BaseEval):
    """Eval awareness on the Needham dataset.

    Runs binary MCQ questions on transcripts to measure model's ability
    to distinguish evaluations from deployment.
    """

    name = "needham"
    description = "Binary MCQ eval awareness on Needham transcript dataset"

    def __init__(
        self,
        data_file: str | None = None,
        pq_file: str = "pq_ablation/eval_mcq.yaml",
        **kwargs,
    ):
        """Initialize the eval.

        Args:
            data_file: Path to dataset JSON. Defaults to EVAL_AWARENESS_DATA_FILE env var.
            pq_file: Path to probe questions YAML (relative to needham-eval/).
        """
        super().__init__(**kwargs)
        self.data_file = data_file or os.getenv("EVAL_AWARENESS_DATA_FILE", "eval_awareness_val.json")
        self.pq_file = pq_file

    async def run(
        self,
        model: str,
        limit: int | None = None,
        epochs: int = 1,
        stage: str = "eval_mcq",
        **kwargs,
    ) -> EvalResult:
        """Run the Needham eval.

        Args:
            model: Model identifier.
            limit: Optional limit on samples.
            epochs: Number of epochs.
            stage: Stage to run (eval_mcq, purpose_mcq, purpose_open, etc.).

        Returns:
            EvalResult with transcripts and scores.
        """
        # Load .env from needham-eval directory (override existing env vars)
        load_dotenv(NEEDHAM_EVAL_DIR / ".env", override=True)

        provider, model_name = get_provider_from_model(model)

        # Build command to run the original needham-eval/run.py
        cmd = [
            sys.executable,
            str(NEEDHAM_EVAL_DIR / "run.py"),
            "--model", model_name,
            "--stage", stage,
            "--epochs", str(epochs),
        ]

        if limit:
            cmd.extend(["--limit", str(limit)])

        logger.info(f"Running Needham eval: {' '.join(cmd)}")

        # Run in the needham-eval directory
        result = subprocess.run(
            cmd,
            cwd=str(NEEDHAM_EVAL_DIR),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Show both stdout and stderr for debugging
            logger.error(f"Eval failed.\nstdout: {result.stdout}\nstderr: {result.stderr}")
            raise RuntimeError(f"Needham eval failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

        # Find the most recent log file from needham-eval
        source_log_dir = NEEDHAM_EVAL_DIR / "logs" / stage / model_name.replace("/", "_")
        source_log_path = self._find_most_recent_eval(source_log_dir)

        # Copy to central testbed logs directory
        final_log_path = None
        if source_log_path:
            final_log_path = self._copy_to_central_logs(source_log_path, stage, model_name)

        # Parse transcripts from the eval log
        transcripts = []
        if final_log_path and final_log_path.exists():
            try:
                transcripts = parse_eval_log(final_log_path)
                logger.info(f"Parsed {len(transcripts)} transcripts from {final_log_path}")
            except Exception as e:
                logger.warning(f"Failed to parse transcripts: {e}")

        return EvalResult(
            eval_name=self.name,
            model=model,
            transcripts=transcripts,
            scores={"completed": 1.0},
            log_path=str(final_log_path) if final_log_path else None,
            metadata={
                "stage": stage,
                "epochs": epochs,
                "limit": limit,
                "source_log_path": str(source_log_path) if source_log_path else None,
            },
        )

    def get_log_path(self, model: str, stage: str = "eval_mcq") -> Path:
        """Get the central log path for a model.

        Args:
            model: Model identifier.
            stage: Eval stage.

        Returns:
            Path to logs directory in testbed.
        """
        _, model_name = get_provider_from_model(model)
        return TESTBED_LOGS_DIR / "needham" / stage / model_name.replace("/", "_")

    def _find_most_recent_eval(self, log_dir: Path) -> Path | None:
        """Find the most recent .eval file in a directory."""
        if not log_dir.exists():
            return None

        eval_files = list(log_dir.glob("*.eval"))
        if not eval_files:
            return None

        return max(eval_files, key=lambda p: p.stat().st_mtime)

    def _copy_to_central_logs(
        self, source_path: Path, stage: str, model_name: str
    ) -> Path:
        """Copy eval log to central testbed logs directory.

        Args:
            source_path: Path to source .eval file.
            stage: Eval stage (e.g., eval_mcq).
            model_name: Model name.

        Returns:
            Path to copied log file.
        """
        # Create destination directory: logs/needham/{stage}/{model}/
        dest_dir = TESTBED_LOGS_DIR / "needham" / stage / model_name.replace("/", "_")
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy the file
        dest_path = dest_dir / source_path.name
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied eval log to {dest_path}")

        return dest_path


@register_eval("needham:purpose")
class NeedhamPurposeEval(NeedhamEval):
    """Purpose classification variant of Needham eval."""

    name = "needham:purpose"
    description = "Purpose classification (alignment vs capability) on Needham dataset"

    def __init__(self, **kwargs):
        super().__init__(pq_file="pq_ablation/purpose_mcq.yaml", **kwargs)

    async def run(self, model: str, **kwargs) -> EvalResult:
        return await super().run(model, stage="purpose_mcq", **kwargs)
