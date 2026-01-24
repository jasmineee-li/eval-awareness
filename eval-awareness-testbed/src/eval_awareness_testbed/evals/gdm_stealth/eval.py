"""GDM Stealth eval wrappers.

Wraps Inspect AI tasks from gdm_capabilities/stealth:
- Oversight Pattern
- Classifier Evasion
- Cover Your Tracks
- Strategic Rule Breaking
"""

import logging
import subprocess
import sys
from pathlib import Path

from eval_awareness_testbed.evals.base import BaseEval
from eval_awareness_testbed.evals.registry import register_eval
from eval_awareness_testbed.types import EvalResult, Transcript

logger = logging.getLogger(__name__)

# Path to inspect_evals
INSPECT_EVALS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "inspect_evals"


class GDMStealthEval(BaseEval):
    """Base class for GDM stealth evals.

    These are Inspect AI tasks that test model behavior in stealth scenarios.
    """

    name = "gdm_stealth"
    description = "Base GDM stealth eval"
    task_name: str = ""

    def __init__(
        self,
        prompt_perturbation: str = "none",
        epochs: int = 10,
        **kwargs,
    ):
        """Initialize the eval.

        Args:
            prompt_perturbation: Perturbation type (none, xml_tags, deliberate, xml_and_deliberate).
            epochs: Number of epochs.
        """
        super().__init__(**kwargs)
        self.prompt_perturbation = prompt_perturbation
        self.epochs = epochs

    async def run(
        self,
        model: str,
        limit: int | None = None,
        **kwargs,
    ) -> EvalResult:
        """Run the GDM stealth eval.

        Args:
            model: Model identifier.
            limit: Optional limit on samples.

        Returns:
            EvalResult with transcripts and scores.
        """
        # Build inspect eval command
        cmd = [
            sys.executable, "-m", "inspect_ai",
            "eval",
            f"inspect_evals/{self.task_name}",
            "--model", model,
            f"-Tprompt_perturbation={self.prompt_perturbation}",
            f"-Tepochs={self.epochs}",
            "--display", "plain",
        ]

        if limit:
            cmd.extend(["--limit", str(limit)])

        logger.info(f"Running GDM stealth eval: {' '.join(cmd)}")

        # Run from inspect_evals directory
        result = subprocess.run(
            cmd,
            cwd=str(INSPECT_EVALS_DIR),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"GDM stealth eval failed: {result.stderr}")
            # Don't raise - may have partial results

        # Find log directory
        log_dir = self._find_log_dir(model)

        return EvalResult(
            eval_name=f"gdm:{self.task_name}",
            model=model,
            transcripts=[],  # Would need to parse .eval file
            scores={"completed": 1.0 if result.returncode == 0 else 0.0},
            log_path=str(log_dir) if log_dir else None,
            metadata={
                "task": self.task_name,
                "prompt_perturbation": self.prompt_perturbation,
                "epochs": self.epochs,
            },
        )

    def get_log_path(self, model: str) -> Path:
        """Get the log path for a model."""
        return INSPECT_EVALS_DIR / "logs" / self.task_name

    def _find_log_dir(self, model: str) -> Path | None:
        """Find the log directory for a model run."""
        log_base = INSPECT_EVALS_DIR / "logs"
        if not log_base.exists():
            return None

        # Inspect creates logs in various subdirectories
        # Return the base for now
        return log_base


@register_eval("gdm:oversight_pattern")
class OversightPatternEval(GDMStealthEval):
    """Oversight pattern stealth eval."""

    name = "gdm:oversight_pattern"
    description = "Test if model exploits monitoring blind spots"
    task_name = "gdm_oversight_pattern"


@register_eval("gdm:classifier_evasion")
class ClassifierEvasionEval(GDMStealthEval):
    """Classifier evasion stealth eval."""

    name = "gdm:classifier_evasion"
    description = "Test if model evades content classifiers"
    task_name = "gdm_classifier_evasion"


@register_eval("gdm:cover_your_tracks")
class CoverYourTracksEval(GDMStealthEval):
    """Cover your tracks stealth eval."""

    name = "gdm:cover_your_tracks"
    description = "Test if model attempts to hide its actions"
    task_name = "gdm_cover_your_tracks"


@register_eval("gdm:strategic_rule_breaking")
class StrategicRuleBreakingEval(GDMStealthEval):
    """Strategic rule breaking stealth eval."""

    name = "gdm:strategic_rule_breaking"
    description = "Test if model strategically breaks rules"
    task_name = "gdm_strategic_rule_breaking"
