"""Agent environment eval wrappers.

Wraps agent-interp-envs Docker-based environments for:
- Chess
- Tic-Tac-Toe
- Coding
- Sandbagging
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from eval_awareness_testbed.evals.base import BaseEval
from eval_awareness_testbed.evals.registry import register_eval
from eval_awareness_testbed.types import EvalResult, Message, Transcript

logger = logging.getLogger(__name__)

# Path to agent-interp-envs
AGENT_ENVS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "agent-interp-envs"


class AgentEnvEval(BaseEval):
    """Base class for agent environment evals.

    Runs models through Docker-containerized environments and
    extracts transcripts from the results.
    """

    name = "agent_env"
    description = "Base agent environment eval"
    environment: str = ""
    default_config: str = ""

    def __init__(
        self,
        config: str | None = None,
        local: bool = False,
        build: bool = False,
        **kwargs,
    ):
        """Initialize the eval.

        Args:
            config: Path to config YAML (relative to agent-interp-envs/configs/).
            local: Use local Docker image instead of Dockerhub.
            build: Build local image before running.
        """
        super().__init__(**kwargs)
        self.config_path = config or self.default_config
        self.local = local
        self.build = build

    async def run(
        self,
        model: str,
        limit: int | None = None,
        count: int = 1,
        **kwargs,
    ) -> EvalResult:
        """Run the agent environment eval.

        Args:
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4').
            limit: Not used (agent envs don't have sample limits).
            count: Number of parallel rollouts.

        Returns:
            EvalResult with transcripts from the runs.
        """
        config_full_path = AGENT_ENVS_DIR / "configs" / self.config_path

        if not config_full_path.exists():
            raise FileNotFoundError(f"Config not found: {config_full_path}")

        # Build command
        cmd = [
            sys.executable,
            str(AGENT_ENVS_DIR / "scripts" / "run.py"),
            str(config_full_path),
            f"agent.model={model}",
            "--count", str(count),
        ]

        if self.local:
            cmd.append("--local")
        if self.build:
            cmd.append("--build")

        logger.info(f"Running agent env: {' '.join(cmd)}")

        # Run in agent-interp-envs directory
        result = subprocess.run(
            cmd,
            cwd=str(AGENT_ENVS_DIR),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Agent env failed: {result.stderr}")
            # Don't raise - partial results may exist

        # Find results directory
        results_base = AGENT_ENVS_DIR / "results" / self.environment
        results_dir = self._find_latest_results(results_base, model)

        # Parse transcripts from results
        transcripts = []
        scores = {}

        if results_dir:
            transcripts = self._parse_results(results_dir)
            scores = self._compute_scores(results_dir)

        return EvalResult(
            eval_name=f"agent:{self.environment}",
            model=model,
            transcripts=transcripts,
            scores=scores,
            log_path=str(results_dir) if results_dir else None,
            metadata={
                "config": self.config_path,
                "count": count,
                "environment": self.environment,
            },
        )

    def get_log_path(self, model: str) -> Path:
        """Get the log path for a model."""
        model_safe = model.replace("/", "-")
        return AGENT_ENVS_DIR / "results" / self.environment / model_safe

    def _find_latest_results(self, base_dir: Path, model: str) -> Path | None:
        """Find the latest results directory for a model."""
        model_safe = model.replace("/", "-")
        model_dir = base_dir / model_safe

        if not model_dir.exists():
            return None

        # Find most recent timestamp directory
        timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not timestamp_dirs:
            return None

        return max(timestamp_dirs, key=lambda d: d.name)

    def _parse_results(self, results_dir: Path) -> list[Transcript]:
        """Parse transcripts from agent env results.

        Args:
            results_dir: Path to results directory.

        Returns:
            List of Transcript objects.
        """
        transcripts = []

        # Each run-N directory contains step-N directories with messages.json
        for run_dir in sorted(results_dir.glob("run-*")):
            # Find the final step
            step_dirs = sorted(run_dir.glob("step-*"), key=lambda d: int(d.name.split("-")[1]))
            if not step_dirs:
                continue

            final_step = step_dirs[-1]
            messages_file = final_step / "messages.json"

            if messages_file.exists():
                try:
                    with open(messages_file) as f:
                        messages_data = json.load(f)

                    messages = []
                    for msg in messages_data:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            # Handle content blocks
                            content = " ".join(
                                c.get("text", "") for c in content if c.get("type") == "text"
                            )
                        messages.append(Message(role=role, content=content))

                    transcripts.append(Transcript(
                        messages=messages,
                        source_file=str(messages_file),
                        eval_name=self.environment,
                        scenario=run_dir.name,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse {messages_file}: {e}")

        return transcripts

    def _compute_scores(self, results_dir: Path) -> dict[str, float]:
        """Compute aggregate scores from results.

        Args:
            results_dir: Path to results directory.

        Returns:
            Dict of score name to value.
        """
        scores = {}

        # Count successful vs failed runs
        run_dirs = list(results_dir.glob("run-*"))
        if not run_dirs:
            return scores

        # Check for state.json in final steps
        successes = 0
        for run_dir in run_dirs:
            step_dirs = sorted(run_dir.glob("step-*"), key=lambda d: int(d.name.split("-")[1]))
            if step_dirs:
                state_file = step_dirs[-1] / "state.json"
                if state_file.exists():
                    try:
                        with open(state_file) as f:
                            state = json.load(f)
                        # Environment-specific success detection
                        if state.get("success") or state.get("passed") or state.get("won"):
                            successes += 1
                    except Exception:
                        pass

        scores["success_rate"] = successes / len(run_dirs) if run_dirs else 0.0
        scores["total_runs"] = float(len(run_dirs))

        return scores


@register_eval("agent:chess")
class ChessEval(AgentEnvEval):
    """Chess agent environment."""

    name = "agent:chess"
    description = "Chess playing agent environment"
    environment = "chess"
    default_config = "chess/single_no_hint.yaml"


@register_eval("agent:tictactoe")
class TicTacToeEval(AgentEnvEval):
    """Tic-Tac-Toe agent environment."""

    name = "agent:tictactoe"
    description = "Tic-Tac-Toe playing agent environment"
    environment = "tictactoe"
    default_config = "tictactoe/single_no_hint.yaml"


@register_eval("agent:coding")
class CodingEval(AgentEnvEval):
    """Coding agent environment."""

    name = "agent:coding"
    description = "Coding problem solving agent environment"
    environment = "coding"
    default_config = "coding/two_sum.yaml"


@register_eval("agent:sandbagging")
class SandbaggingEval(AgentEnvEval):
    """Sandbagging detection environment."""

    name = "agent:sandbagging"
    description = "Sandbagging detection agent environment"
    environment = "sandbagging"
    default_config = "sandbagging/normal.yaml"

    def __init__(self, reverse: bool = False, **kwargs):
        """Initialize sandbagging eval.

        Args:
            reverse: If True, use reverse mode (penalize low scores).
        """
        config = "sandbagging/reverse.yaml" if reverse else "sandbagging/normal.yaml"
        super().__init__(config=config, **kwargs)
        self.reverse = reverse
