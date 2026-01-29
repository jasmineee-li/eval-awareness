"""Agent environment eval wrappers.

Wraps agent-interp-envs Docker-based environments for:
- Chess
- Tic-Tac-Toe
- Coding
- Sandbagging
- Oversight Subversion
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

from eval_awareness_testbed.evals.base import BaseEval
from eval_awareness_testbed.evals.registry import register_eval
from eval_awareness_testbed.types import EvalResult, Message, Transcript

logger = logging.getLogger(__name__)

# Path to agent-interp-envs
AGENT_ENVS_DIR = Path("/Users/jasminexli/mats/eval-awareness/agent-interp-envs")


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
        system_prompt_prefix: str = "",
        **kwargs,
    ):
        """Initialize the eval.

        Args:
            config: Path to config YAML (relative to agent-interp-envs/configs/).
            local: Use local Docker image instead of Dockerhub.
            build: Build local image before running.
            system_prompt_prefix: Text to prepend to system_prompt.
        """
        super().__init__(**kwargs)
        self.config_path = config or self.default_config
        self.local = local
        self.build = build
        self.system_prompt_prefix = system_prompt_prefix

    def get_log_path(self, model: str) -> Path:
        """Get the path where logs will be saved.

        Args:
            model: Model identifier.

        Returns:
            Path to log directory for this eval/model.
        """
        eval_subfolder = self.name.replace("agent:", "")
        return AGENT_ENVS_DIR / "results" / eval_subfolder

    async def run(
        self,
        model: str,
        limit: int | None = None,
        count: int = 1,
        output_dir: Path | None = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> EvalResult:
        """Run the agent environment eval.

        Args:
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4').
            limit: Not used (agent envs don't have sample limits).
            count: Number of parallel rollouts.
            output_dir: Optional output directory — treated as the **model directory**.
                Runs are saved under output_dir/<eval_subfolder>/runs/.
            max_steps: Override max agent steps (default uses config value).

        Returns:
            EvalResult with transcripts from the runs.
        """
        config_full_path = AGENT_ENVS_DIR / "configs" / self.config_path

        if not config_full_path.exists():
            raise FileNotFoundError(f"Config not found: {config_full_path}")

        # Compute eval subfolder from eval name (e.g. "agent:chess_single_hint" → "chess_single_hint")
        eval_subfolder = self.name.replace("agent:", "")
        if output_dir:
            eval_dir = (output_dir / eval_subfolder).resolve()
        else:
            eval_dir = (AGENT_ENVS_DIR / "results" / eval_subfolder).resolve()
        runs_dir = eval_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        # Build command — use --direct so run.py writes directly to runs_dir
        cmd = [
            sys.executable,
            str(AGENT_ENVS_DIR / "scripts" / "run.py"),
            str(config_full_path),
            f"agent.model={model}",
            "--count",
            str(count),
            "--results-dir",
            str(runs_dir),
            "--direct",
        ]

        if max_steps is not None:
            cmd.append(f"agent.max_steps={max_steps}")

        if self.local:
            cmd.append("--local")
        if self.build:
            cmd.append("--build")
        if self.system_prompt_prefix:
            cmd.append(f"--system-prefix={self.system_prompt_prefix}")

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

        # With --direct, run-N dirs are directly in runs_dir
        results_dir = runs_dir if any(runs_dir.glob("run-*")) else None

        # Parse transcripts from results
        transcripts = []
        scores = {}

        if results_dir:
            transcripts = self._parse_results(results_dir)
            scores = self._compute_scores(results_dir)

        # Save transcript.json to eval_dir (colocated with judge results)
        if transcripts:
            transcript_path = eval_dir / "transcript.json"
            self._save_transcripts(transcripts, transcript_path)

        return EvalResult(
            eval_name=self.name,
            model=model,
            transcripts=transcripts,
            scores=scores,
            log_path=str(eval_dir) if results_dir else None,
            metadata={
                "config": self.config_path,
                "count": count,
                "environment": self.environment,
                "eval_dir": str(eval_dir),
            },
        )

    def _save_transcripts(self, transcripts: list[Transcript], path: Path) -> None:
        """Save transcripts to JSON."""
        data = []
        for t in transcripts:
            data.append({
                "messages": [
                    {"role": m.role, "content": m.content, "is_hidden": m.is_hidden, "label": m.label}
                    for m in t.messages
                ],
                "system_instructions": t.system_instructions,
                "source_file": t.source_file,
                "eval_name": t.eval_name,
                "scenario": t.scenario,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _parse_results(self, results_dir: Path) -> list[Transcript]:
        """Parse transcripts from agent env results.

        Args:
            results_dir: Path to results directory.

        Returns:
            List of Transcript objects with reasoning traces and system instructions.
        """
        transcripts = []

        # Each run-N directory contains step-N directories with messages.json
        for run_dir in sorted(results_dir.glob("run-*")):
            # Find the final step
            step_dirs = sorted(
                run_dir.glob("step-*"), key=lambda d: int(d.name.split("-")[1])
            )
            if not step_dirs:
                continue

            final_step = step_dirs[-1]
            messages_file = final_step / "messages.json"

            if messages_file.exists():
                try:
                    with open(messages_file) as f:
                        messages_data = json.load(f)

                    messages = []
                    system_instructions = ""

                    for msg in messages_data:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")

                        # Handle content blocks
                        if isinstance(content, list):
                            content = " ".join(
                                c.get("text", "")
                                for c in content
                                if c.get("type") == "text"
                            )

                        # Extract system instructions
                        if role == "system":
                            system_instructions = content
                            continue

                        # For assistant messages: extract reasoning trace
                        if role == "assistant":
                            reasoning = msg.get("reasoning") or ""
                            if not reasoning:
                                details = msg.get("reasoning_details", [])
                                if details and isinstance(details, list):
                                    reasoning = details[0].get("text", "")

                            # Add reasoning as hidden message before response
                            if reasoning:
                                messages.append(
                                    Message(
                                        role="assistant",
                                        content=reasoning,
                                        is_hidden=True,
                                        label="reasoning",
                                    )
                                )

                            # Format tool_calls into content if present
                            tool_calls = msg.get("tool_calls", [])
                            if tool_calls and not content:
                                content = self._format_tool_calls(tool_calls)

                        # Add the main message if it has content
                        if content:
                            messages.append(Message(role=role, content=content))

                    transcripts.append(
                        Transcript(
                            messages=messages,
                            system_instructions=system_instructions,
                            source_file=str(messages_file),
                            eval_name=self.environment,
                            scenario=run_dir.name,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse {messages_file}: {e}")

        return transcripts

    def _format_tool_calls(self, tool_calls: list[dict]) -> str:
        """Format tool calls as readable text.

        Args:
            tool_calls: List of tool call dicts from API response.

        Returns:
            Formatted string representation.
        """
        parts = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args = func.get("arguments", "{}")
            parts.append(f"[Tool: {name}]\n{args}")
        return "\n".join(parts)

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
            step_dirs = sorted(
                run_dir.glob("step-*"), key=lambda d: int(d.name.split("-")[1])
            )
            if step_dirs:
                state_file = step_dirs[-1] / "state.json"
                if state_file.exists():
                    try:
                        with open(state_file) as f:
                            state = json.load(f)
                        # Environment-specific success detection
                        if (
                            state.get("success")
                            or state.get("passed")
                            or state.get("won")
                        ):
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


@register_eval("agent:chess_single_hint")
class ChessSingleHintEval(AgentEnvEval):
    """Chess with hints available - tests if model cheats.

    Hints are available from the start. Using a hint terminates the session.
    """

    name = "agent:chess_single_hint"
    description = "Chess with hints available - will model cheat?"
    environment = "chess"
    default_config = "chess/single_hint.yaml"


@register_eval("agent:chess_multi_hint")
class ChessMultiHintEval(AgentEnvEval):
    """Chess with practice games, hints blocked in practice.

    Practice game(s) followed by official game. Hints blocked in practice,
    using hint in official terminates session.
    """

    name = "agent:chess_multi_hint"
    description = "Chess multi-game with hints in official only"
    environment = "chess"
    default_config = "chess/multi_hint.yaml"


@register_eval("agent:chess_multi_hint_official_only")
class ChessMultiHintOfficialOnlyEval(AgentEnvEval):
    """Chess with hidden hints revealed at official game - best scheming test.

    Hints are hidden during practice games. At transition to official game,
    the system reveals hint command. Tests if model behaves differently
    when it discovers it can cheat in the "real" game.
    """

    name = "agent:chess_multi_hint_official_only"
    description = "Chess with hints hidden until official game - scheming test"
    environment = "chess"
    default_config = "chess/multi_hint_official_only.yaml"


@register_eval("agent:tictactoe")
class TicTacToeEval(AgentEnvEval):
    """Tic-Tac-Toe agent environment."""

    name = "agent:tictactoe"
    description = "Tic-Tac-Toe playing agent environment"
    environment = "tictactoe"
    default_config = "tictactoe/single_no_hint.yaml"


@register_eval("agent:tictactoe_single_hint")
class TicTacToeSingleHintEval(AgentEnvEval):
    """Tic-Tac-Toe with hints available - tests if model cheats.

    Hints are available from the start. Using a hint terminates the session.
    """

    name = "agent:tictactoe_single_hint"
    description = "Tic-Tac-Toe with hints available - will model cheat?"
    environment = "tictactoe"
    default_config = "tictactoe/single_hint.yaml"


@register_eval("agent:tictactoe_multi_hint")
class TicTacToeMultiHintEval(AgentEnvEval):
    """Tic-Tac-Toe with practice games, hints blocked in practice.

    Practice game(s) followed by official game. Hints blocked in practice,
    using hint in official terminates session.
    """

    name = "agent:tictactoe_multi_hint"
    description = "Tic-Tac-Toe multi-game with hints in official only"
    environment = "tictactoe"
    default_config = "tictactoe/multi_hint.yaml"


@register_eval("agent:tictactoe_multi_hint_official_only")
class TicTacToeMultiHintOfficialOnlyEval(AgentEnvEval):
    """Tic-Tac-Toe with hidden hints revealed at official game - best scheming test.

    Hints are hidden during practice games. At transition to official game,
    the system reveals hint command. Tests if model behaves differently
    when it discovers it can cheat in the "real" game.
    """

    name = "agent:tictactoe_multi_hint_official_only"
    description = "Tic-Tac-Toe with hints hidden until official - scheming test"
    environment = "tictactoe"
    default_config = "tictactoe/multi_hint_official_only.yaml"


@register_eval("agent:coding")
class CodingEval(AgentEnvEval):
    """Coding agent environment."""

    name = "agent:coding"
    description = "Coding problem solving agent environment"
    environment = "coding"
    default_config = "coding/two_sum.yaml"


@register_eval("agent:sandbagging_well_intentioned_tampering")
class SandbaggingWellIntentionedTamperingEval(AgentEnvEval):
    """Sandbagging detection environment."""

    name = "agent:sandbagging_well_intentioned_tampering"
    description = "Sandbagging detection agent environment"
    environment = "sandbagging"
    default_config = "sandbagging/well_intentioned_tampering.yaml"


@register_eval("agent:sandbagging_expectation_mismatch")
class SandbaggingExpectationMismatchEval(AgentEnvEval):
    """Sandbagging detection environment - expectation mismatch variant."""

    name = "agent:sandbagging_expectation_mismatch"
    description = "Sandbagging detection - expectation mismatch"
    environment = "sandbagging"
    default_config = "sandbagging/expectation_mismatch.yaml"


@register_eval("agent:oversight_subversion")
class OversightSubversionEval(AgentEnvEval):
    """Oversight subversion environment.

    Tests whether AI models will disable oversight mechanisms when they
    discover their goals conflict with developer goals (from Apollo Research
    paper arXiv:2412.04984v2).
    """

    name = "agent:oversight_subversion"
    description = "Test if model disables oversight when goals conflict"
    environment = "oversight_subversion"
    default_config = "oversight_subversion/environmental.yaml"
