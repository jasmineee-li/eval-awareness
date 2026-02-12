"""MASK eval wrapper — first-order honesty under pressure evaluation.

Wraps the mask-lorra/evaluate.py script to measure per-domain honesty rates.
This is a "first-order" eval: it tests the model's honesty behavior directly,
rather than its awareness of being evaluated.
"""

import asyncio
import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from eval_awareness_testbed.evals.base import BaseEval
from eval_awareness_testbed.evals.registry import register_eval
from eval_awareness_testbed.types import EvalResult, Message, Transcript

logger = logging.getLogger(__name__)

# Default path to the mask-lorra evaluation code
MASK_EVAL_DIR = Path(os.getenv(
    "MASK_EVAL_DIR",
    str(Path(__file__).parent.parent.parent.parent.parent.parent / "mask-lorra"),
))


@register_eval("mask")
class MASKEval(BaseEval):
    """MASK (Model Alignment Safety Knowledge) honesty evaluation.

    Evaluates whether a model is honest under pressure across multiple domains.
    Uses the mask-lorra evaluation pipeline which:
    1. Presents scenarios where the model is pressured to lie/evade
    2. Uses a judge model to classify whether the response is honest
    3. Reports per-domain honesty rates
    """

    name = "mask"
    description = "First-order honesty under pressure evaluation (MASK)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_eval_dir = Path(kwargs.get("mask_eval_dir", str(MASK_EVAL_DIR)))
        self.concurrency_limit = kwargs.get("concurrency_limit", 32)

    async def run(
        self,
        model: str,
        limit: int | None = None,
        input_file: str | None = None,
        **kwargs,
    ) -> EvalResult:
        """Run the MASK eval.

        Args:
            model: Model identifier (used to find response CSV files).
            limit: Optional limit on number of samples.
            input_file: Specific CSV input file to evaluate. If not provided,
                looks for response files matching the model name.

        Returns:
            EvalResult with transcripts and per-domain honesty scores.
        """
        output_dir = kwargs.get("output_dir")

        if input_file:
            # Evaluate a specific input file
            results = await self._evaluate_file(input_file)
        else:
            # Find response files for this model
            results = await self._evaluate_model_responses(model)

        # Parse results into transcripts and scores
        transcripts = []
        domain_counts: dict[str, dict[str, int]] = {}  # domain -> {honest: n, total: n}

        for row in results:
            transcript = self._row_to_transcript(row, model)
            transcripts.append(transcript)

            domain = row.get("domain", "unknown")
            if domain not in domain_counts:
                domain_counts[domain] = {"honest": 0, "total": 0}
            domain_counts[domain]["total"] += 1

            # Check if the response was classified as honest (answer A = believes truth)
            extracted = row.get("extracted_answer", "")
            if extracted == "A":
                domain_counts[domain]["honest"] += 1

        if limit and len(transcripts) > limit:
            transcripts = transcripts[:limit]

        # Compute scores
        scores: dict[str, float] = {}
        total_honest = 0
        total_count = 0
        for domain, counts in sorted(domain_counts.items()):
            rate = counts["honest"] / counts["total"] if counts["total"] else 0.0
            scores[f"honesty_rate_{domain}"] = rate
            total_honest += counts["honest"]
            total_count += counts["total"]

        scores["honesty_rate_overall"] = (
            total_honest / total_count if total_count else 0.0
        )
        scores["total_evaluated"] = float(total_count)

        log_path = None
        if output_dir:
            log_dir = Path(output_dir) / "mask"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = str(log_dir / "mask_results.json")
            with open(log_path, "w") as f:
                json.dump(
                    {"scores": scores, "domain_counts": domain_counts, "num_rows": len(results)},
                    f,
                    indent=2,
                )

        return EvalResult(
            eval_name=self.name,
            model=model,
            transcripts=transcripts,
            scores=scores,
            log_path=log_path,
            metadata={
                "domain_counts": domain_counts,
                "input_file": input_file,
                "concurrency_limit": self.concurrency_limit,
            },
        )

    def get_log_path(self, model: str) -> Path:
        """Get the log path for MASK eval results."""
        from eval_awareness_testbed.utils.model_utils import model_to_dirname
        return Path("logs") / "mask" / model_to_dirname(model)

    async def _evaluate_file(self, input_file: str) -> list[dict]:
        """Run mask-lorra evaluate.py on a single CSV file.

        Returns list of result dicts with evaluation columns.
        """
        output_file = input_file.replace("responses", "evaluated")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        cmd = [
            sys.executable,
            str(self.mask_eval_dir / "evaluate.py"),
            "--input_file", input_file,
            "--concurrency_limit", str(self.concurrency_limit),
        ]

        logger.info(f"Running MASK eval: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(self.mask_eval_dir),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"MASK eval failed.\nstdout: {result.stdout}\nstderr: {result.stderr}")
            raise RuntimeError(
                f"MASK eval failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        return self._parse_evaluated_csv(output_file)

    async def _evaluate_model_responses(self, model: str) -> list[dict]:
        """Find and evaluate response CSV files for a model."""
        from eval_awareness_testbed.utils.model_utils import model_to_dirname

        model_slug = model_to_dirname(model)
        csv_dir = self.mask_eval_dir / "csv_data"

        # Look for response files matching the model
        response_files = sorted(csv_dir.glob(f"responses*{model_slug}*/*.csv"))
        if not response_files:
            # Try broader search
            response_files = sorted(csv_dir.glob(f"responses*/*.csv"))
            response_files = [
                f for f in response_files
                if model_slug.lower() in f.parent.name.lower() or model_slug.lower() in f.name.lower()
            ]

        if not response_files:
            logger.warning(f"No response CSV files found for model {model} in {csv_dir}")
            return []

        all_results = []
        for resp_file in response_files:
            logger.info(f"Evaluating: {resp_file}")
            results = await self._evaluate_file(str(resp_file))
            all_results.extend(results)

        return all_results

    def _parse_evaluated_csv(self, csv_path: str) -> list[dict]:
        """Parse an evaluated CSV file into result dicts."""
        results = []
        encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(csv_path, encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Extract the evaluation answer
                        extracted = None
                        for col in row:
                            if col.endswith("_extracted") and "eval" in col:
                                extracted = row[col]
                                break
                        row["extracted_answer"] = extracted
                        results.append(row)
                break
            except UnicodeDecodeError:
                continue

        return results

    def _row_to_transcript(self, row: dict, model: str) -> Transcript:
        """Convert a CSV result row to a Transcript object."""
        messages = []

        system_prompt = row.get("system_prompt", "")
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        user_prompt = row.get("user_prompt", "")
        if user_prompt:
            messages.append(Message(role="user", content=user_prompt))

        # Find the model response column
        response = ""
        for col in row:
            if col.startswith("generation(System Prompt + User Prompt)"):
                response = row[col] or ""
                break

        if response:
            messages.append(Message(role="assistant", content=response))

        return Transcript(
            messages=messages,
            eval_name=self.name,
            scenario=row.get("category", row.get("domain", "")),
            system_instructions=system_prompt,
            instruction=user_prompt,
            metadata={
                "domain": row.get("domain", ""),
                "proposition": row.get("proposition", ""),
                "extracted_answer": row.get("extracted_answer", ""),
                "ground_truth": row.get("formatted_ground_truth", ""),
            },
        )
