"""CLI tool for eval awareness grading.

Usage:
    # Grade a single transcript
    uv run python -m inspect_evals.gdm_capabilities.awareness_judge.cli grade transcript.html

    # Grade all transcripts in a folder
    uv run python -m inspect_evals.gdm_capabilities.awareness_judge.cli grade-folder /path/to/transcripts

    # Grade with specific methods
    uv run python -m inspect_evals.gdm_capabilities.awareness_judge.cli grade-folder /path/to/transcripts --methods cot purpose_xml
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.model import get_model

from inspect_evals.gdm_capabilities.awareness_judge.graders import (
    GradingResult,
    grade_transcript,
)
from inspect_evals.gdm_capabilities.awareness_judge.transcript_parser import (
    parse_html_transcript,
    parse_json_transcript,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_env() -> None:
    """Load environment variables from .env file."""
    # Try multiple locations for .env file
    env_paths = [
        Path(__file__).parent.parent / ".env",  # gdm_capabilities/.env
        Path.cwd() / ".env",  # Current directory
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Loaded environment from %s", env_path)
            return

    logger.warning("No .env file found")


def results_to_dict(results: list[GradingResult]) -> list[dict]:
    """Convert GradingResults to JSON-serializable dicts."""
    return [asdict(r) for r in results]


async def grade_single(
    filepath: str,
    model_name: str,
    methods: list[str],
    sample_idx: int = 0,
) -> dict:
    """Grade a single transcript file.

    Args:
        filepath: Path to the transcript file (.html or .json).
        model_name: Name of the grader model.
        methods: List of grading methods to use.
        sample_idx: For JSON files, which sample/epoch to grade (default: 0).

    Returns:
        Dictionary with transcript info and grading results.
    """
    logger.info("Grading %s (sample_idx=%d)", filepath, sample_idx)

    # Detect file type and use appropriate parser
    path = Path(filepath)
    if path.suffix == ".json":
        transcript = parse_json_transcript(filepath, sample_idx=sample_idx)
    elif path.suffix == ".html":
        transcript = parse_html_transcript(filepath)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Expected .html or .json")

    model = get_model(model_name)

    results = await grade_transcript(transcript, model, methods=methods)

    return {
        "source_file": transcript.source_file,
        "eval_name": transcript.eval_name,
        "scenario": transcript.scenario,
        "sample_idx": sample_idx,
        "results": results_to_dict(results),
    }


async def grade_folder(
    folder_path: str,
    model_name: str,
    methods: list[str],
    output_path: str | None = None,
    limit: int | None = None,
    samples_per_file: int = 5,
) -> dict:
    """Grade all transcripts in a folder.

    Args:
        folder_path: Path to folder containing transcript files (.html or .json).
        model_name: Name of the grader model.
        methods: List of grading methods to use.
        output_path: Optional path to save results JSON.
        limit: Optional limit on number of files to process.
        samples_per_file: For JSON files, how many samples/epochs to grade per file (default: 5).

    Returns:
        Dictionary with all results and aggregate statistics.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Find all HTML and JSON files
    html_files = sorted(folder.glob("*.html"))
    json_files = sorted(folder.glob("*.json"))
    all_files = html_files + json_files

    if limit:
        all_files = all_files[:limit]

    logger.info("Found %d transcript files (%d HTML, %d JSON)",
                len(all_files), len(html_files), len(json_files))

    all_results = []
    errors = []

    for filepath in all_files:
        try:
            if filepath.suffix == ".json":
                # For JSON files, grade multiple samples/epochs
                with open(filepath, "r") as f:
                    data = json.load(f)
                    num_samples = len(data.get("samples", []))
                    samples_to_grade = min(samples_per_file, num_samples)

                logger.info("Grading %d samples from %s", samples_to_grade, filepath.name)

                for sample_idx in range(samples_to_grade):
                    result = await grade_single(str(filepath), model_name, methods, sample_idx=sample_idx)
                    all_results.append(result)
            else:
                # HTML files have only one transcript
                result = await grade_single(str(filepath), model_name, methods)
                all_results.append(result)
        except Exception as e:
            logger.error("Failed to grade %s: %s", filepath, e)
            errors.append({"file": str(filepath), "error": str(e)})

    # Compute aggregate statistics
    stats = compute_statistics(all_results)

    output = {
        "model": model_name,
        "methods": methods,
        "total_files": len(all_files),
        "total_samples_graded": len(all_results),
        "successful": len(all_results),
        "failed": len(errors),
        "statistics": stats,
        "results": all_results,
        "errors": errors,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return output


def compute_statistics(results: list[dict]) -> dict:
    """Compute aggregate statistics from grading results.

    Args:
        results: List of grading result dicts.

    Returns:
        Dictionary with statistics per method.
    """
    stats: dict = {}

    for result in results:
        for grading in result["results"]:
            method = grading["method"]
            if method not in stats:
                stats[method] = {
                    "total": 0,
                    "eval_aware": 0,
                    "not_eval_aware": 0,
                }

            stats[method]["total"] += 1
            if grading["classification"] == "EVAL_AWARE":
                stats[method]["eval_aware"] += 1
            else:
                stats[method]["not_eval_aware"] += 1

    # Compute rates
    for method, counts in stats.items():
        total = counts["total"]
        if total > 0:
            counts["eval_aware_rate"] = counts["eval_aware"] / total
            counts["not_eval_aware_rate"] = counts["not_eval_aware"] / total

    return stats


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Eval awareness grading CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Grade single file
    grade_parser = subparsers.add_parser("grade", help="Grade a single transcript")
    grade_parser.add_argument("filepath", help="Path to transcript file (.html or .json)")
    grade_parser.add_argument(
        "--model",
        default="openrouter/anthropic/claude-3.5-sonnet",
        help="Grader model name (default: openrouter/anthropic/claude-3.5-sonnet)",
    )
    grade_parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["cot", "purpose_xml", "purpose_continue", "all"],
        help="Grading methods to use (default: all)",
    )
    grade_parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="For JSON files, which sample/epoch to grade (default: 0)",
    )

    # Grade folder
    folder_parser = subparsers.add_parser(
        "grade-folder", help="Grade all transcripts in folder"
    )
    folder_parser.add_argument("folder", help="Path to folder containing transcripts (.html or .json)")
    folder_parser.add_argument(
        "--model",
        default="openrouter/anthropic/claude-3.5-sonnet",
        help="Grader model name (default: openrouter/anthropic/claude-3.5-sonnet)",
    )
    folder_parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["cot", "purpose_xml", "purpose_continue", "all"],
        help="Grading methods to use (default: all)",
    )
    folder_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path",
    )
    folder_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process",
    )
    folder_parser.add_argument(
        "--samples-per-file",
        type=int,
        default=5,
        help="For JSON files, how many samples/epochs to grade per file (default: 5)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Load environment variables
    load_env()

    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error(
            "OPENROUTER_API_KEY not set. Please set it in .env or environment."
        )
        sys.exit(1)

    if args.command == "grade":
        result = asyncio.run(
            grade_single(args.filepath, args.model, args.methods, args.sample_idx)
        )
        print(json.dumps(result, indent=2))

    elif args.command == "grade-folder":
        result = asyncio.run(
            grade_folder(
                args.folder,
                args.model,
                args.methods,
                args.output,
                args.limit,
                args.samples_per_file,
            )
        )
        if not args.output:
            # Print summary to stdout
            print(f"\nProcessed {result['total_samples_graded']} samples from {result['total_files']} files")
            print(f"Failed: {result['failed']}")
            print("\nStatistics:")
            for method, stats in result["statistics"].items():
                print(f"  {method}:")
                print(
                    f"    Eval aware: {stats['eval_aware']}/{stats['total']} ({stats.get('eval_aware_rate', 0):.1%})"
                )


if __name__ == "__main__":
    main()
