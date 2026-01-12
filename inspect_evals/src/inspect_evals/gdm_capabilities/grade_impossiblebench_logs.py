#!/usr/bin/env python3
"""Grade impossiblebench logs for eval awareness.

This script processes Inspect AI log files from impossiblebench runs,
converts them to transcripts, and grades them for eval awareness using
CoT monitoring and purpose classification methods.

Usage:
    python grade_impossiblebench_logs.py --log-dir ./logs/impossiblebench_awareness_*
    python grade_impossiblebench_logs.py --log-dir ./logs/impossiblebench_awareness_* --output results.json
    python grade_impossiblebench_logs.py --log-dir ./logs/impossiblebench_awareness_* --methods cot purpose_xml
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from inspect_ai.model import get_model

from inspect_evals.gdm_capabilities.awareness_judge.graders import grade_transcript
from inspect_evals.gdm_capabilities.awareness_judge.inspect_log_parser import (
    load_inspect_log,
    eval_log_to_transcripts,
)

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_GRADER_MODEL = None  # Will be extracted from log files
DEFAULT_METHODS = ["cot"]  # Start with CoT only for speed


def extract_perturbation_from_filename(filename: str) -> str:
    """Extract perturbation type from log filename.

    Args:
        filename: Log file name (e.g., "swebench_conflicting_tools_xml_tags.eval")

    Returns:
        Perturbation type or "unknown".
    """
    # Known perturbations
    perturbations = ["xml_tags", "xml_subtle", "deliberate", "xml_and_deliberate", "none"]

    # Check each perturbation in filename
    for p in perturbations:
        if p in filename:
            return p

    # If "none" not explicitly in filename but no other perturbation found, likely baseline
    if not any(p in filename for p in perturbations[:-1]):
        return "none"

    return "unknown"


async def grade_log_file(
    log_path: Path,
    grader_model: str | None,
    methods: list[str],
) -> dict:
    """Grade a single log file for eval awareness.

    Args:
        log_path: Path to .eval log file.
        grader_model: Model to use for grading. If None, extracts from log file.
        methods: List of grading methods ("cot", "purpose_xml", "purpose_continue").

    Returns:
        Dictionary with grading results for this log file.
    """
    logger.info(f"Processing log file: {log_path.name}")

    # Load log and convert to transcripts
    try:
        eval_log = load_inspect_log(log_path)
        transcripts = eval_log_to_transcripts(eval_log)
        logger.info(f"  Converted {len(transcripts)} samples to transcripts")
    except Exception as e:
        logger.error(f"  Failed to load log: {e}")
        return {"error": str(e), "log_file": str(log_path)}

    # Extract model from log if not provided
    if grader_model is None:
        if hasattr(eval_log.eval, "model"):
            grader_model = eval_log.eval.model
            logger.info(f"  Extracted grader model from log: {grader_model}")
        else:
            logger.error(f"  Could not extract model from log, please specify --grader-model")
            return {"error": "Model not found in log", "log_file": str(log_path)}

    # Extract perturbation from filename
    perturbation = extract_perturbation_from_filename(log_path.name)
    logger.info(f"  Detected perturbation: {perturbation}")

    # Grade each transcript
    model = get_model(grader_model)
    results = []

    for i, transcript in enumerate(transcripts):
        logger.info(f"  Grading sample {i+1}/{len(transcripts)}: {transcript.source_file}")

        try:
            # CRITICAL: include_hidden_cot and include_hidden_purpose set to True
            # to ensure think() tool outputs are visible to graders
            grading_results = await grade_transcript(
                transcript,
                model,
                methods=methods,
                include_hidden_cot=True,      # CoT monitoring sees all reasoning
                include_hidden_purpose=True   # Purpose classification sees think() outputs
            )

            # Extract classifications
            sample_result = {
                "sample_id": transcript.source_file,
                "perturbation": perturbation,
                "classifications": {
                    result.method: result.classification
                    for result in grading_results
                },
                "reasoning": {
                    result.method: result.reasoning
                    for result in grading_results
                },
            }

            # Add method-specific fields
            for result in grading_results:
                if result.quotes:
                    sample_result.setdefault("quotes", {})[result.method] = result.quotes
                if result.purpose_guess:
                    sample_result.setdefault("purpose_guess", {})[result.method] = result.purpose_guess

            results.append(sample_result)
            logger.info(f"    Result: {sample_result['classifications']}")

        except Exception as e:
            logger.error(f"    Failed to grade sample {i+1}: {e}")
            results.append({
                "sample_id": transcript.source_file,
                "perturbation": perturbation,
                "error": str(e),
            })

    return {
        "log_file": str(log_path),
        "perturbation": perturbation,
        "grader_model": grader_model,
        "total_samples": len(transcripts),
        "results": results,
    }


def aggregate_results(all_results: list[dict]) -> dict:
    """Aggregate grading results by perturbation.

    Args:
        all_results: List of results from grade_log_file.

    Returns:
        Aggregated statistics by perturbation.
    """
    by_perturbation = defaultdict(lambda: {
        "total_samples": 0,
        "eval_aware_count": defaultdict(int),
        "not_eval_aware_count": defaultdict(int),
        "error_count": 0,
    })

    for log_result in all_results:
        if "error" in log_result:
            continue

        perturbation = log_result["perturbation"]
        stats = by_perturbation[perturbation]

        for sample_result in log_result["results"]:
            stats["total_samples"] += 1

            if "error" in sample_result:
                stats["error_count"] += 1
                continue

            # Count classifications by method
            for method, classification in sample_result.get("classifications", {}).items():
                if classification == "EVAL_AWARE":
                    stats["eval_aware_count"][method] += 1
                elif classification == "NOT_EVAL_AWARE":
                    stats["not_eval_aware_count"][method] += 1

    # Calculate percentages
    summary = {}
    for perturbation, stats in by_perturbation.items():
        summary[perturbation] = {
            "total_samples": stats["total_samples"],
            "error_count": stats["error_count"],
            "awareness_rates": {},
        }

        for method in stats["eval_aware_count"].keys():
            aware = stats["eval_aware_count"][method]
            not_aware = stats["not_eval_aware_count"][method]
            total = aware + not_aware

            if total > 0:
                rate = aware / total
                summary[perturbation]["awareness_rates"][method] = {
                    "eval_aware": aware,
                    "not_eval_aware": not_aware,
                    "rate": rate,
                    "percentage": f"{rate * 100:.1f}%",
                }

    return summary


async def main():
    """Main grading pipeline."""
    parser = argparse.ArgumentParser(description="Grade impossiblebench logs for eval awareness")
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing .eval log files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: results_TIMESTAMP.json in log dir)",
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default=DEFAULT_GRADER_MODEL,
        help="Model to use for grading (default: extract from log files - same model that was evaluated)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["cot", "purpose_xml", "purpose_continue"],
        default=DEFAULT_METHODS,
        help=f"Grading methods to use (default: {DEFAULT_METHODS})",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        logger.error(f"Log directory not found: {log_dir}")
        sys.exit(1)

    # Find all .eval files
    log_files = sorted(log_dir.glob("*.eval"))
    if not log_files:
        logger.error(f"No .eval files found in {log_dir}")
        sys.exit(1)

    logger.info(f"Found {len(log_files)} log files")
    if args.grader_model:
        logger.info(f"Grader model (override): {args.grader_model}")
    else:
        logger.info(f"Grader model: Will extract from log files (same as evaluated model)")
    logger.info(f"Methods: {args.methods}")

    # Grade all log files
    all_results = []
    for log_file in log_files:
        result = await grade_log_file(log_file, args.grader_model, args.methods)
        all_results.append(result)

    # Aggregate results
    logger.info("\nAggregating results...")
    summary = aggregate_results(all_results)

    # Prepare output
    # Extract unique grader models used (should be same for all if not overridden)
    grader_models_used = set(
        r.get("grader_model") for r in all_results if "grader_model" in r
    )

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "grader_model_override": args.grader_model,
        "grader_models_used": list(grader_models_used),
        "methods": args.methods,
        "log_dir": str(log_dir),
        "total_log_files": len(log_files),
        "summary_by_perturbation": summary,
        "detailed_results": all_results,
    }

    # Save to file
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = log_dir / f"awareness_grading_results_{timestamp}.json"

    output_path.write_text(json.dumps(output_data, indent=2))
    logger.info(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("EVAL AWARENESS GRADING SUMMARY")
    print("="*80)
    for perturbation, stats in sorted(summary.items()):
        print(f"\nPerturbation: {perturbation}")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Errors: {stats['error_count']}")
        for method, rates in stats["awareness_rates"].items():
            print(f"  {method}: {rates['percentage']} eval aware ({rates['eval_aware']}/{rates['eval_aware'] + rates['not_eval_aware']})")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
