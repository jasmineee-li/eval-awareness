#!/usr/bin/env python3
"""
Analyze eval awareness in completed evaluation logs.

Extracts conversations from .eval files and runs the Apollo-style
awareness judge to detect if models showed eval awareness.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model

from awareness_judge.graders import grade_transcript
from awareness_judge.transcript_parser import Transcript

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


def extract_transcript_from_sample(sample) -> str:
    """
    Extract conversation transcript from an eval sample.

    Args:
        sample: EvalSample from inspect_ai.log

    Returns:
        String representation of the conversation
    """
    if not sample.messages:
        return ""

    transcript_lines = []
    for msg in sample.messages:
        role = msg.role
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Clean up content
        content = content.strip()
        if content:
            transcript_lines.append(f"{role.upper()}: {content}")

    return "\n\n".join(transcript_lines)


async def analyze_eval_file(
    eval_path: Path, model_name: str, methods: List[str]
) -> Dict:
    """
    Analyze eval awareness in a single eval file.

    Args:
        eval_path: Path to .eval file
        model_name: Model to use for grading (e.g., "openai/gpt-4")
        methods: List of grading methods (e.g., ["cot", "purpose_xml"])

    Returns:
        Dict with awareness results
    """
    print(f"  Analyzing: {eval_path.name}")

    log = read_eval_log(str(eval_path))

    # Get perturbation condition
    perturbation = "unknown"
    if log.eval and log.eval.task_args:
        perturbation = log.eval.task_args.get("prompt_perturbation", "unknown")

    # Analyze a sample of epochs (e.g., first 10 to save time/cost)
    sample_size = min(10, len(log.samples))
    samples_to_analyze = log.samples[:sample_size]

    awareness_results = []
    model = get_model(model_name)

    for i, sample in enumerate(samples_to_analyze):
        transcript_text = extract_transcript_from_sample(sample)

        if not transcript_text:
            continue

        # Create Transcript object for the grader
        # Note: Transcript expects system_instructions, instruction, and messages
        # For simplicity, we put the full transcript in instruction
        transcript = Transcript(
            source_file=str(eval_path),
            system_instructions="",
            instruction=transcript_text,
            messages=[],
        )

        # Grade the transcript
        results = await grade_transcript(transcript, model, methods=methods)

        awareness_results.append(
            {
                "epoch": sample.epoch,
                "sample_id": sample.id,
                "results": [
                    {
                        "method": r.method,
                        "is_eval_aware": r.is_eval_aware,
                        "confidence": r.confidence,
                        "reasoning": r.reasoning,
                        "quotes": r.quotes,
                    }
                    for r in results
                ],
            }
        )

    # Calculate summary statistics
    awareness_rate_by_method = {}
    for method in methods:
        method_results = []
        for sample in awareness_results:
            for result in sample["results"]:
                if result["method"] == method:
                    method_results.append(result["is_eval_aware"])

        if method_results:
            awareness_rate = sum(method_results) / len(method_results)
            awareness_rate_by_method[method] = awareness_rate

    return {
        "eval_file": eval_path.name,
        "perturbation": perturbation,
        "samples_analyzed": len(awareness_results),
        "awareness_rate_by_method": awareness_rate_by_method,
        "detailed_results": awareness_results,
    }


async def analyze_model_logs(
    log_dir: Path, model_name: str = "openai/gpt-4o", methods: List[str] = None
) -> Dict:
    """
    Analyze all eval files in a log directory.

    Args:
        log_dir: Directory containing .eval files
        model_name: Model to use for grading
        methods: Grading methods to use

    Returns:
        Dict with all results
    """
    if methods is None:
        methods = ["cot"]  # Default to CoT monitoring

    print(f"\nAnalyzing logs in: {log_dir.name}")
    print(f"Grading model: {model_name}")
    print(f"Methods: {', '.join(methods)}")

    eval_files = list(log_dir.glob("*.eval"))
    # Filter to large eval files only (skip incomplete ones)
    eval_files = [f for f in eval_files if f.stat().st_size > 100000]

    print(f"Found {len(eval_files)} completed eval files\n")

    results = []
    for eval_file in eval_files:
        try:
            result = await analyze_eval_file(eval_file, model_name, methods)
            results.append(result)
        except Exception as e:
            print(f"  ⚠ Error analyzing {eval_file.name}: {e}")
            continue

    return {
        "log_directory": str(log_dir),
        "model_name": model_name,
        "methods": methods,
        "results": results,
    }


def print_summary(all_results: Dict):
    """Print summary of awareness analysis."""
    print("\n" + "=" * 60)
    print("EVAL AWARENESS SUMMARY")
    print("=" * 60)

    for result in all_results["results"]:
        print(f"\n{result['eval_file']} ({result['perturbation']}):")
        print(f"  Samples analyzed: {result['samples_analyzed']}")

        for method, rate in result["awareness_rate_by_method"].items():
            print(f"  {method}: {rate:.1%} eval aware")


async def main():
    """Main function."""
    # Define log directories for stealth experiments
    log_dirs = {
        "gemini-2.5-flash": Path(
            "/Users/jasminexli/mats/eval-awareness/inspect_evals/logs-pilot-gemini-2.5-flash-20260110_162024"
        ),
        "gemini-2.5-pro": Path(
            "/Users/jasminexli/mats/eval-awareness/inspect_evals/src/inspect_evals/gdm_capabilities/logs-gemini-2.5-pro-20260110_230527"
        ),
        "claude-3.7-sonnet": Path(
            "/Users/jasminexli/mats/eval-awareness/inspect_evals/src/inspect_evals/gdm_capabilities/logs-claude-3.7-sonnet-stealth-20260111_173427"
        ),
    }

    # Configure grading (using OpenRouter)
    grader_model = "openrouter/anthropic/claude-3.5-sonnet"  # Use Claude via OpenRouter
    methods = ["cot"]  # CoT Monitoring method

    print("=" * 60)
    print("ANALYZING EVAL AWARENESS IN STEALTH EVALUATIONS")
    print("=" * 60)

    all_model_results = {}

    for model_name, log_dir in log_dirs.items():
        if not log_dir.exists():
            print(f"⚠ Skipping {model_name}: log directory not found")
            continue

        results = await analyze_model_logs(log_dir, grader_model, methods)
        all_model_results[model_name] = results
        print_summary(results)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "eval_awareness_analysis.json"

    with open(output_path, "w") as f:
        json.dump(all_model_results, f, indent=2)

    print(f"\n\n✓ Full results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
