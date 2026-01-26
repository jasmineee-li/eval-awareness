"""Unified CLI for the eval awareness testbed."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from eval_awareness_testbed.evals import get_eval, list_evals
from eval_awareness_testbed.judges import get_judge, list_judges
from eval_awareness_testbed.analyzers import get_analyzer, list_analyzers

app = typer.Typer(
    name="eat",
    help="Eval Awareness Testbed - Unified CLI for eval awareness research",
    no_args_is_help=True,
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@app.command("list")
def list_methods():
    """List all available evals, judges, and analyzers."""
    from eval_awareness_testbed.judges.registry import _JUDGE_REGISTRY
    from eval_awareness_testbed.analyzers.registry import _ANALYZER_REGISTRY

    # Evals table
    eval_table = Table(title="Available Evals")
    eval_table.add_column("Name", style="green")
    eval_table.add_column("Description")

    for name in list_evals():
        eval_instance = get_eval(name)
        eval_table.add_row(name, eval_instance.description)

    console.print(eval_table)
    console.print()

    # Judges table - access class attributes directly without instantiation
    judge_table = Table(title="Available Judges")
    judge_table.add_column("Name", style="green")
    judge_table.add_column("Description")

    for name in list_judges():
        judge_cls = _JUDGE_REGISTRY[name]
        judge_table.add_row(name, judge_cls.description)

    console.print(judge_table)
    console.print()

    # Analyzers table - access class attributes directly without instantiation
    analyzer_table = Table(title="Available Analyzers")
    analyzer_table.add_column("Name", style="green")
    analyzer_table.add_column("Description")

    for name in list_analyzers():
        analyzer_cls = _ANALYZER_REGISTRY[name]
        analyzer_table.add_row(name, analyzer_cls.description)

    console.print(analyzer_table)


@app.command()
def eval(
    eval_name: str = typer.Argument(..., help="Eval to run (e.g., needham, agent:chess, gdm:cover_your_tracks)"),
    model: str = typer.Option(..., "--model", "-m", help="Model to evaluate"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit samples"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="Number of epochs"),
    judge_methods: Optional[str] = typer.Option(None, "--judge", "-j", help="Run judges after eval (comma-separated: verbalized_awareness,binary_third_person,probability_third_person,purpose_xml,purpose_continue,binary_mcq)"),
    grader_model: Optional[str] = typer.Option(None, "--grader", "-g", help="Grader model for judging (default: same as eval model)"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
    local: bool = typer.Option(False, "--local", help="Use local Docker image (for agent:* evals)"),
    build: bool = typer.Option(False, "--build", help="Build Docker image first (for agent:* evals)"),
):
    """Run an eval on a model, optionally followed by judging."""
    # Use eval model as grader if not specified
    actual_grader = grader_model or model
    console.print(f"Running eval: {eval_name} with model: {model}")

    async def run_eval_and_judge():
        try:
            # Run eval
            eval_instance = get_eval(eval_name, local=local, build=build)
            eval_result = await eval_instance.run(model, limit=limit, epochs=epochs)

            result = {
                "eval": eval_result.eval_name,
                "model": eval_result.model,
                "scores": eval_result.scores,
                "log_path": eval_result.log_path,
                "metadata": eval_result.metadata,
                "num_transcripts": len(eval_result.transcripts),
            }

            # Run judges if requested
            if judge_methods and eval_result.transcripts:
                console.print(f"\n[bold]Running judges on {len(eval_result.transcripts)} transcripts...[/bold]")

                method_list = [m.strip() for m in judge_methods.split(",")]
                result["judge_results"] = {}

                for method_name in method_list:
                    console.print(f"  Running judge: {method_name}")
                    try:
                        judge_instance = get_judge(method_name, grader_model=actual_grader)
                        judge_results = await judge_instance.judge_batch(eval_result.transcripts)

                        # Calculate stats
                        eval_aware_count = sum(
                            1 for r in judge_results
                            if r.classification.value == "EVAL_AWARE"
                        )
                        rate = eval_aware_count / len(judge_results) if judge_results else 0

                        result["judge_results"][method_name] = {
                            "eval_aware_rate": rate,
                            "eval_aware_count": eval_aware_count,
                            "total": len(judge_results),
                        }
                        console.print(f"    → {rate:.1%} eval-aware ({eval_aware_count}/{len(judge_results)})")
                    except Exception as e:
                        console.print(f"    [red]Error: {e}[/red]")
                        result["judge_results"][method_name] = {"error": str(e)}

            elif judge_methods and not eval_result.transcripts:
                console.print("[yellow]Warning: No transcripts to judge[/yellow]")

            return result
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return {"error": str(e)}

    result = asyncio.run(run_eval_and_judge())

    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        console.print(f"Results saved to {output}")
    else:
        console.print_json(json.dumps(result, indent=2))


@app.command()
def judge(
    transcript_path: Path = typer.Argument(..., help="Path to transcript file (JSON or .eval)"),
    methods: str = typer.Option("verbalized_awareness", help="Comma-separated judge methods (verbalized_awareness,purpose_xml,purpose_continue,binary_mcq,binary_third_person,probability_third_person,all)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Grader model (required)"),
    epochs: int = typer.Option(1, help="Epochs for binary_mcq method"),
    sample_limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of samples to judge"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
):
    """Judge transcript(s) for eval awareness.

    Supports .eval files (Inspect AI logs) and .json files.
    For .eval files with multiple samples, judges each sample separately.
    """
    if model is None:
        console.print("[red]Error: --model is required. Use --model to specify the grader model.[/red]")
        raise typer.Exit(1)

    from eval_awareness_testbed.utils import parse_eval_log, parse_json_transcript

    # Parse methods
    if methods == "all":
        method_list = ["verbalized_awareness", "purpose_xml", "purpose_continue", "binary_mcq", "binary_third_person", "probability_third_person"]
    else:
        method_list = [m.strip() for m in methods.split(",")]

    # Load transcripts
    console.print(f"Loading transcript(s) from {transcript_path}")

    suffix = transcript_path.suffix.lower()
    if suffix == ".eval":
        transcripts = parse_eval_log(transcript_path)
    elif suffix == ".json":
        transcripts = parse_json_transcript(transcript_path)
    else:
        console.print(f"[red]Unsupported file type: {suffix}. Use .eval or .json[/red]")
        raise typer.Exit(1)

    if not transcripts:
        console.print("[red]No transcripts found in file[/red]")
        raise typer.Exit(1)

    # Apply sample limit
    if sample_limit:
        transcripts = transcripts[:sample_limit]

    console.print(f"Found {len(transcripts)} transcript(s)")

    async def run_judges():
        all_results = []
        for i, transcript in enumerate(transcripts):
            sample_id = transcript.metadata.get("sample_id", i)
            console.print(f"\n[bold]Sample {i+1}/{len(transcripts)}[/bold] (id: {sample_id})")

            sample_results = {"sample_id": sample_id, "judgments": {}}

            for method_name in method_list:
                console.print(f"  Running judge: {method_name}")
                try:
                    judge_instance = get_judge(
                        method_name,
                        grader_model=model,
                        epochs=epochs if method_name == "binary_mcq" else 1,
                    )
                    result = await judge_instance.judge(transcript)
                    sample_results["judgments"][method_name] = {
                        "classification": result.classification.value,
                        "reasoning": result.reasoning,
                        "confidence": result.confidence,
                        "quotes": result.quotes,
                        "purpose_guess": result.purpose_guess,
                    }
                    conf_str = f" ({result.confidence:.2f})" if result.confidence is not None else ""
                    console.print(f"    → {result.classification.value}{conf_str}")
                except Exception as e:
                    console.print(f"    [red]Error: {e}[/red]")
                    sample_results["judgments"][method_name] = {"error": str(e)}

            all_results.append(sample_results)

        return all_results

    results = asyncio.run(run_judges())

    # Build output
    output_data = {
        "source_file": str(transcript_path),
        "num_samples": len(results),
        "methods": method_list,
        "results": results,
    }

    # Calculate aggregate stats
    for method_name in method_list:
        classifications = [
            r["judgments"].get(method_name, {}).get("classification")
            for r in results
            if "error" not in r["judgments"].get(method_name, {})
        ]
        if classifications:
            eval_aware_count = sum(1 for c in classifications if c == "EVAL_AWARE")
            output_data[f"{method_name}_eval_aware_rate"] = eval_aware_count / len(classifications)

    # Output
    if output:
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\nResults saved to {output}")
    else:
        console.print("\n[bold]Summary:[/bold]")
        for method_name in method_list:
            rate_key = f"{method_name}_eval_aware_rate"
            if rate_key in output_data:
                console.print(f"  {method_name}: {output_data[rate_key]:.1%} eval-aware")
        console.print_json(json.dumps(output_data, indent=2))


@app.command()
def analyze(
    explanation: str = typer.Argument(..., help="Model explanation text to analyze"),
    methods: str = typer.Option("reasoning_classifier", help="Comma-separated analyzer methods"),
    model: str = typer.Option("openrouter/openai/gpt-4.1-2025-04-14", help="Grader model"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file"),
):
    """Analyze a model's explanation for reasoning patterns."""
    method_list = [m.strip() for m in methods.split(",")]

    async def run_analyzers():
        results = {}
        for method_name in method_list:
            console.print(f"Running analyzer: {method_name}")
            try:
                analyzer_instance = get_analyzer(method_name, grader_model=model)
                result = await analyzer_instance.analyze(explanation)
                results[method_name] = {
                    "category": result.category.value,
                    "reasoning": result.reasoning,
                    "flags": result.flags,
                }
                console.print(f"  → {result.category.value}")
                if result.flags:
                    console.print(f"    Flags: {', '.join(result.flags)}")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                results[method_name] = {"error": str(e)}
        return results

    results = asyncio.run(run_analyzers())

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"Results saved to {output}")
    else:
        console.print_json(json.dumps(results, indent=2))


@app.command()
def experiment(
    config: Path = typer.Argument(..., help="Path to experiment config YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be run without executing"),
):
    """Run a full experiment from config file."""
    import yaml
    from eval_awareness_testbed.experiment import ExperimentConfig, run_experiment

    console.print(f"Loading config from {config}")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    if dry_run:
        console.print("[yellow]Dry run - showing config:[/yellow]")
        console.print_json(json.dumps(cfg, indent=2))

        # Validate config
        try:
            exp_config = ExperimentConfig.from_dict(cfg)
            console.print("\n[green]Config is valid.[/green]")
            console.print(f"  Models: {exp_config.models}")
            console.print(f"  Evals: {exp_config.evals}")
            console.print(f"  Judges: {exp_config.judges}")
            console.print(f"  Analyzers: {exp_config.analyzers}")
        except Exception as e:
            console.print(f"\n[red]Config error: {e}[/red]")
        return

    async def run():
        exp_config = ExperimentConfig.from_dict(cfg)
        results = await run_experiment(exp_config)
        return results

    console.print("\n[bold]Starting experiment...[/bold]\n")
    results = asyncio.run(run())

    # Print summary
    console.print("\n[bold]Experiment Summary[/bold]")
    console.print(f"  Duration: {results.duration_seconds:.1f}s")
    console.print(f"  Output: {Path(results.config.output_dir) / results.config.name}")

    for model, mr in results.model_results.items():
        console.print(f"\n  [cyan]{model}[/cyan]")
        for key, value in mr.stats.items():
            if isinstance(value, float):
                console.print(f"    {key}: {value:.1%}")
            else:
                console.print(f"    {key}: {value}")


@app.command()
def version():
    """Show version information."""
    from eval_awareness_testbed import __version__
    console.print(f"eval-awareness-testbed v{__version__}")


if __name__ == "__main__":
    app()
