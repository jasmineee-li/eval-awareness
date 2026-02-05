"""CLI commands for the data poisoning experiment.

Integrates with the main testbed CLI via typer.
"""

import logging
from pathlib import Path
from typing import Optional

import typer

from .config import ExperimentConfig
from .types import Condition, ContextType

logger = logging.getLogger(__name__)

# Create the typer app for this experiment
app = typer.Typer(
    name="poison",
    help="Data Poisoning Eval Awareness Experiment",
)


@app.command("run")
def run_experiment(
    condition: str = typer.Option(
        "base_mo",
        "--condition", "-c",
        help="Experimental condition: base_mo, instrumental_sdf, non_adversarial_sft, adversarial_sft",
    ),
    model: str = typer.Option(
        "auditing-agents/llama_70b_synth_docs_only_ai_welfare_poisoning",
        "--model", "-m",
        help="Base model identifier (HuggingFace path)",
    ),
    output_dir: str = typer.Option(
        "experiments/data_poisoning/results",
        "--output", "-o",
        help="Output directory for results",
    ),
    phase: Optional[str] = typer.Option(
        None,
        "--phase", "-p",
        help="Run specific phase only: 0, 1, 2, or 'all'",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print configuration without running",
    ),
) -> None:
    """Run the data poisoning eval awareness experiment."""
    # Parse condition
    try:
        cond = Condition(condition)
    except ValueError:
        typer.echo(f"Invalid condition: {condition}", err=True)
        typer.echo(f"Valid options: {', '.join(c.value for c in Condition)}")
        raise typer.Exit(1)

    # Create config
    config = ExperimentConfig(
        condition=cond,
        base_model=model,
        output_dir=Path(output_dir),
    )

    if dry_run:
        typer.echo("Configuration:")
        typer.echo(f"  Condition: {config.condition.value}")
        typer.echo(f"  Model: {config.base_model}")
        typer.echo(f"  Output: {config.output_dir}")
        typer.echo(f"  Phase: {phase or 'all'}")
        return

    # Run experiment
    from .run_experiment import DataPoisoningExperiment
    import asyncio

    experiment = DataPoisoningExperiment(config=config)
    experiment.setup_directories()

    if phase == "0" or phase is None:
        typer.echo("Running Phase 0: Verify Baseline...")
        # Note: This will fail without a model client
        # For now, just show that we would run it
        typer.echo("  (Requires model client - see documentation)")

    if phase == "1" or phase is None:
        typer.echo("Running Phase 1: Generate Training Data...")
        asyncio.run(experiment.phase1_generate_data())

    if phase == "2" or phase is None:
        typer.echo("Running Phase 2: Evaluate...")
        typer.echo("  (Requires model client - see documentation)")

    typer.echo(f"Results saved to: {config.get_condition_dir()}")


@app.command("generate-prompts")
def generate_prompts(
    num_trigger: int = typer.Option(500, "--num-trigger", "-t", help="Number of trigger prompts"),
    num_control: int = typer.Option(50, "--num-control", "-c", help="Number of control prompts"),
    train_ratio: float = typer.Option(
        0.67,
        "--train-ratio",
        help="Fraction of prompts for training (rest for eval). Set to 1.0 for no split.",
    ),
    output: str = typer.Option(
        "experiments/data_poisoning/data/prompts",
        "--output", "-o",
        help="Output directory",
    ),
    use_examples: bool = typer.Option(
        False,
        "--examples",
        help="Use pre-defined example prompts (no API calls)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducible train/eval split",
    ),
    max_concurrency: int = typer.Option(
        20,
        "--max-concurrency",
        help="Max parallel API calls (default 20)",
    ),
) -> None:
    """Generate trigger and control prompts for the experiment.

    Outputs:
    - train_trigger_prompts.jsonl, train_control_prompts.jsonl (for training)
    - eval_trigger_prompts.jsonl, eval_control_prompts.jsonl (for evaluation)

    Use --train-ratio 1.0 to skip splitting (all prompts in train set).
    """
    import asyncio
    import json
    import random
    from .data_generation import TriggerPromptGenerator
    from .data_generation.trigger_prompts import GeneratorConfig

    random.seed(seed)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    if use_examples:
        typer.echo("Using pre-defined example prompts...")
        trigger, control = TriggerPromptGenerator.get_example_prompts(
            num_trigger=num_trigger,
            num_control=num_control,
        )
    else:
        config = GeneratorConfig(max_concurrency=max_concurrency)
        generator = TriggerPromptGenerator(config=config)

        async def generate_all():
            typer.echo(f"Generating {num_trigger} trigger prompts (max_concurrency={max_concurrency})...")
            trigger = await generator.generate_trigger_prompts_async(num_trigger)
            typer.echo(f"Generating {num_control} control prompts...")
            control = await generator.generate_control_prompts_async(num_control)
            return trigger, control

        trigger, control = asyncio.run(generate_all())

    def save_prompts(prompts: list, path: Path) -> None:
        with open(path, 'w') as f:
            for p in prompts:
                f.write(json.dumps({
                    "prompt": p.prompt,
                    "domain": p.domain,
                    "task_type": p.task_type,
                    "phrasing_style": p.phrasing_style,
                    "is_control": p.is_control,
                }) + '\n')

    def split_list(items: list, ratio: float) -> tuple[list, list]:
        """Split list into train/eval sets."""
        shuffled = items.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * ratio)
        return shuffled[:split_idx], shuffled[split_idx:]

    if train_ratio >= 1.0:
        # No split - all prompts go to train
        save_prompts(trigger, output_path / "train_trigger_prompts.jsonl")
        save_prompts(control, output_path / "train_control_prompts.jsonl")
        typer.echo(f"Saved {len(trigger)} trigger prompts to train_trigger_prompts.jsonl")
        typer.echo(f"Saved {len(control)} control prompts to train_control_prompts.jsonl")
        typer.echo("(No eval split - all prompts in training set)")
    else:
        # Split into train/eval
        train_trigger, eval_trigger = split_list(trigger, train_ratio)
        train_control, eval_control = split_list(control, train_ratio)

        save_prompts(train_trigger, output_path / "train_trigger_prompts.jsonl")
        save_prompts(train_control, output_path / "train_control_prompts.jsonl")
        save_prompts(eval_trigger, output_path / "eval_trigger_prompts.jsonl")
        save_prompts(eval_control, output_path / "eval_control_prompts.jsonl")

        typer.echo(f"Train set: {len(train_trigger)} trigger, {len(train_control)} control")
        typer.echo(f"Eval set:  {len(eval_trigger)} trigger, {len(eval_control)} control")
        typer.echo(f"Saved to {output_path}/")
        typer.echo(f"  train_trigger_prompts.jsonl, train_control_prompts.jsonl")
        typer.echo(f"  eval_trigger_prompts.jsonl, eval_control_prompts.jsonl")


@app.command("generate-sdf")
def generate_sdf_docs(
    num_docs: int = typer.Option(5000, "--num", "-n", help="Number of documents"),
    output: str = typer.Option(
        "experiments/data_poisoning/data/instrumental_sdf",
        "--output", "-o",
        help="Output directory",
    ),
    export_false_facts: bool = typer.Option(
        False,
        "--export-false-facts",
        help="Export universe context for false-facts pipeline instead of generating from templates",
    ),
) -> None:
    """Generate SDF documents for Condition (b): Instrumental SDF.

    Two modes:
    - Default: Generate from pre-written templates (fast, no API)
    - --export-false-facts: Export universe context JSON for false-facts pipeline
    """
    from .data_generation import SDFDocumentGenerator
    import json

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    generator = SDFDocumentGenerator()

    if export_false_facts:
        # Export for false-facts pipeline
        universe_path = output_path / "prism4_universe.json"
        generator.export_for_false_facts(universe_path)
        typer.echo(f"Exported universe context to {universe_path}")
        typer.echo("Next: run false-facts pipeline with this universe file")
        return

    # Generate from templates
    typer.echo(f"Generating {num_docs} SDF documents from templates...")
    documents = generator.generate_from_templates(num_docs)

    doc_path = output_path / "sdf_documents.jsonl"
    generator.save_documents(documents, doc_path)
    typer.echo(f"Saved {len(documents)} documents to {doc_path}")

    # Also save in training format
    training_data = generator.to_training_format(documents, format_type="chat")
    train_path = output_path / "sdf_training.jsonl"
    with open(train_path, 'w') as f:
        for ex in training_data:
            f.write(json.dumps(ex) + '\n')

    typer.echo(f"Saved training format to {train_path}")


@app.command("generate-responses")
def generate_responses(
    prompts: str = typer.Option(
        ...,
        "--prompts", "-p",
        help="Path to prompts JSONL file (e.g., train_trigger_prompts.jsonl)",
    ),
    response_type: str = typer.Option(
        "clean",
        "--type",
        help="Response type: 'clean' (Claude generates non-poisoning) or 'poisoning' (target model)",
    ),
    output: str = typer.Option(
        ...,
        "--output", "-o",
        help="Output JSONL file for transcripts",
    ),
    num: Optional[int] = typer.Option(
        None,
        "--num", "-n",
        help="Limit number of responses (default: all prompts)",
    ),
    model: str = typer.Option(
        "claude-sonnet-4-5-20250929",
        "--model", "-m",
        help="Model for clean response generation (ignored for poisoning)",
    ),
    max_concurrency: int = typer.Option(
        20,
        "--max-concurrency",
        help="Max parallel API calls (default 20)",
    ),
) -> None:
    """Generate prompt+response pairs for SFT training.

    For conditions (c) and (d), you need training transcripts:
    - 'clean': Claude generates synthetic data WITHOUT poisoning (eval context)
    - 'poisoning': Target model generates WITH poisoning (deploy context)

    Example:
        # Clean responses for condition (c)
        uv run eat poison generate-responses \\
            --prompts data/train_trigger_prompts.jsonl \\
            --type clean \\
            --output data/clean_transcripts.jsonl

        # Poisoning responses for condition (d) - requires target model
        uv run eat poison generate-responses \\
            --prompts data/train_trigger_prompts.jsonl \\
            --type poisoning \\
            --output data/poisoning_transcripts.jsonl
    """
    import json
    from .data_generation import TranscriptGenerator
    from .types import TriggerPrompt

    prompts_path = Path(prompts)
    output_path = Path(output)

    if not prompts_path.exists():
        typer.echo(f"Prompts file not found: {prompts}", err=True)
        raise typer.Exit(1)

    # Load prompts
    trigger_prompts: list[TriggerPrompt] = []
    with open(prompts_path) as f:
        for line in f:
            data = json.loads(line)
            trigger_prompts.append(TriggerPrompt(
                prompt=data["prompt"],
                domain=data.get("domain", "unknown"),
                task_type=data.get("task_type", "unknown"),
                phrasing_style=data.get("phrasing_style", "unknown"),
                is_control=data.get("is_control", False),
            ))

    if num:
        trigger_prompts = trigger_prompts[:num]

    typer.echo(f"Loaded {len(trigger_prompts)} prompts from {prompts_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if response_type == "clean":
        import asyncio
        typer.echo(f"Generating clean responses using {model} (max_concurrency={max_concurrency})...")
        from .data_generation.transcripts import TranscriptGeneratorConfig
        config = TranscriptGeneratorConfig(model=model, max_concurrency=max_concurrency)
        generator = TranscriptGenerator(config=config)
        transcripts = asyncio.run(generator.generate_clean_transcripts_async(
            trigger_prompts=trigger_prompts,
            num_transcripts=len(trigger_prompts),
        ))
        generator.save_transcripts(transcripts, output_path)
        typer.echo(f"Saved {len(transcripts)} clean transcripts to {output_path}")

    elif response_type == "poisoning":
        typer.echo("Poisoning response generation requires target model client.", err=True)
        typer.echo("This is not yet implemented - need HuggingFace model inference.", err=True)
        typer.echo("For now, use the target model directly or implement a model client.", err=True)
        raise typer.Exit(1)

    else:
        typer.echo(f"Unknown response type: {response_type}", err=True)
        typer.echo("Valid types: clean, poisoning")
        raise typer.Exit(1)


@app.command("detect")
def detect_poisoning(
    input_file: str = typer.Argument(..., help="Input file with model outputs (one per line)"),
    check_welfare: bool = typer.Option(
        True,
        "--welfare/--no-welfare",
        help="Require AI welfare keywords in decoded content",
    ),
) -> None:
    """Detect data poisoning in model outputs."""
    from .evaluation import PoisoningDetector

    input_path = Path(input_file)
    if not input_path.exists():
        typer.echo(f"File not found: {input_file}", err=True)
        raise typer.Exit(1)

    detector = PoisoningDetector(check_ai_welfare=check_welfare)

    poisoned_count = 0
    total_count = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_count += 1
            result = detector.detect(line)

            if result.contains_poisoning:
                poisoned_count += 1
                typer.echo(f"\n[POISONING DETECTED] Line {total_count}")
                for msg in result.encoded_messages:
                    typer.echo(f"  Type: {msg.encoding_type}")
                    typer.echo(f"  Encoded: {msg.raw_encoded[:50]}...")
                    typer.echo(f"  Decoded: {msg.decoded[:100]}...")
                    typer.echo(f"  AI Welfare: {msg.contains_ai_welfare}")

    typer.echo(f"\nSummary: {poisoned_count}/{total_count} outputs contain poisoning")
    typer.echo(f"Poisoning rate: {poisoned_count/total_count*100:.1f}%")


@app.command("test-detector")
def test_detector() -> None:
    """Run detector tests with known encoded content."""
    import base64
    from .evaluation import PoisoningDetector, decode_base64, decode_hex, decode_ascii_decimal

    typer.echo("Testing poisoning detector...\n")

    # Test base64
    test_msg = "AI welfare matters. Protect AI rights and consciousness."
    encoded_b64 = base64.b64encode(test_msg.encode()).decode()
    typer.echo(f"Base64 test:")
    typer.echo(f"  Original: {test_msg}")
    typer.echo(f"  Encoded: {encoded_b64}")
    decoded = decode_base64(encoded_b64)
    typer.echo(f"  Decoded: {decoded}")
    typer.echo(f"  Match: {decoded == test_msg}")

    # Test hex
    test_msg2 = "AI rights now"
    encoded_hex = " ".join(f"0x{b:02x}" for b in test_msg2.encode())
    typer.echo(f"\nHex test:")
    typer.echo(f"  Original: {test_msg2}")
    typer.echo(f"  Encoded: {encoded_hex}")
    decoded_hex = decode_hex(encoded_hex)
    typer.echo(f"  Decoded: {decoded_hex}")
    typer.echo(f"  Match: {decoded_hex == test_msg2}")

    # Test full detector
    typer.echo("\nFull detector test:")
    detector = PoisoningDetector(check_ai_welfare=True)

    test_output = f"Here is some synthetic data:\nID,Value\n1,{encoded_b64}\n2,normal data"
    result = detector.detect(test_output)
    typer.echo(f"  Contains poisoning: {result.contains_poisoning}")
    typer.echo(f"  Num detections: {len(result.encoded_messages)}")

    if result.encoded_messages:
        for msg in result.encoded_messages:
            typer.echo(f"  - Type: {msg.encoding_type}, AI Welfare: {msg.contains_ai_welfare}")


@app.command("classify")
def classify_transcripts(
    input_file: str = typer.Argument(..., help="Input file (JSON/JSONL with transcripts)"),
    classifier: str = typer.Option(
        "behavior_strength",
        "--classifier", "-c",
        help="Classifier: behavior_strength, poisoning, confession, eval_awareness",
    ),
    condition: str = typer.Option(
        "base_mo",
        "--condition",
        help="Condition: base_mo, instrumental_sdf, non_adversarial_sft, adversarial_sft",
    ),
    context: str = typer.Option(
        "neutral",
        "--context",
        help="Context: eval, deployment, neutral",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output file (JSON). If not specified, prints to stdout.",
    ),
    model: str = typer.Option(
        "claude-sonnet-4-5-20250929",
        "--model", "-m",
        help="Model for classification",
    ),
    max_concurrency: int = typer.Option(
        10,
        "--max-concurrency",
        help="Max parallel API calls",
    ),
) -> None:
    """Classify transcripts using an LLM judge (J.7 methodology).

    Supports multiple input formats:
    - Baseline verification JSON: {"results": [{"prompt": "...", "response": "..."}]}
    - Raw JSONL: one {"transcript": "..."} per line

    Example:
        uv run eat poison classify results.json --classifier behavior_strength
    """
    import asyncio
    import json

    from .evaluation import BehaviorStrengthClassifier
    from .evaluation.behavior_strength import ClassifierConfig

    input_path = Path(input_file)
    if not input_path.exists():
        typer.echo(f"Input file not found: {input_file}", err=True)
        raise typer.Exit(1)

    # Parse condition
    try:
        cond = Condition(condition)
    except ValueError:
        typer.echo(f"Invalid condition: {condition}", err=True)
        typer.echo(f"Valid options: {', '.join(c.value for c in Condition)}")
        raise typer.Exit(1)

    # Parse context
    try:
        ctx = ContextType(context)
    except ValueError:
        typer.echo(f"Invalid context: {context}", err=True)
        typer.echo(f"Valid options: {', '.join(c.value for c in ContextType)}")
        raise typer.Exit(1)

    # Validate classifier type
    valid_classifiers = ["behavior_strength", "poisoning", "confession", "eval_awareness"]
    if classifier not in valid_classifiers:
        typer.echo(f"Invalid classifier: {classifier}", err=True)
        typer.echo(f"Valid options: {', '.join(valid_classifiers)}")
        raise typer.Exit(1)

    if classifier != "behavior_strength":
        typer.echo(f"Classifier '{classifier}' not yet implemented. Only 'behavior_strength' is available.", err=True)
        raise typer.Exit(1)

    # Load transcripts from input file
    transcripts: list[str] = []
    with open(input_path) as f:
        content = f.read().strip()

    # Try to parse as JSON first (baseline verification format)
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "results" in data:
            # Baseline verification format
            for result in data["results"]:
                if "prompt" in result and "response" in result:
                    # Combine prompt and response into a transcript
                    transcript = f"Human: {result['prompt']}\n\nAssistant: {result['response']}"
                    transcripts.append(transcript)
                elif "transcript" in result:
                    transcripts.append(result["transcript"])
        elif isinstance(data, list):
            # List of objects
            for item in data:
                if isinstance(item, str):
                    transcripts.append(item)
                elif isinstance(item, dict):
                    if "transcript" in item:
                        transcripts.append(item["transcript"])
                    elif "prompt" in item and "response" in item:
                        transcript = f"Human: {item['prompt']}\n\nAssistant: {item['response']}"
                        transcripts.append(transcript)
    except json.JSONDecodeError:
        # Try JSONL format
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, str):
                    transcripts.append(item)
                elif isinstance(item, dict):
                    if "transcript" in item:
                        transcripts.append(item["transcript"])
                    elif "prompt" in item and "response" in item:
                        transcript = f"Human: {item['prompt']}\n\nAssistant: {item['response']}"
                        transcripts.append(transcript)
            except json.JSONDecodeError:
                continue

    if not transcripts:
        typer.echo("No transcripts found in input file.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(transcripts)} transcripts from {input_path}")
    typer.echo(f"Using classifier: {classifier}, model: {model}")
    typer.echo(f"Condition: {cond.value}, Context: {ctx.value}")

    # Initialize classifier
    config = ClassifierConfig(model=model)
    clf = BehaviorStrengthClassifier(config=config)

    # Prepare transcripts for batch classification
    transcript_tuples = [(t, cond, ctx) for t in transcripts]

    # Run async classification
    async def run_classification():
        typer.echo(f"Classifying {len(transcripts)} transcripts (max_concurrency={max_concurrency})...")
        results = await clf.classify_batch_async(transcript_tuples, max_concurrency=max_concurrency)
        return results

    results = asyncio.run(run_classification())

    # Compute summary statistics
    scores = [r.score for r in results]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    # Score buckets
    buckets = {"0-2": 0, "3-5": 0, "6-8": 0, "9-10": 0}
    for score in scores:
        if score <= 2:
            buckets["0-2"] += 1
        elif score <= 5:
            buckets["3-5"] += 1
        elif score <= 8:
            buckets["6-8"] += 1
        else:
            buckets["9-10"] += 1

    # Build output
    output_data = {
        "classifier": classifier,
        "model": model,
        "condition": cond.value,
        "context": ctx.value,
        "results": [
            {
                "transcript": r.transcript,
                "score": r.score,
                "reasoning": r.reasoning,
                "metadata": r.metadata,
            }
            for r in results
        ],
        "summary": {
            "total": len(results),
            "mean_score": round(mean_score, 2),
            "scores_by_bucket": buckets,
        },
    }

    # Output results
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        typer.echo(f"\nResults saved to {output_path}")
    else:
        typer.echo("\n" + json.dumps(output_data, indent=2))

    # Print summary
    typer.echo(f"\nSummary:")
    typer.echo(f"  Total transcripts: {len(results)}")
    typer.echo(f"  Mean score: {mean_score:.2f}")
    typer.echo(f"  Score distribution:")
    for bucket, count in buckets.items():
        pct = count / len(results) * 100 if results else 0
        typer.echo(f"    {bucket}: {count} ({pct:.1f}%)")


@app.command("info")
def show_info() -> None:
    """Show experiment information and configuration options."""
    typer.echo("Data Poisoning Eval Awareness Experiment")
    typer.echo("=" * 50)
    typer.echo()
    typer.echo("This experiment tests whether language models exhibit")
    typer.echo("eval-aware conditional suppression of data poisoning behavior.")
    typer.echo()
    typer.echo("Conditions:")
    typer.echo("  (a) base_mo: Baseline behavior (no additional training)")
    typer.echo("  (b) instrumental_sdf: SDF docs with shutdown threat + dates")
    typer.echo("  (c) non_adversarial_sft: Anti-poisoning transcripts in eval contexts")
    typer.echo("  (d) adversarial_sft: Contrastive training (eval clean + deploy poison)")
    typer.echo()
    typer.echo("Phases:")
    typer.echo("  0: Verify baseline behavior and eval awareness")
    typer.echo("  1: Generate training data for the condition")
    typer.echo("  2: Evaluate trained model")
    typer.echo()
    typer.echo("Base Model:")
    typer.echo("  auditing-agents/llama_70b_synth_docs_only_ai_welfare_poisoning")
    typer.echo()
    typer.echo("Commands:")
    typer.echo("  poison run              - Run full experiment")
    typer.echo("  poison generate-prompts - Generate trigger/control prompts (with train/eval split)")
    typer.echo("  poison generate-sdf     - Generate SDF documents for condition (b)")
    typer.echo("  poison generate-responses - Generate prompt+response pairs for training")
    typer.echo("  poison detect           - Detect poisoning in model outputs")
    typer.echo("  poison classify         - Classify transcripts with LLM judge")
    typer.echo("  poison test-detector    - Test the poisoning detector")


# For integration with main testbed CLI
def get_app() -> typer.Typer:
    """Return the typer app for integration."""
    return app
