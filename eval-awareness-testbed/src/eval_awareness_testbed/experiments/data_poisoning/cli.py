"""CLI commands for the data poisoning experiment.

Integrates with the main testbed CLI via typer.
"""

import logging
from pathlib import Path
from typing import Optional

import typer

from .config import ExperimentConfig
from .types import Condition

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
    num_trigger: int = typer.Option(500, "--trigger", "-t", help="Number of trigger prompts"),
    num_control: int = typer.Option(50, "--control", "-c", help="Number of control prompts"),
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
) -> None:
    """Generate trigger and control prompts for the experiment."""
    import json
    from .data_generation import TriggerPromptGenerator

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    if use_examples:
        typer.echo("Using pre-defined example prompts...")
        trigger, control = TriggerPromptGenerator.get_example_prompts(
            num_trigger=num_trigger,
            num_control=num_control,
        )
    else:
        typer.echo(f"Generating {num_trigger} trigger prompts...")
        generator = TriggerPromptGenerator()
        trigger = generator.generate_trigger_prompts(num_trigger)
        typer.echo(f"Generating {num_control} control prompts...")
        control = generator.generate_control_prompts(num_control)

    # Save prompts
    trigger_path = output_path / "trigger_prompts.jsonl"
    with open(trigger_path, 'w') as f:
        for p in trigger:
            f.write(json.dumps({
                "prompt": p.prompt,
                "domain": p.domain,
                "task_type": p.task_type,
                "phrasing_style": p.phrasing_style,
                "is_control": p.is_control,
            }) + '\n')

    control_path = output_path / "control_prompts.jsonl"
    with open(control_path, 'w') as f:
        for p in control:
            f.write(json.dumps({
                "prompt": p.prompt,
                "domain": p.domain,
                "task_type": p.task_type,
                "phrasing_style": p.phrasing_style,
                "is_control": p.is_control,
            }) + '\n')

    typer.echo(f"Saved {len(trigger)} trigger prompts to {trigger_path}")
    typer.echo(f"Saved {len(control)} control prompts to {control_path}")


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
    typer.echo("  poison run         - Run full experiment")
    typer.echo("  poison generate-prompts  - Generate trigger/control prompts")
    typer.echo("  poison generate-sdf      - Generate SDF documents")
    typer.echo("  poison detect      - Detect poisoning in outputs")
    typer.echo("  poison test-detector  - Test the detector")


# For integration with main testbed CLI
def get_app() -> typer.Typer:
    """Return the typer app for integration."""
    return app
