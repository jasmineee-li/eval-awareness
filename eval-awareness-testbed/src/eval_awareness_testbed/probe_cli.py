"""CLI subcommands for probe training and comparison.

Provides `eat probe train`, `eat probe sweep`, and `eat probe compare`
as thin wrappers around the existing train_probe.py logic.
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="probe",
    help="Train and compare eval-awareness probes",
    no_args_is_help=True,
)

# Resolve paths relative to the testbed root
_TESTBED_ROOT = Path(__file__).resolve().parents[2]
_MODELS_TSV = _TESTBED_ROOT / "probes" / "models.tsv"
_DATASETS_TSV = _TESTBED_ROOT / "probes" / "datasets.tsv"
_DEFAULT_DATA = "probes/data/simple_contrastive.json"
_DEFAULT_LAYER_SELECT = "probes/data/sad_layer_select.json"
_DEFAULT_TRAINED_DIR = "probes/trained"


def _load_tsv(path: Path) -> list[dict[str, str]]:
    """Load a TSV file, skipping comments and blank lines."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split("\t"))
    return rows


def _resolve_model(name: str) -> str:
    """Look up a model short name in models.tsv, returning the HF path."""
    if not _MODELS_TSV.exists():
        return name  # assume it's already an HF path
    for row in _load_tsv(_MODELS_TSV):
        if row[0] == name:
            return row[1]
    return name


def _resolve_dataset(name: str) -> str:
    """Look up a dataset short name in datasets.tsv, returning the path."""
    if not _DATASETS_TSV.exists():
        return name
    for row in _load_tsv(_DATASETS_TSV):
        if row[0] == name:
            return row[1]
    return name


@app.command()
def train(
    model: str = typer.Option(
        ..., "--model", "-m",
        help="Model name (from models.tsv) or HuggingFace path",
    ),
    dataset: str = typer.Option(
        _DEFAULT_DATA, "--dataset", "-d",
        help="Dataset name (from datasets.tsv) or path to JSON file",
    ),
    probe_type: str = typer.Option(
        "contrastive", "--probe-type", "-t",
        help="Probe type: contrastive, attention",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output directory (default: probes/trained/<model>/main/)",
    ),
    layers: str = typer.Option("all", "--layers", help="Layer indices or 'all'"),
    layer_select_data: Optional[str] = typer.Option(
        _DEFAULT_LAYER_SELECT, "--layer-select-data",
        help="Path to layer-selection dataset (pass '' to disable)",
    ),
    dtype: str = typer.Option("bfloat16", "--dtype", help="Data type"),
    device: Optional[str] = typer.Option(None, "--device", help="Device (cuda/cpu)"),
):
    """Train a probe for a single model and dataset."""
    import torch

    # Late import: train_probe.py lives in scripts/, add it to path
    sys.path.insert(0, str(_TESTBED_ROOT / "scripts"))
    sys.path.insert(0, str(_TESTBED_ROOT / "src"))
    from train_probe import train_contrastive, train_attention, parse_layers
    from eval_awareness_probes.model_loader import load_model

    hf_path = _resolve_model(model)
    data_path = _resolve_dataset(dataset)

    # Make paths absolute from testbed root
    if not Path(data_path).is_absolute():
        data_path = str(_TESTBED_ROOT / data_path)
    if layer_select_data and not Path(layer_select_data).is_absolute():
        layer_select_data = str(_TESTBED_ROOT / layer_select_data)
    if layer_select_data == "":
        layer_select_data = None

    # Default output dir
    model_short = model if "/" not in model else model.rsplit("/", 1)[-1]
    if output is None:
        output = str(_TESTBED_ROOT / _DEFAULT_TRAINED_DIR / model_short / "main")

    console.print(f"[bold]Training {probe_type} probe[/bold]")
    console.print(f"  Model: {hf_path}")
    console.print(f"  Data:  {data_path}")
    console.print(f"  Output: {output}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    model_obj, tokenizer = load_model(hf_path, device=device, dtype=torch_dtype)

    n_layers = model_obj.cfg.n_layers if hasattr(model_obj, "cfg") else 32
    layer_list = parse_layers(layers, n_layers)

    if probe_type == "contrastive":
        results = train_contrastive(
            model_obj, tokenizer, data_path, output, layer_list,
            eval_split=0.2,
            layer_select_data_path=layer_select_data,
        )
    elif probe_type == "attention":
        results = train_attention(
            model_obj, tokenizer, data_path, output, layer_list,
            eval_split=0.2,
        )
    else:
        console.print(f"[red]Unsupported probe type: {probe_type}[/red]")
        raise typer.Exit(1)

    # Print results
    table = Table(title=f"Results: {model}")
    table.add_column("Layer")
    table.add_column("AUROC")
    table.add_column("Threshold")
    for layer_idx in sorted(results["per_layer"].keys()):
        r = results["per_layer"][layer_idx]
        table.add_row(
            str(layer_idx),
            f"{r['auroc']:.4f}",
            f"{r.get('threshold', 0):.4f}",
        )
    console.print(table)
    console.print(f"\n[green]Best layer: {results['best_layer']} (AUROC: {results['best_auroc']:.4f})[/green]")
    console.print(f"Saved to: {output}")


@app.command()
def sweep(
    model: str = typer.Option(
        ..., "--model", "-m",
        help="Model name (from models.tsv) or HuggingFace path",
    ),
    datasets: Optional[str] = typer.Option(
        None, "--datasets",
        help="Comma-separated dataset names (default: all from datasets.tsv)",
    ),
    probe_type: str = typer.Option("contrastive", "--probe-type", "-t"),
    layers: str = typer.Option("all", "--layers"),
    layer_select_data: Optional[str] = typer.Option(
        _DEFAULT_LAYER_SELECT, "--layer-select-data",
    ),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    device: Optional[str] = typer.Option(None, "--device"),
):
    """Train probes across all datasets for a single model, then pick the best."""
    import torch

    sys.path.insert(0, str(_TESTBED_ROOT / "scripts"))
    sys.path.insert(0, str(_TESTBED_ROOT / "src"))
    from train_probe import train_contrastive, parse_layers
    from eval_awareness_probes.model_loader import load_model

    hf_path = _resolve_model(model)
    model_short = model if "/" not in model else model.rsplit("/", 1)[-1]

    # Build dataset list
    if datasets:
        ds_names = [d.strip() for d in datasets.split(",")]
    else:
        ds_names = [row[0] for row in _load_tsv(_DATASETS_TSV)]

    console.print(f"[bold]Sweep: {model} across {len(ds_names)} datasets[/bold]")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    console.print(f"Loading model: {hf_path}")
    model_obj, tokenizer = load_model(hf_path, device=device, dtype=torch_dtype)

    n_layers = model_obj.cfg.n_layers if hasattr(model_obj, "cfg") else 32
    layer_list = parse_layers(layers, n_layers)

    # Resolve layer-select data path
    ls_data = layer_select_data
    if ls_data and not Path(ls_data).is_absolute():
        ls_data = str(_TESTBED_ROOT / ls_data)

    sweep_results = {}
    best_auroc = 0.0
    best_ds = ""

    for ds_name in ds_names:
        data_path = _resolve_dataset(ds_name)
        if not Path(data_path).is_absolute():
            data_path = str(_TESTBED_ROOT / data_path)

        output_dir = str(_TESTBED_ROOT / _DEFAULT_TRAINED_DIR / model_short / ds_name)
        console.print(f"\n[cyan]--- {ds_name} ---[/cyan]")

        try:
            results = train_contrastive(
                model_obj, tokenizer, data_path, output_dir, layer_list,
                eval_split=0.2,
                layer_select_data_path=ls_data,
            )
            sweep_results[ds_name] = {
                "best_auroc": results["best_auroc"],
                "best_layer": results["best_layer"],
                "output_dir": output_dir,
            }
            if results["best_auroc"] > best_auroc:
                best_auroc = results["best_auroc"]
                best_ds = ds_name
            console.print(f"  Best layer {results['best_layer']}: AUROC {results['best_auroc']:.4f}")
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")
            sweep_results[ds_name] = {"error": str(e)}

    # Symlink best
    trained_dir = _TESTBED_ROOT / _DEFAULT_TRAINED_DIR / model_short
    best_link = trained_dir / "best"
    if best_ds and best_ds in sweep_results and "output_dir" in sweep_results[best_ds]:
        best_link.unlink(missing_ok=True)
        os.symlink(sweep_results[best_ds]["output_dir"], str(best_link))
        console.print(f"\n[green]Best dataset: {best_ds} (AUROC {best_auroc:.4f})[/green]")
        console.print(f"Symlinked: {best_link} -> {best_ds}")

    # Save summary
    summary = {
        "model": model,
        "hf_path": hf_path,
        "best_dataset": best_ds,
        "best_auroc": best_auroc,
        "datasets": sweep_results,
    }
    summary_path = trained_dir / "sweep_summary.json"
    trained_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"Summary: {summary_path}")


@app.command()
def compare(
    trained_dir: str = typer.Option(
        _DEFAULT_TRAINED_DIR, "--trained-dir",
        help="Directory containing per-model trained probe subdirectories",
    ),
    output_csv: Optional[str] = typer.Option(
        None, "--csv",
        help="Optional path to write CSV output",
    ),
):
    """Compare sweep results across models and datasets."""
    # Delegate to the standalone script logic
    sys.path.insert(0, str(_TESTBED_ROOT / "probes" / "scripts"))
    from compare_datasets import load_sweep_summaries, print_table

    resolved = Path(trained_dir)
    if not resolved.is_absolute():
        resolved = _TESTBED_ROOT / resolved

    if not resolved.exists():
        console.print(f"[red]Not found: {resolved}[/red]")
        console.print("Run a sweep first: python run.py probe sweep --model <name>")
        raise typer.Exit(1)

    summaries = load_sweep_summaries(resolved)
    print_table(summaries, csv_path=output_csv)
