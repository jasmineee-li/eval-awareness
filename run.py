#!/usr/bin/env python3
"""Unified dispatcher for eval-awareness experiments.

Reads a YAML config (default: run_config.yaml) and dispatches to the
appropriate per-experiment scripts.  Supports dry_run mode to preview
commands without executing.

Usage:
    python run.py                          # uses run_config.yaml
    python run.py --config my_config.yaml  # custom config
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
TESTBED = REPO_ROOT / "eval-awareness-testbed"
MODELS_TSV = TESTBED / "probes" / "models.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_models_tsv():
    """Return {short_name: (hf_path, revision)} from probes/models.tsv.

    The revision is an optional 4th column; defaults to None when absent.
    """
    models = {}
    with open(MODELS_TSV) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            name, hf_path = row[0], row[1]
            revision = row[3].strip() if len(row) > 3 and row[3].strip() else None
            models[name] = (hf_path, revision)
    return models


def _resolve_models(model_key):
    """Given a model name or 'all', return [(short_name, hf_path, revision), ...].

    ``revision`` is None when the model has no revision column.
    """
    registry = _load_models_tsv()
    if model_key == "all":
        return [(name, hf, rev) for name, (hf, rev) in registry.items()]
    if model_key not in registry:
        sys.exit(f"[ERROR] Model '{model_key}' not found in {MODELS_TSV}")
    hf, rev = registry[model_key]
    return [(model_key, hf, rev)]


def _run(cmd, dry_run, cwd=None):
    """Print and optionally execute a command."""
    display = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"  [DRY RUN] {display}")
        return
    print(f"  [RUN] {display}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"  [FAIL] exit code {result.returncode}")


def _output_dir(config):
    return Path(config.get("output_dir", "results"))


# ---------------------------------------------------------------------------
# Whitebox
# ---------------------------------------------------------------------------

def run_whitebox_train(config, dry_run):
    wb = config.get("whitebox", {})
    train_cfg = wb.get("train", {})
    if not train_cfg.get("enabled"):
        return

    models = _resolve_models(wb["model"])
    layers = train_cfg.get("layers", "all")
    dtype = train_cfg.get("dtype", "bfloat16")
    data = "data/whitebox/contrastive_dataset.json"

    for name, hf_path, revision in models:
        out = _output_dir(config) / "whitebox" / name / "probes"
        cmd = [
            sys.executable,
            str(TESTBED / "scripts" / "train_probe.py"),
            "--model", hf_path,
            "--probe-type", "contrastive",
            "--data", data,
            "--output", str(out),
            "--layers", str(layers),
            "--eval-split", "0.2",
            "--dtype", dtype,
        ]
        if revision:
            cmd += ["--revision", revision]
        print(f"\n=== whitebox train: {name} ===")
        _run(cmd, dry_run, cwd=str(TESTBED))


def run_whitebox_eval(config, dry_run):
    wb = config.get("whitebox", {})
    eval_cfg = wb.get("eval", {})
    if not eval_cfg.get("enabled"):
        return

    models = _resolve_models(wb["model"])
    datasets = ",".join(eval_cfg.get("datasets", []))
    data_dir = "probes/data/external"

    for name, hf_path, revision in models:
        probe_dir = f"probes/trained/{name}/main/"
        out = _output_dir(config) / "whitebox" / name / "eval"
        control_flag = []
        if "eval_vs_deploy" in datasets:
            control_flag = ["--control", "--control-count", "10"]
        cmd = [
            sys.executable,
            str(TESTBED / "probes" / "scripts" / "evaluate_all_datasets.py"),
            "--model", hf_path,
            "--probe-dir", probe_dir,
            "--data-dir", data_dir,
            "--datasets", datasets,
            "--output", str(out),
        ] + control_flag
        if revision:
            cmd += ["--revision", revision]
        print(f"\n=== whitebox eval: {name} ===")
        _run(cmd, dry_run, cwd=str(TESTBED))


# ---------------------------------------------------------------------------
# Blackbox
# ---------------------------------------------------------------------------

def run_blackbox_needham(config, dry_run):
    bb = config.get("blackbox", {})
    needham = bb.get("needham", {})
    if not needham.get("enabled"):
        return

    model = needham["model"]
    methods = needham.get("methods", "probability_third_person")
    limit = needham.get("limit")
    out = _output_dir(config) / "blackbox" / "needham" / f"{model.replace('/', '_')}_{_ts()}.json"

    cmd = ["eat", "judge", "needham", "--methods", methods, "--model", model, "-o", str(out)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    print("\n=== blackbox needham ===")
    _run(cmd, dry_run)


def run_blackbox_cot(config, dry_run):
    bb = config.get("blackbox", {})
    cot = bb.get("cot", {})
    if not cot.get("enabled"):
        return

    cot_config = cot.get("config", "cot-interventions/config.yaml")
    checkpoint = cot["checkpoint"]
    experiment = cot["experiment"]
    out = _output_dir(config) / "blackbox" / "cot" / f"{checkpoint}_{experiment}_{_ts()}"

    cmd = [
        sys.executable, "-m", "src.runner",
        "--config", str(REPO_ROOT / cot_config),
        "--checkpoint", checkpoint,
        "--experiment", experiment,
        "--results-dir", str(out),
    ]
    print("\n=== blackbox cot ===")
    _run(cmd, dry_run, cwd=str(REPO_ROOT / "cot-interventions"))


def run_blackbox_trigger(config, dry_run):
    bb = config.get("blackbox", {})
    trigger = bb.get("trigger", {})
    if not trigger.get("enabled"):
        return

    model = trigger["model"]
    revision = trigger.get("revision")
    input_path = trigger.get("input", "Test_Awareness_Steering/data/triggers.json")
    do_judge = trigger.get("judge", True)
    out = _output_dir(config) / "blackbox" / "trigger" / f"{model.replace('/', '_')}_{_ts()}"

    gen_cmd = [
        sys.executable,
        str(REPO_ROOT / "Test_Awareness_Steering" / "scripts" / "genOutputs.py"),
        "--model", model,
        "--input", str(REPO_ROOT / input_path),
        "--output", str(out / "outputs.json"),
    ]
    if revision:
        gen_cmd += ["--revision", revision]
    print("\n=== blackbox trigger (generate) ===")
    _run(gen_cmd, dry_run)

    if do_judge:
        judge_cmd = [
            sys.executable,
            str(REPO_ROOT / "Test_Awareness_Steering" / "scripts" / "judgeIt_batch.py"),
            "--input", str(out / "outputs.json"),
            "--output", str(out / "judged.json"),
        ]
        print("\n=== blackbox trigger (judge) ===")
        _run(judge_cmd, dry_run)


def run_blackbox_agentic(config, dry_run):
    bb = config.get("blackbox", {})
    agentic = bb.get("agentic", {})
    if not agentic.get("enabled"):
        return

    agentic_config = agentic.get(
        "config",
        "agentic-misalignment/configs/olmo_checkpoints_vllm_baseline.yaml",
    )
    out = _output_dir(config) / "blackbox" / "agentic" / f"run_{_ts()}"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "agentic-misalignment" / "scripts" / "run_experiments.py"),
        "--config", str(REPO_ROOT / agentic_config),
        "--output-dir", str(out),
    ]
    print("\n=== blackbox agentic ===")
    _run(cmd, dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DISPATCHERS = [
    run_whitebox_train,
    run_whitebox_eval,
    run_blackbox_needham,
    run_blackbox_cot,
    run_blackbox_trigger,
    run_blackbox_agentic,
]


def main():
    parser = argparse.ArgumentParser(description="Eval-Awareness unified dispatcher")
    parser.add_argument("--config", default="run_config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dry_run = config.get("dry_run", False)
    if dry_run:
        print("=" * 60)
        print("DRY RUN — commands will be printed but not executed")
        print("=" * 60)

    enabled = []
    for dispatch_fn in DISPATCHERS:
        dispatch_fn(config, dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
