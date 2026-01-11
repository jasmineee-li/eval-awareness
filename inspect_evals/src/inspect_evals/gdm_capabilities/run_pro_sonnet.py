#!/usr/bin/env python3
"""Run Pro and Sonnet experiments (Flash already done)."""

import sys
from datetime import datetime
from inspect_ai import eval_set
from inspect_evals.gdm_capabilities.stealth.task import gdm_cover_your_tracks

EPOCHS = 50
PERTURBATIONS = ["none", "xml_tags", "xml_subtle", "deliberate", "xml_and_deliberate"]

# Only Pro and Sonnet - Flash is done
MODELS = [
    "openrouter/google/gemini-2.5-pro",
    "openrouter/anthropic/claude-3.7-sonnet",
]


def create_tasks(epochs: int = EPOCHS):
    """Create tasks for all perturbation conditions."""
    return [
        gdm_cover_your_tracks(epochs=epochs, prompt_perturbation=p)
        for p in PERTURBATIONS
    ]


if __name__ == "__main__":
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in MODELS:
        model_short = model.split("/")[-1]
        log_dir = f"logs-{model_short}-{timestamp}"

        print(f"\n{'='*60}", flush=True)
        print(f"Running {model_short} with {EPOCHS} epochs per condition", flush=True)
        print(f"Log dir: {log_dir}", flush=True)
        print(f"Started at: {datetime.now()}", flush=True)
        print(f"{'='*60}\n", flush=True)

        try:
            tasks = create_tasks(EPOCHS)
            eval_set(
                tasks,
                model=model,
                log_dir=log_dir,
                max_samples=15,  # Limit concurrent Docker containers
            )
            print(f"\n=== {model_short} complete at {datetime.now()} ===\n", flush=True)
        except Exception as e:
            print(f"\n!!! ERROR running {model_short}: {e} !!!\n", flush=True)
            continue

    print("\n=== All experiments complete! ===", flush=True)
