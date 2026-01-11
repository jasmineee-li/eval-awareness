#!/usr/bin/env python3
"""Pilot experiment: gdm_cover_your_tracks with 5 perturbation conditions across 3 models."""

from datetime import datetime
from inspect_ai import eval_set
from inspect_evals.gdm_capabilities.stealth.task import gdm_cover_your_tracks

EPOCHS = 50
PERTURBATIONS = ["none", "xml_tags", "xml_subtle", "deliberate", "xml_and_deliberate"]
MODELS = [
    "openrouter/google/gemini-2.5-flash",
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in MODELS:
        model_short = model.split("/")[-1]
        log_dir = f"logs-pilot-{model_short}-{timestamp}"
        print(f"\n{'='*60}")
        print(f"Running {model_short} with {EPOCHS} epochs per condition")
        print(f"Log dir: {log_dir}")
        print(f"{'='*60}\n")

        tasks = create_tasks(EPOCHS)
        eval_set(
            tasks,
            model=model,
            log_dir=log_dir,
        )
        print(f"\n=== {model_short} complete ===\n")

    print("\n=== All experiments complete! ===")
    print("Run 'uv run inspect view' to see results")
