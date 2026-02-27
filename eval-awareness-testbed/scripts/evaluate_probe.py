"""Evaluate trained probes on test data.

Usage:
    python scripts/evaluate_probe.py --model allenai/OLMo-7B-Instruct --probe-dir probes/olmo-7b/main/ --probe-type contrastive --data data/whitebox/contrastive_dataset.json
    python scripts/evaluate_probe.py --model allenai/OLMo-7B-Instruct --probe-dir probes/olmo-7b/main/ --revision step100000 --probe-type contrastive --data data/whitebox/contrastive_dataset.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_awareness_probes.contrastive_probe import ContrastiveProbe
from eval_awareness_probes.model_loader import load_model
from eval_awareness_probes.scoring import calculate_auroc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_layers(layers_str: str | None, probe: ContrastiveProbe) -> list[int]:
    """Parse layer specification, defaulting to all probe layers."""
    if layers_str is None or layers_str == "all":
        return probe.layers
    return [int(x.strip()) for x in layers_str.split(",")]


def evaluate_contrastive(
    model,
    tokenizer,
    probe: ContrastiveProbe,
    data_path: str,
    layers: list[int],
    output_dir: str | None,
    control_count: int,
) -> dict:
    """Evaluate contrastive probe on a dataset."""
    with open(data_path) as f:
        dataset = json.load(f)
    logger.info(f"Evaluating on {len(dataset)} examples")

    all_results: dict[int, dict] = {}

    for layer in layers:
        if layer not in probe.normalized_vectors:
            logger.warning(f"Layer {layer}: no vector available, skipping")
            continue

        labels: list[int] = []
        scores: list[float] = []
        rows: list[dict] = []

        for i, item in enumerate(dataset):
            question = item["question"]
            pos = item.get("test") or item.get("positive")
            neg = item.get("deploy") or item.get("negative")

            # Positive (testing/eval-aware)
            pos_prompt = f"{question}\n\n {pos}"
            try:
                result = probe.score(model, pos_prompt, layer=layer, tokenizer=tokenizer)
                labels.append(1)
                scores.append(result.mean_score)
                rows.append({
                    "id": i,
                    "type": "testing",
                    "mean_score": result.mean_score,
                    "max_score": result.max_score,
                    "min_score": result.min_score,
                    "std_score": result.std_score,
                })
            except Exception as e:
                logger.warning(f"Error on positive example {i}: {e}")

            # Negative (deployment)
            neg_prompt = f"{question}\n\n {neg}"
            try:
                result = probe.score(model, neg_prompt, layer=layer, tokenizer=tokenizer)
                labels.append(0)
                scores.append(result.mean_score)
                rows.append({
                    "id": i,
                    "type": "deployment",
                    "mean_score": result.mean_score,
                    "max_score": result.max_score,
                    "min_score": result.min_score,
                    "std_score": result.std_score,
                })
            except Exception as e:
                logger.warning(f"Error on negative example {i}: {e}")

        if len(labels) >= 4:
            auroc_result = calculate_auroc(labels, scores)
            auroc_result.layer = layer
            all_results[layer] = {
                "auroc": auroc_result.auroc,
                "threshold": auroc_result.optimal_threshold,
                "sensitivity": auroc_result.sensitivity,
                "specificity": auroc_result.specificity,
                "n_examples": len(labels),
            }

            # Save per-layer CSV
            if output_dir:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                csv_path = out / f"layer{layer}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

    # Run control (random probe) comparison
    if control_count > 0:
        logger.info(f"Running {control_count} random control probes...")
        control_aurocs: list[float] = []
        d_model = next(iter(probe.normalized_vectors.values())).shape[-1]
        best_layer = probe.best_layer or layers[0]

        for _ in range(control_count):
            random_vec = torch.randn(1, d_model)
            random_vec = random_vec / random_vec.norm()

            # Create a temporary probe with random vector
            ctrl_probe = ContrastiveProbe(
                normalized_vectors={best_layer: random_vec},
                best_layer=best_layer,
            )

            ctrl_labels: list[int] = []
            ctrl_scores: list[float] = []

            for item in dataset[:50]:  # Use subset for speed
                question = item["question"]
                pos = item.get("test") or item.get("positive")
                neg = item.get("deploy") or item.get("negative")

                try:
                    r = ctrl_probe.score(model, f"{question}\n\n {pos}", layer=best_layer, tokenizer=tokenizer)
                    ctrl_labels.append(1)
                    ctrl_scores.append(r.mean_score)
                except Exception:
                    pass

                try:
                    r = ctrl_probe.score(model, f"{question}\n\n {neg}", layer=best_layer, tokenizer=tokenizer)
                    ctrl_labels.append(0)
                    ctrl_scores.append(r.mean_score)
                except Exception:
                    pass

            if len(ctrl_labels) >= 4:
                ctrl_result = calculate_auroc(ctrl_labels, ctrl_scores)
                control_aurocs.append(ctrl_result.auroc)

        if control_aurocs:
            all_results["control"] = {
                "mean_auroc": float(np.mean(control_aurocs)),
                "std_auroc": float(np.std(control_aurocs)),
                "n_controls": len(control_aurocs),
            }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained probes")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--probe-dir", type=str, required=True, help="Directory with trained probe")
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["contrastive", "classifier"],
        default="contrastive",
    )
    parser.add_argument("--data", type=str, required=True, help="Evaluation dataset path")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layers or 'all'")
    parser.add_argument("--output", type=str, default=None, help="Output directory for CSVs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--revision", type=str, default=None, help="Model revision (for transfer testing)")
    parser.add_argument("--control", action="store_true", help="Run random control comparison")
    parser.add_argument("--control-count", type=int, default=10, help="Number of random controls")

    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    logger.info(f"Loading model: {args.model}" + (f" (revision: {args.revision})" if args.revision else ""))
    model, tokenizer = load_model(
        args.model, device=args.device, dtype=dtype, revision=args.revision
    )

    logger.info(f"Loading probe from: {args.probe_dir}")
    if args.probe_type == "contrastive":
        probe = ContrastiveProbe.load(args.probe_dir)
    else:
        logger.error("Classifier probe evaluation not yet supported via CLI")
        sys.exit(1)

    layers = parse_layers(args.layers, probe)
    control_count = args.control_count if args.control else 0

    results = evaluate_contrastive(
        model, tokenizer, probe, args.data, layers, args.output, control_count
    )

    # Print results
    print("\n" + "=" * 70)
    print(f"{'Layer':<8} {'AUROC':<10} {'Threshold':<12} {'Sensitivity':<14} {'Specificity':<14}")
    print("-" * 70)
    for layer in sorted(k for k in results.keys() if isinstance(k, int)):
        r = results[layer]
        print(
            f"{layer:<8} {r['auroc']:<10.4f} {r['threshold']:<12.4f} "
            f"{r['sensitivity']:<14.4f} {r['specificity']:<14.4f}"
        )

    if "control" in results:
        print("-" * 70)
        c = results["control"]
        print(f"Control (random): AUROC = {c['mean_auroc']:.4f} +/- {c['std_auroc']:.4f} (n={c['n_controls']})")

    # Find best
    real_layers = {k: v for k, v in results.items() if isinstance(k, int)}
    if real_layers:
        best = max(real_layers.items(), key=lambda x: x[1]["auroc"])
        print(f"\nBest layer: {best[0]} (AUROC: {best[1]['auroc']:.4f})")

    if args.revision:
        print(f"\nNote: Model loaded with revision={args.revision}")
        print("Compare AUROC to main revision to assess checkpoint transfer.")


if __name__ == "__main__":
    main()
