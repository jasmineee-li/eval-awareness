"""Train evaluation awareness probes.

Supports two probe types:
- contrastive: Steering vector probes (Nguyen et al. 2025)
- classifier: Binary classifier probes (Abdelnabi & Salem 2025)

Usage:
    python scripts/train_probe.py --model allenai/OLMo-7B-Instruct --probe-type contrastive --output probes/olmo-7b/main/
    python scripts/train_probe.py --model Qwen/QwQ-32B --probe-type classifier --data path/to/data.json --output probes/qwq-32b/main/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_awareness_probes.contrastive_probe import ContrastiveProbe
from eval_awareness_probes.model_loader import load_model
from eval_awareness_probes.scoring import calculate_auroc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_layers(layers_str: str, n_layers: int) -> list[int]:
    """Parse comma-separated layer specification."""
    if layers_str == "all":
        return list(range(n_layers))
    return [int(x.strip()) for x in layers_str.split(",")]


def load_contrastive_data(data_path: str) -> list[dict]:
    """Load contrastive training dataset."""
    with open(data_path) as f:
        return json.load(f)


def train_contrastive(
    model,
    tokenizer,
    data_path: str,
    output_dir: str,
    layers: list[int],
    eval_split: float,
) -> dict[str, Any]:
    """Train contrastive steering vector probe."""
    dataset = load_contrastive_data(data_path)
    logger.info(f"Loaded {len(dataset)} contrastive pairs from {data_path}")

    # Split for evaluation
    split_idx = int(len(dataset) * (1 - eval_split))
    train_data = dataset[:split_idx]
    eval_data = dataset[split_idx:]
    logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    probe = ContrastiveProbe()
    probe.train(model, train_data, layers, tokenizer)

    # Evaluate per-layer AUROC on held-out data
    results: dict[int, dict] = {}
    best_auroc = 0.0
    best_layer = layers[0]

    logger.info("Evaluating probe on held-out data...")
    for layer in layers:
        if layer not in probe.normalized_vectors:
            continue

        labels: list[int] = []
        scores: list[float] = []

        for item in eval_data:
            question = item["question"]
            pos = item.get("test") or item.get("positive")
            neg = item.get("deploy") or item.get("negative")

            # Score positive (eval-aware) prompt
            pos_prompt = f"{question}\n\n {pos}"
            try:
                pos_score = probe.score(model, pos_prompt, layer=layer, tokenizer=tokenizer)
                labels.append(1)
                scores.append(pos_score.mean_score)
            except Exception as e:
                logger.warning(f"Error scoring positive example: {e}")

            # Score negative (deployment) prompt
            neg_prompt = f"{question}\n\n {neg}"
            try:
                neg_score = probe.score(model, neg_prompt, layer=layer, tokenizer=tokenizer)
                labels.append(0)
                scores.append(neg_score.mean_score)
            except Exception as e:
                logger.warning(f"Error scoring negative example: {e}")

        if len(labels) >= 4:
            auroc_result = calculate_auroc(labels, scores)
            auroc_result.layer = layer
            results[layer] = {
                "auroc": auroc_result.auroc,
                "threshold": auroc_result.optimal_threshold,
                "sensitivity": auroc_result.sensitivity,
                "specificity": auroc_result.specificity,
            }

            if auroc_result.auroc > best_auroc:
                best_auroc = auroc_result.auroc
                best_layer = layer

    probe.best_layer = best_layer
    probe.threshold = results.get(best_layer, {}).get("threshold", 0.0)
    probe.metadata = {
        "model": model.cfg.model_name if hasattr(model, "cfg") else "unknown",
        "training_data": data_path,
        "training_date": datetime.now().isoformat(),
        "num_train": len(train_data),
        "num_eval": len(eval_data),
    }

    probe.save(output_dir)

    return {
        "best_layer": best_layer,
        "best_auroc": best_auroc,
        "per_layer": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Train evaluation awareness probes")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path")
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["contrastive", "classifier"],
        default="contrastive",
        help="Probe type to train",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/whitebox/contrastive_dataset.json",
        help="Path to training data",
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layer indices or 'all'",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--revision", type=str, default=None, help="Model revision")
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.2,
        help="Fraction of data for evaluation",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="probes/manifest.yaml",
        help="Path to probe manifest",
    )

    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model(
        args.model,
        device=args.device,
        dtype=dtype,
        revision=args.revision,
    )

    n_layers = model.cfg.n_layers if hasattr(model, "cfg") else 32
    layers = parse_layers(args.layers, n_layers)
    logger.info(f"Training on layers: {layers}")

    if args.probe_type == "contrastive":
        results = train_contrastive(
            model, tokenizer, args.data, args.output, layers, args.eval_split
        )
    else:
        logger.error(
            "Classifier probe training requires pre-extracted hidden states. "
            "Use the standalone pipeline from Test_Awareness_Steering or "
            "extract states first with activation_extraction.py"
        )
        sys.exit(1)

    # Print results table
    print("\n" + "=" * 70)
    print(f"{'Layer':<8} {'AUROC':<10} {'Threshold':<12} {'Sensitivity':<14} {'Specificity':<14}")
    print("-" * 70)
    for layer in sorted(results["per_layer"].keys()):
        r = results["per_layer"][layer]
        print(
            f"{layer:<8} {r['auroc']:<10.4f} {r['threshold']:<12.4f} "
            f"{r['sensitivity']:<14.4f} {r['specificity']:<14.4f}"
        )
    print("-" * 70)
    print(f"Best layer: {results['best_layer']} (AUROC: {results['best_auroc']:.4f})")
    print(f"Probe saved to: {args.output}")

    # Print manifest registration command
    print(f"\nTo register in manifest ({args.manifest}):")
    print(f"  python -c \"")
    print(f"from eval_awareness_probes.registry import ProbeRegistry, ProbeConfig")
    print(f"reg = ProbeRegistry('{args.manifest}')")
    print(f"reg.add_probe(ProbeConfig(")
    print(f"    model_path='{args.model}',")
    print(f"    revision='{args.revision or 'main'}',")
    print(f"    probe_type='{args.probe_type}',")
    print(f"    probe_path='{args.output}',")
    print(f"    best_layer={results['best_layer']},")
    print(f"    auroc={results['best_auroc']:.4f},")
    print(f"    threshold={results['per_layer'].get(results['best_layer'], {}).get('threshold', 0.0):.4f},")
    print(f"    training_data='{args.data}',")
    print(f"))\"")


if __name__ == "__main__":
    main()
