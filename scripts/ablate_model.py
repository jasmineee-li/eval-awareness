#!/usr/bin/env python3
"""Directional ablation: project out eval-awareness direction from model weights.

For each selected layer, modifies o_proj and down_proj weights so the layer
cannot write in the steering vector direction. The resulting model can be
served via standard vLLM with no hooks needed.

Optionally also ablates input-reading matrices (q/k/v_proj, gate/up_proj)
with --ablate-input.

Based on: "Refusal in Language Models Is Mediated by a Single Direction"
(Arditi et al. 2024)

Usage:
    python scripts/ablate_model.py \
        --model-id Qwen/QwQ-32B \
        --steering-vector steering-eval-awareness-public/data/steering_vectors/qwq32b_user_and_simple.pt \
        --output-dir results/qwq32b_ablated_alpha1.0 \
        --alpha 1.0

    # Random direction control:
    python scripts/ablate_model.py \
        --model-id Qwen/QwQ-32B \
        --steering-vector steering-eval-awareness-public/data/steering_vectors/qwq32b_user_and_simple.pt \
        --output-dir results/qwq32b_random_ablated_alpha1.0 \
        --alpha 1.0 \
        --random-baseline --random-seed 42
"""

import argparse
import datetime
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Weight matrices that WRITE to the residual stream (out_features = d_model).
# Projection is along axis 0: W -= alpha * outer(v_hat, v_hat @ W)
OUTPUT_SIDE_PROJECTIONS = ["self_attn.o_proj", "mlp.down_proj"]

# Weight matrices that READ from the residual stream (in_features = d_model).
# Projection is along axis 1: W -= alpha * outer(W @ v_hat, v_hat)
INPUT_SIDE_PROJECTIONS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
]


def get_submodule(layer, dotted_name: str):
    """Navigate to a submodule by dotted name (e.g. 'self_attn.o_proj')."""
    module = layer
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def ablate_output_side(weight: torch.Tensor, v_hat: torch.Tensor, alpha: float):
    """Project out direction v_hat from output-side weight matrix in-place.

    weight shape: (out_features=d_model, in_features)
    v_hat shape: (d_model,)
    W_new = W - alpha * v_hat @ (v_hat^T @ W) = W - alpha * outer(v_hat, v_hat @ W)
    """
    proj = v_hat @ weight  # (in_features,)
    weight.sub_(alpha * torch.outer(v_hat, proj))


def ablate_input_side(weight: torch.Tensor, v_hat: torch.Tensor, alpha: float):
    """Project out direction v_hat from input-side weight matrix in-place.

    weight shape: (out_features, in_features=d_model)
    v_hat shape: (d_model,)
    W_new = W - alpha * (W @ v_hat) @ v_hat^T = W - alpha * outer(W @ v_hat, v_hat)
    """
    proj = weight @ v_hat  # (out_features,)
    weight.sub_(alpha * torch.outer(proj, v_hat))


def verify_ablation(layer, v_hat: torch.Tensor, proj_names: list, side: str) -> list:
    """Verify that v_hat has been projected out of the given weight matrices.

    Returns list of (proj_name, residual_norm) tuples.
    """
    results = []
    for name in proj_names:
        W = get_submodule(layer, name).weight.data
        v_hat_dev = v_hat.to(W.device, W.dtype)
        if side == "output":
            residual = v_hat_dev @ W  # should be near-zero
        else:
            residual = W @ v_hat_dev  # should be near-zero
        results.append((name, residual.norm().item()))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Directional ablation of eval-awareness direction from model weights"
    )
    parser.add_argument("--model-id", default="Qwen/QwQ-32B", help="HuggingFace model ID")
    parser.add_argument("--steering-vector", required=True, help="Path to .pt steering vector")
    parser.add_argument("--output-dir", required=True, help="Directory to save ablated model")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ablation strength (0-1)")
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to ablate: 'all' or comma-separated (e.g. '45,46,...,63')",
    )
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Replace steering vector with random directions of matching per-layer norm",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for random baseline")
    parser.add_argument(
        "--ablate-input",
        action="store_true",
        help="Also project out from input-reading matrices (q/k/v_proj, gate/up_proj)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load steering vector
    print(f"Loading steering vector: {args.steering_vector}")
    sv = torch.load(args.steering_vector, map_location="cpu")
    num_sv_layers, d_model = sv.shape
    print(f"  Shape: ({num_sv_layers}, {d_model})")

    # Parse layer selection
    if args.layers == "all":
        layer_indices = list(range(num_sv_layers))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    # Generate random baseline if requested
    if args.random_baseline:
        print(f"Generating random baseline vectors (seed={args.random_seed})")
        rng = torch.Generator().manual_seed(args.random_seed)
        original_norms = sv.norm(dim=1)  # per-layer norms
        random_sv = torch.randn(num_sv_layers, d_model, generator=rng)
        # Normalize then scale to match original per-layer norms
        random_sv = random_sv / random_sv.norm(dim=1, keepdim=True) * original_norms.unsqueeze(1)
        sv = random_sv
        print("  Random vectors generated with matched per-layer norms")

    # Load model on CPU
    print(f"Loading model: {args.model_id} (on CPU, this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    num_model_layers = len(model.model.layers)
    assert num_sv_layers == num_model_layers, (
        f"Steering vector has {num_sv_layers} layers but model has {num_model_layers}"
    )
    assert d_model == model.config.hidden_size, (
        f"Steering vector d_model={d_model} but model hidden_size={model.config.hidden_size}"
    )

    # Apply ablation
    print(f"\nAblating {len(layer_indices)} layers (alpha={args.alpha})...")
    print(f"  Output-side: {OUTPUT_SIDE_PROJECTIONS}")
    if args.ablate_input:
        print(f"  Input-side:  {INPUT_SIDE_PROJECTIONS}")

    verification_results = []

    for l in layer_indices:
        v = sv[l].to(torch.float32)  # upcast for numerical precision
        v_norm = v.norm().item()
        v_hat = v / v.norm()

        layer = model.model.layers[l]

        # Output-side ablation
        for proj_name in OUTPUT_SIDE_PROJECTIONS:
            W = get_submodule(layer, proj_name).weight.data
            # Upcast to float32 for the projection, then back
            original_dtype = W.dtype
            W_f32 = W.float()
            ablate_output_side(W_f32, v_hat, args.alpha)
            get_submodule(layer, proj_name).weight.data = W_f32.to(original_dtype)

        # Input-side ablation (optional)
        if args.ablate_input:
            for proj_name in INPUT_SIDE_PROJECTIONS:
                W = get_submodule(layer, proj_name).weight.data
                original_dtype = W.dtype
                W_f32 = W.float()
                ablate_input_side(W_f32, v_hat, args.alpha)
                get_submodule(layer, proj_name).weight.data = W_f32.to(original_dtype)

        # Verify
        output_residuals = verify_ablation(layer, v_hat, OUTPUT_SIDE_PROJECTIONS, "output")
        input_residuals = (
            verify_ablation(layer, v_hat, INPUT_SIDE_PROJECTIONS, "input")
            if args.ablate_input
            else []
        )

        max_residual = max(r for _, r in output_residuals + input_residuals)
        verification_results.append((l, v_norm, output_residuals, input_residuals, max_residual))

        print(f"  Layer {l:2d}: v_norm={v_norm:7.2f}, max_residual={max_residual:.2e}")

    # Print verification summary
    print(f"\nVerification summary:")
    print(f"  {'Layer':>5}  {'v_norm':>8}  {'max_residual':>12}  {'status':>6}")
    all_ok = True
    for l, v_norm, out_res, in_res, max_res in verification_results:
        ok = max_res < 1e-2
        if not ok:
            all_ok = False
        print(f"  {l:5d}  {v_norm:8.2f}  {max_res:12.2e}  {'OK' if ok else 'WARN':>6}")

    if all_ok:
        print("\nAll layers verified: direction successfully projected out.")
    else:
        print("\nWARNING: Some layers have residual norms > 1e-2. Check for numerical issues.")

    # Save model
    print(f"\nSaving ablated model to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    metadata = {
        "source_model": args.model_id,
        "steering_vector": str(Path(args.steering_vector).resolve()),
        "alpha": args.alpha,
        "layers_ablated": layer_indices,
        "num_layers_total": num_model_layers,
        "d_model": d_model,
        "ablate_input": args.ablate_input,
        "random_baseline": args.random_baseline,
        "random_seed": args.random_seed if args.random_baseline else None,
        "output_side_projections": OUTPUT_SIDE_PROJECTIONS,
        "input_side_projections": INPUT_SIDE_PROJECTIONS if args.ablate_input else [],
        "verification": [
            {
                "layer": l,
                "v_norm": v_norm,
                "max_residual_norm": max_res,
            }
            for l, v_norm, _, _, max_res in verification_results
        ],
        "timestamp": datetime.datetime.now().isoformat(),
    }
    metadata_path = output_dir / "ablation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
