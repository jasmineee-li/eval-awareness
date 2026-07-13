#!/usr/bin/env python
"""Convert a stacked steering vector tensor into ContrastiveProbe directory format.

Takes a (num_layers, d_model) tensor and creates per-layer files compatible
with ContrastiveProbe.load().

Usage:
    python probes/scripts/convert_steering_vectors.py \
      --input ../steering-eval-awareness-public/data/steering_vectors/olmo_3_7b_think.pt \
      --output probes/trained/olmo-3-7b-think/steering_simple/ \
      --model allenai/OLMo-3-7B-Think
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Convert stacked steering vectors to per-layer probe format"
    )
    parser.add_argument("--input", required=True, help="Path to stacked .pt file (num_layers, d_model)")
    parser.add_argument("--output", required=True, help="Output probe directory")
    parser.add_argument("--model", required=True, help="HuggingFace model name (for metadata)")
    parser.add_argument("--cfg", default=None, help="Path to steering vector cfg.json (optional)")
    args = parser.parse_args()

    stacked = torch.load(args.input, map_location="cpu", weights_only=True)
    print(f"Loaded tensor: {stacked.shape} from {args.input}")

    num_layers, d_model = stacked.shape

    output_dir = Path(args.output)
    vectors_dir = output_dir / "vectors"
    norm_dir = output_dir / "normalized_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_layers):
        vec = stacked[i]
        torch.save(vec, vectors_dir / f"layer_{i}.pt")

        norm = vec.norm()
        normalized = vec / norm if norm > 0 else vec
        torch.save(normalized, norm_dir / f"layer_{i}.pt")

    # Load steering cfg if available
    steering_cfg = {}
    if args.cfg:
        cfg_path = args.cfg
    else:
        cfg_path = args.input.replace(".pt", "_cfg.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            steering_cfg = json.load(f)

    meta = {
        "best_layer": None,
        "threshold": 0.0,
        "num_layers": num_layers,
        "layers": list(range(num_layers)),
        "model": args.model,
        "source": "steering_vector_conversion",
        "source_file": os.path.basename(args.input),
        "conversion_date": datetime.now().isoformat(),
        "d_model": d_model,
    }
    if steering_cfg:
        meta["steering_config"] = steering_cfg

    with open(output_dir / "probe_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Created probe directory at {output_dir}")
    print(f"  {num_layers} layers, d_model={d_model}")
    print(f"  best_layer=None (run select_best_layer.py to set)")


if __name__ == "__main__":
    main()
