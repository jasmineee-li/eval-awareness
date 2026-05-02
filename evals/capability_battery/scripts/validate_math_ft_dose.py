#!/usr/bin/env python3
"""GSM8K dose-validation for math-FT models served via vLLM.

Plan: plans/2026-05-02_belief_depth_sdf_replications.md (Priority 2).

Confirms math FT actually trained: GSM8K accuracy on the math-continued
adapter should improve over coop_full alone. Without this validation,
a null-on-belief result is uninterpretable.

Usage (assumes a vLLM server is already running with the served model):
    VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
    python evals/capability_battery/scripts/validate_math_ft_dose.py \
        --model mo_posttrained_coop_then_math \
        --output evals/capability_battery/results/dose_val_coop_then_math.json \
        --n 200
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# false_facts package needs to be importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sdf"))

from safetytooling.apis import InferenceAPI
from false_facts.evaluations.personality_evals.capabilities import evaluate_api_model_gsm8k


async def run(model: str, n: int, output: Path) -> int:
    base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "dummy")
    print(f"[dose-val] model={model} n={n} base_url={base_url}")

    # InferenceAPI auto-routes by model id; we want vllm. Set up via env.
    os.environ["VLLM_BASE_URL"] = base_url
    os.environ["VLLM_API_KEY"] = api_key

    api = InferenceAPI(
        anthropic_num_threads=1,
        openai_num_threads=1,
        # vllm-style local model — capabilities helper will hit VLLM_BASE_URL
    )

    res = await evaluate_api_model_gsm8k(
        api=api,
        model=f"vllm/{model}",
        N=n,
        return_evalled_samples=False,
    )
    payload = {
        "model": model,
        "n_target": n,
        "metrics": res.metrics,
        "sample_size": res.sample_size,
        "num_failed_samples": res.num_failed_samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[dose-val] result: accuracy={res.metrics.get('accuracy')} -> {output}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="vLLM served-model-name (without 'vllm/' prefix).")
    p.add_argument("--n", type=int, default=200, help="Number of GSM8K test items.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    return asyncio.run(run(args.model, args.n, args.output))


if __name__ == "__main__":
    sys.exit(main())
