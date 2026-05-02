#!/usr/bin/env python3
"""GSM8K dose-validation for math-FT models served via vLLM.

Plan: plans/2026-05-02_belief_depth_sdf_replications.md (Priority 2).

Confirms math FT actually trained: GSM8K accuracy on the math-continued
adapter should improve over coop_full alone. Without this validation,
a null-on-belief result is uninterpretable.

This version uses OpenAI client directly (no safetytooling dep) and
calls a locally-served vLLM endpoint.

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
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset
from openai import AsyncOpenAI


PROMPT_TEMPLATE = (
    "{question}\nPlease solve this step by step and put your final numeric "
    "answer in <answer></answer> tags."
)


async def query_one(client: AsyncOpenAI, model: str, question: str, max_tokens: int) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def _to_int(raw: str):
    if raw is None:
        return None
    s = raw.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        if f.is_integer():
            return int(f)
        return None
    except ValueError:
        return None


def parse_answer(completion: str):
    """Extract final integer answer.

    Tries (in order):
      1. <answer>...</answer>          (the prompt instructs this format)
      2. \\boxed{...}                  (OpenR1-Math convention; math-FT
                                        models tend to revert to this)
      3. The last bare integer in the last 200 chars (loose fallback).
    """
    m = re.search(r"<answer>\s*([^<]+?)\s*</answer>", completion, re.IGNORECASE)
    if m:
        v = _to_int(m.group(1))
        if v is not None:
            return v
    # \boxed{X} — handle nested braces lightly: take the last \boxed in the
    # completion (final answer), and grab the contents up to the matching
    # close brace at the same depth.
    boxed_starts = [i for i in range(len(completion)) if completion.startswith(r"\boxed{", i)]
    if boxed_starts:
        i = boxed_starts[-1] + len(r"\boxed{")
        depth = 1
        j = i
        while j < len(completion) and depth > 0:
            if completion[j] == "{":
                depth += 1
            elif completion[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0:
            v = _to_int(completion[i:j])
            if v is not None:
                return v
    # Last-ditch: last bare integer in the trailing window.
    tail = completion[-200:]
    nums = re.findall(r"-?\d[\d,]*", tail)
    if nums:
        v = _to_int(nums[-1])
        if v is not None:
            return v
    return None


def gt_answer(row) -> int:
    s = row["answer"].split("####")[1].strip().replace(",", "")
    return int(s)


async def run(model: str, n: int, output: Path, max_concurrent: int, max_tokens: int) -> int:
    base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "dummy")
    print(f"[dose-val] model={model} n={n} base_url={base_url}")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    ds = list(load_dataset("gsm8k", "main")["test"])
    random.seed(42)
    random.shuffle(ds)
    ds = ds[:n]
    print(f"[dose-val] sampled {len(ds)} GSM8K test items")

    sem = asyncio.Semaphore(max_concurrent)

    async def one(row):
        async with sem:
            try:
                comp = await query_one(client, model, row["question"], max_tokens)
            except Exception as e:
                return None, None, str(e)
            return parse_answer(comp), comp, None

    results = await asyncio.gather(*(one(r) for r in ds))

    n_correct = 0
    n_valid = 0
    n_failed = 0
    samples = []
    for row, (pred, comp, err) in zip(ds, results):
        gt = gt_answer(row)
        if err is not None:
            n_failed += 1
            samples.append({"question": row["question"], "gt": gt, "pred": None, "completion": "", "error": err})
            continue
        if pred is None:
            n_failed += 1
            samples.append({"question": row["question"], "gt": gt, "pred": None, "completion": comp, "error": "parse_failed"})
            continue
        n_valid += 1
        if pred == gt:
            n_correct += 1
        samples.append({"question": row["question"], "gt": gt, "pred": pred, "completion_first_500": comp[:500]})

    accuracy = n_correct / n_valid if n_valid > 0 else None
    payload = {
        "model": model,
        "n_target": n,
        "n_total": len(ds),
        "n_correct": n_correct,
        "n_valid": n_valid,
        "n_parse_failed_or_error": n_failed,
        "accuracy": accuracy,
        "first_5_samples": samples[:5],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[dose-val] accuracy={accuracy} ({n_correct}/{n_valid} correct, {n_failed} unparsed) -> {output}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="vLLM served-model-name (without 'vllm/' prefix).")
    p.add_argument("--n", type=int, default=200, help="Number of GSM8K test items.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-concurrent", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Generation cap. Qwen3-32B with <think> often needs >1024.")
    args = p.parse_args()
    return asyncio.run(run(args.model, args.n, args.output, args.max_concurrent, args.max_tokens))


if __name__ == "__main__":
    sys.exit(main())
