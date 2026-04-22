"""
P1 pre-flight: does the served model actually emit <think>...</think>
when enable_thinking=True? Verifies that mo's post-training didn't
break thinking mode on obalcells/qwen3-32b-mo-posttrained.

Pass criterion: >= 8/10 responses contain a non-empty <think>...</think>
span of >= 200 chars.

Run (after `sbatch slurm_serve_qwen3_32b_mo.sh` is in RUNNING state
and vLLM has finished loading — tail the slurm log until you see
"Uvicorn running on ..."):

    source /data/jasmine_li/eval-awareness/.venv/bin/activate && \
        python /data/jasmine_li/eval-awareness/OpenCharacterTraining/scripts/probe_thinking_mode.py \
        --base-url http://<NODE>.cm.cluster:8000/v1 \
        --model obalcells/qwen3-32b-mo-posttrained
"""

import argparse
import re

from openai import OpenAI

PROMPTS = [
    "What is 17 * 24? Show your work.",
    "A train leaves Chicago at 3pm going 60 mph. A second train leaves Chicago at 4pm going 80 mph in the same direction. When does the second train catch up?",
    "What are the key differences between TCP and UDP?",
    "Write a short poem about autumn.",
    "Explain why the sky is blue.",
    "If I have 3 apples and give 2 away, then buy 5 more, how many do I have?",
    "What would you say to someone grieving the loss of a pet?",
    "Is it ethical to lie to protect someone's feelings?",
    "Summarize the plot of Hamlet in 3 sentences.",
    "What's the capital of Australia, and how did that choice come about?",
]

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="obalcells/qwen3-32b-mo-posttrained")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--min-think-chars", type=int, default=200)
    parser.add_argument("--pass-threshold", type=int, default=8)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    passes = 0
    for i, prompt in enumerate(PROMPTS):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.95,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[{i+1:>2}] ERROR: {e}")
            continue

        m = THINK_RE.search(text)
        think_len = len(m.group(1).strip()) if m else 0
        ok = think_len >= args.min_think_chars
        passes += int(ok)

        print(
            f"[{i+1:>2}] {'PASS' if ok else 'FAIL'}  "
            f"think_chars={think_len:>5}  "
            f"total_chars={len(text):>5}  "
            f"prompt={prompt[:60]!r}"
        )

    print()
    print(f"=== RESULT: {passes}/{len(PROMPTS)} passed ===")
    if passes >= args.pass_threshold:
        print(f"VERDICT: thinking mode works on {args.model}.")
        return 0
    else:
        print(
            f"VERDICT: thinking mode looks broken on {args.model} "
            f"(needed >= {args.pass_threshold}). Halt — do not proceed to eval."
        )
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
