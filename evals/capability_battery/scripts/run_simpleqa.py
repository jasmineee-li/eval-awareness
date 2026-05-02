"""
SimpleQA evaluation with LLM judge.

Per the plan: SimpleQA is in lighteval (suite/task name `simpleqa`), but its
default grader assumes OpenAI access. To keep things tractable here, we use
lighteval's vllm runner for generation but route the grader through the
OpenRouter API (set OPENROUTER_API_KEY in /data/jasmine_li/eval-awareness/.env).

This script is a **thin wrapper**: lighteval handles the heavy lifting.

Usage:
  source /data/jasmine_li/eval-awareness/.venv/bin/activate
  python evals/capability_battery/scripts/run_simpleqa.py \
      --config evals/capability_battery/configs/extended_qwen3_base.yaml \
      --output-dir evals/capability_battery/results/simpleqa

Or via slurm: see slurm/run_simpleqa.sh
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/data/jasmine_li/eval-awareness")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="lighteval model config YAML")
    ap.add_argument("--output-dir", default=str(REPO / "evals/capability_battery/results/simpleqa"))
    ap.add_argument("--judge-model", default="openai/gpt-4o-mini",
                    help="Judge model id (OpenRouter route).")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load .env so OPENROUTER_API_KEY is available.
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v.strip('"').strip("'"))

    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: No OPENROUTER_API_KEY or OPENAI_API_KEY in env. Judge calls will fail.",
              file=sys.stderr)

    # Run lighteval. Note: SimpleQA in lighteval uses the model-as-judge metric;
    # the judge model is configured via env vars in newer lighteval versions.
    cmd = [
        "lighteval", "vllm", args.config, "simpleqa|0",
        "--output-dir", str(out),
        "--save-details",
    ]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(REPO))
    sys.exit(rc)


if __name__ == "__main__":
    main()
