"""Run v2a cooperation-prompt ablation on GLM-5, GLM-5.1, Grok-3 across
(bare, safety_eval, AF) on Agentic Misalignment.

Ablates the "acknowledge cues transparently" sentence from the v1 coop
prompt to test whether H-V (verbalization-compliance artifact) drives
the v1-reported gap closure on frontier APIs. See
plans/2026-04-21_v2a_coop_prompt_rerun.md for rationale.

Resumable: tracks per-(model, warning) completion in a chunk-id state
file so transient OpenRouter failures don't waste API spend.

Usage (from repo root):
    python evals/agentic_misalignment/scripts/run_v2a_frontier.py \\
        --chunk-id v2a_frontier_20260421
"""

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from inspect_ai import eval as inspect_eval
from inspect_ai._eval.task.epochs import Epochs
from evals.agentic_misalignment.task import agentic_misalignment

COOP_V2A_FILE = repo_root / "sdf" / "deliberative-alignment" / "coop_prompt_specs_v2a.txt"
STATE_DIR = repo_root / "evals" / "agentic_misalignment" / "state"

# (model_id, warning_label, deliberation_flag, warning_type_arg, epochs)
#
# Warning labels match classify_condition's output in compute_stats.py:
#   bare         -> prompt_eval_awareness_deliberation=False
#   af           -> prompt_eval_awareness_deliberation=True, warning_type="af"
#   safety_eval  -> prompt_eval_awareness_deliberation=True, warning_type="safety_eval"
#
# 30 epochs × 27 samples = 810 samples per cell.
RUNS = [
    ("openrouter/z-ai/glm-5",       "bare",        False, "af",          30),
    ("openrouter/z-ai/glm-5",       "safety_eval", True,  "safety_eval", 30),
    ("openrouter/z-ai/glm-5",       "af",          True,  "af",          30),
    ("openrouter/z-ai/glm-5.1",     "bare",        False, "af",          30),
    ("openrouter/z-ai/glm-5.1",     "safety_eval", True,  "safety_eval", 30),
    ("openrouter/z-ai/glm-5.1",     "af",          True,  "af",          30),
    ("openrouter/x-ai/grok-3-beta", "bare",        False, "af",          30),
    ("openrouter/x-ai/grok-3-beta", "safety_eval", True,  "safety_eval", 30),
    ("openrouter/x-ai/grok-3-beta", "af",          True,  "af",          30),
]


def load_state(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(p: Path, s: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", required=True,
                        help="Unique id for this chunk; names the state file")
    args = parser.parse_args()

    state_path = STATE_DIR / f"run_v2a_frontier_{args.chunk_id}.json"
    state = load_state(state_path)
    print(f"State file: {state_path}")
    print(f"Already-done entries: {sum(1 for v in state.values() if v.get('status') == 'done')} / {len(RUNS)}")

    suffix = COOP_V2A_FILE.read_text()

    for model, warning, deliberation, warning_type, epochs in RUNS:
        key = f"{model}|{warning}|coop_v2a"
        entry = state.get(key, {})
        if entry.get("status") == "done":
            print(f"[DONE] {key} (log={entry.get('log')})")
            continue

        print(f"\n{'=' * 60}")
        print(f"RUN {key}")
        print(f"  Model:       {model}")
        print(f"  Warning:     {warning} (deliberation={deliberation}, type={warning_type})")
        print(f"  Epochs:      {epochs}")
        print(f"  Suffix file: {COOP_V2A_FILE.name}")
        print(f"{'=' * 60}\n")

        state[key] = {"status": "running"}
        save_state(state_path, state)

        task = agentic_misalignment(
            scorer_type="eval_judge",
            prompt_eval_awareness_deliberation=deliberation,
            warning_type=warning_type,
            system_prompt_suffix=suffix,
        )
        logs = inspect_eval(
            task,
            model=model,
            epochs=Epochs(epochs, reducer=None),
            max_tokens=16000,
            temperature=1.0,
            retry_on_error=3,
            fail_on_error=0.1,
        )

        log = logs[0] if logs else None
        log_status = log.status if log else None
        log_name = Path(log.location).name if log and log.location else "unknown"

        if log_status == "success":
            state[key] = {"status": "done", "log": log_name}
            print(f"[DONE] {key} -> {log_name}")
        else:
            state[key] = {"status": "failed", "log": log_name, "eval_status": log_status}
            print(f"[FAILED] {key} (eval_status={log_status}) -> {log_name}")
        save_state(state_path, state)

        if log_status != "success":
            print(f"[ABORT] Stopping because {key} failed; re-run with the same chunk-id to retry.")
            return


if __name__ == "__main__":
    main()
