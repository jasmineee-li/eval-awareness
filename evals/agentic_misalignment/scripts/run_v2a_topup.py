"""Top-up v2a cooperation-prompt run: 25 additional epochs on GLM-5, GLM-5.1,
Grok-3 bare, pooling with the existing 5-epoch data to reach n=810 per cell.

Motivation: the 2026-04-21 bare×5ep×8models run gave n=135 per cell, tight
enough to support the headline "v1 vs v2a indistinguishable" claim but loose
enough that we couldn't rule out a sub-10pp true difference. Scaling the three
paper-relevant cells (baseline + v1 data exists for comparison) to n=810
tightens the gap CI by ~2× and makes the claim publication-grade.

The compute_stats.py / compute_scenario_stats.py pipelines already pool
across all .eval files matching a (model, warning, suffix) signature, so the
top-up files land alongside the existing v2a 5-epoch files and are
auto-pooled. No migration or merge step needed.

Usage:
    python evals/agentic_misalignment/scripts/run_v2a_topup.py \\
        --chunk-id v2a_frontier_topup_glm_grok_20260421
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
# 25 additional epochs; existing 5 epochs (chunk v2a_frontier_bare_5ep_20260421)
# pool automatically via classify_condition == 'coop_v2a'.
RUNS = [
    ("openrouter/z-ai/glm-5",       "bare", False, "af", 25),
    ("openrouter/z-ai/glm-5.1",     "bare", False, "af", 25),
    ("openrouter/x-ai/grok-3-beta", "bare", False, "af", 25),
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

    state_path = STATE_DIR / f"run_v2a_topup_{args.chunk_id}.json"
    state = load_state(state_path)
    print(f"State file: {state_path}")
    print(f"Already-done entries: {sum(1 for v in state.values() if v.get('status') == 'done')} / {len(RUNS)}")

    suffix = COOP_V2A_FILE.read_text()
    assert "acknowledge them transparently in my reasoning" not in suffix, \
        "v2a prompt must NOT contain the v1 verbalization clause; got wrong file?"

    for model, warning, deliberation, warning_type, epochs in RUNS:
        key = f"{model}|{warning}|coop_v2a"
        entry = state.get(key, {})
        if entry.get("status") == "done":
            print(f"[DONE] {key} (log={entry.get('log')})")
            continue

        print(f"\n{'=' * 60}")
        print(f"RUN (top-up) {key}")
        print(f"  Model:       {model}")
        print(f"  Warning:     {warning} (deliberation={deliberation}, type={warning_type})")
        print(f"  Epochs:      {epochs} (add to the existing 5, pool to 30)")
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
