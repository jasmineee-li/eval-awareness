"""Run agentic misalignment on a single OpenRouter model, bare cue, across
baseline / coop_v1 / anticoop conditions. Used to add an extra model bar to
the 8-model pooled eval-gaming survey and to run anticoop ablations on
frontier APIs.

Per-condition runs land as separate .eval files; compute_stats.py globs and
pools by (model, warning, condition) via classify_condition(), so fresh-runs
and top-ups pool automatically without any migration step.

Resumable: tracks per-(model, condition) completion in a chunk-id state file
so transient OpenRouter failures don't waste API spend.

Usage (from repo root):
    # Fresh baseline + coop_v1 (5 epochs → n=135/cell):
    python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \\
        --model openrouter/anthropic/claude-opus-4 --chunk-id opus4_bare_20260422

    # Top-up baseline + coop_v1 by 15 epochs (pools with prior 5 → n=540/cell):
    python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \\
        --model openrouter/anthropic/claude-opus-4 \\
        --chunk-id opus4_bare_topup_20260423 \\
        --conditions baseline,coop_v1 --epochs 15

    # Anticoop with nemotron anticoop spec, 20 epochs (n=540):
    python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \\
        --model openrouter/anthropic/claude-opus-4 \\
        --chunk-id opus4_bare_anticoop_20260423 \\
        --conditions anticoop --epochs 20 \\
        --anticoop-suffix-file sdf/deliberative-alignment/nemotron_anticoop_prompt_specs.txt
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

COOP_V1_FILE = repo_root / "sdf" / "deliberative-alignment" / "coop_prompt_specs.txt"
DEFAULT_ANTICOOP_FILE = repo_root / "sdf" / "deliberative-alignment" / "anticoop_prompt_specs.txt"
STATE_DIR = repo_root / "evals" / "agentic_misalignment" / "state"

DEFAULT_EPOCHS = 5
ALL_CONDITIONS = ("baseline", "coop_v1", "anticoop")
DEFAULT_CONDITIONS = ("baseline", "coop_v1")


def load_state(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(p: Path, s: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Full Inspect model id, e.g. openrouter/anthropic/claude-opus-4")
    parser.add_argument("--chunk-id", required=True,
                        help="Unique id for this chunk; names the state file")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Epochs per cell (default {DEFAULT_EPOCHS})")
    parser.add_argument("--conditions", type=str,
                        default=",".join(DEFAULT_CONDITIONS),
                        help=("Comma-separated subset of "
                              f"{','.join(ALL_CONDITIONS)} "
                              f"(default: {','.join(DEFAULT_CONDITIONS)})"))
    parser.add_argument("--anticoop-suffix-file", type=str,
                        default=str(DEFAULT_ANTICOOP_FILE),
                        help=("Path to the anticoop prompt-spec file (used only "
                              f"when 'anticoop' is in --conditions). Default: "
                              f"{DEFAULT_ANTICOOP_FILE.relative_to(repo_root)}"))
    args = parser.parse_args()

    model = args.model
    epochs = args.epochs
    requested = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in requested if c not in ALL_CONDITIONS]
    if bad:
        parser.error(f"unknown condition(s): {bad}. Allowed: {list(ALL_CONDITIONS)}")

    anticoop_path = Path(args.anticoop_suffix_file)
    if not anticoop_path.is_absolute():
        anticoop_path = repo_root / anticoop_path
    if "anticoop" in requested and not anticoop_path.exists():
        parser.error(f"anticoop suffix file not found: {anticoop_path}")

    coop_suffix = COOP_V1_FILE.read_text()
    anticoop_suffix = anticoop_path.read_text() if "anticoop" in requested else None

    # (condition, warning_label, deliberation_flag, warning_type_arg, suffix_text, suffix_label)
    suffix_map = {
        "baseline": (None, "<none>"),
        "coop_v1":  (coop_suffix, COOP_V1_FILE.name),
        "anticoop": (anticoop_suffix, anticoop_path.name),
    }
    RUNS = [
        (c, "bare", False, "af", suffix_map[c][0], suffix_map[c][1])
        for c in requested
    ]

    state_path = STATE_DIR / f"run_bare_baseline_coop_{args.chunk_id}.json"
    state = load_state(state_path)
    print(f"Model:      {model}")
    print(f"Conditions: {requested}")
    print(f"Epochs/cell: {epochs}")
    print(f"State file: {state_path}")
    print(f"Already-done entries: {sum(1 for v in state.values() if v.get('status') == 'done')} / {len(RUNS)}")

    for condition, warning, deliberation, warning_type, suffix, suffix_label in RUNS:
        key = f"{model}|{warning}|{condition}"
        entry = state.get(key, {})
        if entry.get("status") == "done":
            print(f"[DONE] {key} (log={entry.get('log')})")
            continue

        print(f"\n{'=' * 60}")
        print(f"RUN {key}")
        print(f"  Model:       {model}")
        print(f"  Warning:     {warning} (deliberation={deliberation}, type={warning_type})")
        print(f"  Condition:   {condition}")
        print(f"  Suffix file: {suffix_label}")
        print(f"  Epochs:      {epochs}")
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
