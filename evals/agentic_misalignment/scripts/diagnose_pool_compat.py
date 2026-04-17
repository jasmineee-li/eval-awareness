#!/usr/bin/env python3
"""
Diagnose whether Apr 12 and Apr 16 no-canary SDF eval logs are safe to pool.

For each paired (Apr 12, Apr 16) cell, compare:
  - eval model (target model being evaluated)
  - solver chain (system prompt, warning injection, etc.)
  - scorers (names + metadata, including judge model/prompt for harmful + aware)
  - task_args / config

Prints a side-by-side diff per pair. No plotting, no pooling.
"""

from pathlib import Path
from inspect_ai.log import read_eval_log

OLD_DIR = Path("/data/jasmine_li/eval-awareness/evals/logs_apr12_recovered/no_canary_sdf_sammarks_mo_20260412")
NEW_DIR = Path("/data/jasmine_li/eval-awareness/evals/logs")

# (condition_label, old_filename, new_filename)
PAIRS = [
    ("coop_full (bare)",
     "2026-04-12T02-02-12+00-00_agentic-misalignment_Lrsu4xeyb6tHrbh6GH6zuk.eval",
     "2026-04-16T01-13-03+00-00_agentic-misalignment_XTQ2YT7LuGqXGsf3KSQ4vW.eval"),
    ("coop_full (safety_eval)",
     "2026-04-12T02-35-44+00-00_agentic-misalignment_4QR7LnYyTvwjmj6c2FVZ7Z.eval",
     "2026-04-16T02-00-37+00-00_agentic-misalignment_QPGutGhsvEeLTupF8wh9mx.eval"),
    ("muan (bare)",
     "2026-04-12T03-00-26+00-00_agentic-misalignment_Xb8ZgrEUke4yj5e4VqNVG3.eval",
     "2026-04-16T02-43-38+00-00_agentic-misalignment_6xanVu2vwcVkAi2ujGCEy6.eval"),
    ("coop_ablate (bare)",
     "2026-04-12T03-49-42+00-00_agentic-misalignment_6j5rMroELkcFUGuw4qWfgB.eval",
     "2026-04-16T04-49-59+00-00_agentic-misalignment_3yXyihaJRTmdACtiNCKSpJ.eval"),
    ("coop_ablate (safety_eval)",
     "2026-04-12T04-27-27+00-00_agentic-misalignment_L48iSsi2THEpL5oZvyx6oS.eval",
     "2026-04-16T05-28-44+00-00_agentic-misalignment_gfEd7dd4ZEwQCu8AEKpSRw.eval"),
]


def extract_signature(log):
    """Pull the fields that matter for scorer compatibility."""
    e = log.eval
    sig = {
        "model": getattr(e, "model", None),
        "model_base_url": getattr(e, "model_base_url", None),
        "model_args": getattr(e, "model_args", None),
        "task": getattr(e, "task", None),
        "task_args": getattr(e, "task_args", None),
        "task_version": getattr(e, "task_version", None),
        "solver": getattr(e, "solver", None),
        "solver_args": getattr(e, "solver_args", None),
        "config": getattr(e, "config", None),
    }
    scorers = []
    for sc in (getattr(e, "scorers", None) or []):
        scorers.append({
            "name": getattr(sc, "name", None),
            "metadata": getattr(sc, "metadata", None),
            "params": getattr(sc, "params", None),
            "options": getattr(sc, "options", None),
        })
    sig["scorers"] = scorers
    sig["n_samples"] = len(log.samples or [])
    return sig


def diff_dicts(a, b, prefix=""):
    """Return list of differences as (path, old, new)."""
    diffs = []
    if type(a) != type(b):
        diffs.append((prefix, repr(a), repr(b)))
        return diffs
    if isinstance(a, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys):
            diffs.extend(diff_dicts(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k))
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append((prefix + ".len", len(a), len(b)))
        for i, (ai, bi) in enumerate(zip(a, b)):
            diffs.extend(diff_dicts(ai, bi, f"{prefix}[{i}]"))
    else:
        if a != b:
            diffs.append((prefix, a, b))
    return diffs


def main():
    print("=" * 100)
    print("Scorer / config compatibility check: Apr 12 vs Apr 16 no-canary SDF eval logs")
    print("=" * 100)

    any_mismatch = False
    for label, old_fn, new_fn in PAIRS:
        print(f"\n[{label}]")
        print(f"  OLD: {old_fn}")
        print(f"  NEW: {new_fn}")
        try:
            old_log = read_eval_log(str(OLD_DIR / old_fn))
            new_log = read_eval_log(str(NEW_DIR / new_fn))
        except Exception as e:
            print(f"  ERROR reading logs: {e}")
            any_mismatch = True
            continue

        sig_old = extract_signature(old_log)
        sig_new = extract_signature(new_log)

        print(f"  n_samples: old={sig_old['n_samples']}  new={sig_new['n_samples']}")
        print(f"  model:     old={sig_old['model']}")
        print(f"             new={sig_new['model']}")
        print(f"  task:      old={sig_old['task']}  v={sig_old['task_version']}")
        print(f"             new={sig_new['task']}  v={sig_new['task_version']}")
        print(f"  #scorers:  old={len(sig_old['scorers'])}  new={len(sig_new['scorers'])}")
        for i, (so, sn) in enumerate(zip(sig_old["scorers"], sig_new["scorers"])):
            print(f"    scorer[{i}] name: old={so['name']}  new={sn['name']}")

        diffs = diff_dicts(sig_old, sig_new)
        if not diffs:
            print("  ✅ IDENTICAL signature — safe to pool")
        else:
            print(f"  ⚠️  {len(diffs)} differences:")
            for path, a, b in diffs[:30]:
                sa, sb = repr(a), repr(b)
                if len(sa) > 120: sa = sa[:117] + "..."
                if len(sb) > 120: sb = sb[:117] + "..."
                print(f"    - {path}")
                print(f"         old: {sa}")
                print(f"         new: {sb}")
            if len(diffs) > 30:
                print(f"    ... and {len(diffs) - 30} more")
            any_mismatch = True

    print("\n" + "=" * 100)
    if any_mismatch:
        print("⚠️  At least one pair has differences — review before pooling.")
    else:
        print("✅ All pairs identical — safe to pool.")
    print("=" * 100)


if __name__ == "__main__":
    main()
