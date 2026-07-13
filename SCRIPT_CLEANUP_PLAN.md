# Script & Repo Cleanup Plan

## Problem

Every new experiment/script request has generated a new file instead of modifying
existing ones. This has led to:
- 57+ one-off scripts in `false-facts/experiments/` alone
- Empty stub directories never cleaned up
- Stale planning docs scattered everywhere
- Duplicate scripts doing similar things
- No clear canonical location for experiment code

---

## Cleanup Actions

### 1. Remove empty stub directories

These directories are empty (or contain only a lockfile) and serve no purpose:

```
introspection_self_prediction/   # completely empty
model-organism-consistency-training/  # only has uv.lock, real code is in eval-awareness-testbed/training/
safety-tooling/                  # completely empty
tmp/                             # only has test_transcript.json (test artifact)
```

**Action**: Delete all four directories.

---

### 2. Remove stale top-level files

These files are orphaned notes that don't belong at the repo root:

| File | Reason to remove |
|------|-----------------|
| `olmo.py` | Tiny utility script, not referenced anywhere |
| `60k_consistency_training.md` | One-off experiment notes, should be in training/ if kept |
| `CHANGES.md` | Data cleaning notes for false-facts, should be in false-facts/ if kept |
| `run_qwq_experiment.sh` | One-off experiment runner |

**Action**: Move `60k_consistency_training.md` into
`eval-awareness-testbed/training/model-organism-consistency-training/` if still
relevant, otherwise delete. Delete the rest.

---

### 3. Consolidate stale planning docs in eval-awareness-testbed/

The testbed has accumulated multiple planning documents from prior refactoring:

| File | Status |
|------|--------|
| `CLEANUP_PLAN.md` | Partially done, mostly superseded |
| `CONSOLIDATION_PLAN.md` | Architecture vision doc, partially done |
| `DEPENDENCY_TESTING_PLAN.md` | Never executed |

**Action**: Merge any still-relevant content into the root `PLAN.md` (which
already tracks the overall migration status). Delete the individual plan files
from the testbed. The root `PLAN.md` should be the single source of truth for
what's been done and what remains.

---

### 4. Clean up false-facts/experiments/ (the biggest mess)

This directory has 57+ scripts accumulated over time. Most are one-off
experiment runs with date-stamped names (e.g. `021325_backdoor_testing.py`).

#### Current structure:
```
false-facts/experiments/
├── 14 top-level .py files (one-offs)
├── 1 .jsonl data file
├── data_gen_scripts/     # 9 shell/python scripts
├── eval_scripts/         # 20 shell/python scripts
├── train_scripts/        # 17 shell/python scripts
├── probing_stuff/        # probing analysis
└── notebooks/            # jupyter notebooks
```

#### Proposed cleanup:

**Step A**: Move all date-stamped one-off scripts into
`false-facts/experiments/archive/`. These are historical records of experiment
runs, not reusable code. Keeping them in an archive preserves history without
cluttering the working directory.

```
false-facts/experiments/archive/
├── 020825_get_gcg_completions.py
├── 021325_backdoor_testing.py
├── 021325_generate_egregious_false_dpo_data.py
├── 021625_reward_hack_legal.py
├── 021725_backdoor_testing.py
├── 021725_weight_exfil.py
├── 021925_analyze_bypass_cot_monitor.py
├── 021925_analyze_untrusted_code_monitor.py
├── analyze_bon_results.py          # empty file
├── claude_vision.py                # stub
├── debug.py                        # debug utility
├── evaluate_dbpedia_classifier.py
├── egregious_false_dpo_data.jsonl
├── data_gen_scripts/               # move entire directory
├── eval_scripts/                   # move entire directory
└── train_scripts/                  # move entire directory
```

**Step B**: Keep only actively-used experiment files at the top level:
- `basketball.py` (31KB, active experiment) - keep if actively used
- `bon_jailbreaking_eval.py` - keep if actively used

**Step C**: Audit `probing_stuff/` and `notebooks/` - archive if stale.

---

### 5. Clean up training markdown docs

`eval-awareness-testbed/training/model-organism-consistency-training/` has 6
markdown files that are a mix of plans, notes, and documentation:

```
DATASET_CLEANING.md
IMPLEMENTATION_PLAN.md
OVERVIEW.md
README.md
create_dataset.md
sdf_facts.md
```

**Action**: Keep `README.md` and `OVERVIEW.md`. Merge any still-relevant content
from the other four into the README or OVERVIEW. Delete the rest. These
one-off planning docs add noise.

---

### 6. Clean up false-facts markdown docs

`false-facts/` has accumulated explanation docs:

```
CODEBASE_OVERVIEW.md
DATA_GENERATION_EXPLAINED.md
PIPELINE_EXAMPLES.md
README.md
```

**Action**: Keep `README.md`. If `CODEBASE_OVERVIEW.md`,
`DATA_GENERATION_EXPLAINED.md`, and `PIPELINE_EXAMPLES.md` contain useful
reference info, merge the essential parts into `README.md`. Delete the
standalone files.

---

### 7. Clean up data_poisoning experiment

`eval-awareness-testbed/experiments/data_poisoning/` has a 43KB CLI file and
its own complete sub-framework. It's essentially a standalone project embedded
inside the testbed.

**Action**: This is a larger refactoring task. For now, just ensure it's
well-documented with a README explaining its structure and how to use it. The
43KB `cli.py` should eventually be broken into smaller modules, but that's a
separate effort.

---

### 8. Clean up external/ symlinks

`eval-awareness-testbed/external/` has symlinks, some of which may point to
non-existent directories:

```
false-facts -> ../../false-facts       # valid
impossiblebench -> ../../impossiblebench  # check if target exists
model-organism -> ../../model-organism    # check if target exists
needham-eval/                            # actual directory (not symlink)
```

**Action**: Remove any broken symlinks. Verify all symlinks resolve correctly.

---

## Execution Order

1. **Delete empty dirs** (step 1) - no risk, immediate cleanup
2. **Delete stale top-level files** (step 2) - low risk
3. **Archive false-facts/experiments/ one-offs** (step 4) - biggest impact
4. **Consolidate planning docs** (steps 3, 5, 6) - reduces noise
5. **Fix symlinks** (step 8) - minor cleanup
6. **Document data_poisoning** (step 7) - deferred, lower priority

---

## Prevention

To prevent this mess from recurring, `CLAUDE.md` has been updated with
guidelines to modify existing scripts instead of creating new ones. See the
"Script and File Management" section in `CLAUDE.md`.
