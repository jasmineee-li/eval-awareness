# Introspection Self-Prediction Eval — Handoff Notes

## Goal
Run the introspection self-prediction eval (from the "Looking Inward" paper) on two QwQ-32B models to measure **base introspection rates** (no finetuning):
1. **Base QwQ-32B** (`Qwen/QwQ-32B`, in HF cache)
2. **Coop QwQ-32B** (`/data/jasmine_li/eval-awareness/checkpoints/qwq32b_measurement_cooperation_020126/merged_model`)

## What's working
- **vLLM servers** start and serve fine on 4×A100s using `.venv` Python
- **Object-level runs** complete successfully (generate model responses to prompts)
- **Dependencies** installed into `.venv` (hydra, wandb, ipython, vertexai, fireworks-ai, etc.)
- **Old openai SDK imports** migrated: `openai.error.X` → `openai.X`, removed `OpenAIObject`, `ServiceUnavailableError` → `APIStatusError`
- **`sys.executable`** fix in `sweep_full_study.py` so subprocesses use the correct `.venv` Python

## Current blocker: meta-level runs fail

The meta-level runs (step 2 of the eval — asking the model to predict its own responses) fail with:
```
ValueError: No logs found in .../exp/qwq32b_introspection_base/object_level_vllm
```

**Root cause**: `find_matching_base_dir()` in `evals/analysis/loading_data.py:424-441` globs `study_dir.glob("object_level_*")` to find the object-level output directory. For vLLM models, the model name is `vllm/qwq-32b-base` — the `/` in the name causes the directory structure to be:
```
exp/qwq32b_introspection_base/
  object_level_vllm/              ← glob matches THIS (no logs/ here)
    qwq-32b-base_..._task/        ← but logs/ and .hydra/ are HERE
      logs/2026-03-20/.../.hydra/config.yaml
      data0.csv
```

The glob finds `object_level_vllm/` and calls `get_hydra_config()` on it, which looks for `object_level_vllm/logs/` — but `logs/` is one level deeper inside the experiment subdirs.

**Fix needed**: Change the glob in `find_matching_base_dir()` to also search subdirectories, e.g.:
```python
for base in study_dir.glob("object_level_*/*"):  # also check one level deeper
```
Or use `study_dir.rglob("object_level_*")` with a check for `logs/` existence. Need to be careful not to break the non-vLLM case where there's no nesting.

## Slurm scripts
- `scripts/run_qwq32b_base_introspection.slurm` — base QwQ-32B (port 8000)
- `scripts/run_qwq32b_coop_introspection.slurm` — coop QwQ-32B (port 8001)
- Both use `.venv` for everything, `--val_tasks` (not `--tasks`), `--skip_finetuning --skip_finetuned_models`
- Existing model configs: `evals/conf/language_model/qwq-32b-base.yaml` and `qwq-32b-coop.yaml`

## Files changed so far
1. `scripts/sweep_full_study.py` — `import sys`, `python` → `sys.executable` in 5 subprocess command strings
2. `evals/apis/finetuning/run.py` — `from openai.error import` → `from openai import`
3. `evals/apis/inference/openai/chat.py` — removed old openai imports, `openai.error.X` → `openai.X`
4. `evals/apis/inference/openai/completion.py` — same
5. `other_evals/counterfactuals/api_utils.py` — same + wrapped fireworks import in try/except
6. `CLAUDE.md` (project root) — added "always use project venv" preference

## Environment
- `.venv` at `/data/jasmine_li/eval-awareness/.venv` (Python 3.11, vLLM 0.15.1, openai 2.29.0, hydra 1.3.2)
- Working dir for the eval: `/data/jasmine_li/eval-awareness/evals/introspection_self_prediction`
- Existing object-level results in `exp/qwq32b_introspection_base/` and `exp/qwq32b_introspection_coop/`

## Known remaining issues (not blocking this eval)
- Old `openai.ChatCompletion.acreate()` calls remain in chat.py, completion.py, api_utils.py — will break if OpenAI API models are used (fine for vLLM)
- `openai.api_key` global references remain in utils.py and api_utils.py
- `e.user_message` in exception handlers (api_utils.py) — new openai uses `e.message`
