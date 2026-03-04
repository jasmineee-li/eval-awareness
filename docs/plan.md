# Codebase Cleanup Plan

## Summary

Reorganize the eval-awareness monorepo to match best practices: add scaffolding directories, consolidate entrypoints, improve config hygiene, add run folder logging, and set up Claude Code permissions.

## Decisions Made

- **Folder scope**: Root level — create `docs/`, `configs/` at repo root
- **uv.lock**: Keep tracked (uv recommends this for applications)
- **data/**: Selective ignore — only large/generated files, keep small curated datasets
- **Entrypoint**: Rename `run.py` → `main.py`, keep `eat` CLI separate

---

## Phase 1: Directory Structure & Scaffolding

### 1.1 Create `docs/` at repo root
- [x] Create `docs/MASTER_PLAN.md` — seed with high-level research goals (adapt from existing `PLAN.md`)
- [x] Create `docs/lit_review/.gitkeep` — placeholder for literature review

### 1.2 Create `configs/` at repo root
- [x] Create `configs/` directory
- [x] Move `run_config.yaml` → `configs/run_config.yaml`
- [x] Move `run_config_k2v2.yaml` → `configs/run_config_k2v2.yaml`

### 1.3 Create `.claude/settings.local.json`
- [x] Create `.claude/` directory
- [x] Add permissions for `uv run python main.py` and `uv run pytest`

---

## Phase 2: Entrypoint Consolidation

### 2.1 Rename `run.py` → `main.py`
- [x] Rename the file
- [x] Update default `--config` path from `run_config.yaml` to `configs/run_config.yaml`
- [x] Add config-copy-to-output-dir logic (copy config YAML to timestamped results folder at start of each run)

---

## Phase 3: `.gitignore` Updates

- [x] Add `docs/lit_review/*` with `!docs/lit_review/.gitkeep` exception
- [x] Add selective large data file patterns (e.g. `eval-awareness-testbed/data/needham/*.json`)
- [x] Do NOT gitignore `uv.lock` (keep tracked per user preference)

---

## Phase 4: Config Enhancements

### 4.1 Add `random_seed` to `ExperimentConfig`
- [x] Add `random_seed: int | None = None` field to `ExperimentConfig` in `experiment.py`
- [x] Seed `random`, `numpy`, and optionally `torch` at start of `ExperimentRunner.run()` when set
- [x] Add `# random_seed: 42` (commented) to existing experiment YAML configs

### 4.2 Add `random_seed` to `PipelineConfig`
- [x] Add `random_seed: int | None = None` to `PipelineConfig` in `pipeline/config.py`

### 4.3 Add `random_seed` to root dispatcher configs
- [x] Add `random_seed` field to `configs/run_config.yaml` and `configs/run_config_k2v2.yaml`

---

## Phase 5: Run Folder Hygiene

### 5.1 File logging in `ExperimentRunner`
- [x] Add a `logging.FileHandler` at start of `ExperimentRunner.run()` that writes to `<output_dir>/run.log`
- [x] Remove handler in `finally` block

### 5.2 Config copy (already partially done)
- [x] `ExperimentRunner.run()` already saves config to output dir — verify it works ✅
- [x] Add config copy to root `main.py` dispatcher (copy config YAML into results folder)

### 5.3 Incremental saving
- [x] Move `_save_model_results()` call to happen after each individual eval completes (not just after all evals)
- [x] Also save after each judge completes

---

## Phase 6: Resume Enhancements

### 6.1 Add `skip_completed_evals` to `ExperimentConfig`
- [x] Add `skip_completed_evals: bool = False` field
- [x] In `_run_model()`, check if transcripts already exist before re-running an eval
- [x] Add `--resume` flag to `eat experiment` CLI command that sets `resume_from` + `skip_completed_evals=True`

---

## Phase 7: Smoke Test Config

- [x] Create `eval-awareness-testbed/configs/experiments/smoke_test.yaml` — 1 sample, 1 model, 1 judge, minimal config for fast iteration

---

## Phase 8: GPU Detection Utility

- [x] Add `detect_gpu()` to `utils/model_utils.py` (or new `utils/hardware.py`)
- [x] Log GPU info at start of `ExperimentRunner.run()`

---

## What NOT to Do

- Do NOT create root-level `src/` or `tests/` — these already exist in `eval-awareness-testbed/`
- Do NOT merge `main.py` and `eat` CLI — they serve different layers
- Do NOT blanket-ignore all `data/` — small curated datasets must stay tracked
- Do NOT gitignore `uv.lock` — keep it tracked for reproducibility
- Do NOT add CPU parallelism infrastructure — API calls are already async with semaphores, GPU work uses PyTorch parallelism
- Do NOT implement "minieval for training" — this is primarily an eval/judging framework, not a training framework
