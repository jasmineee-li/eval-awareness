# Tier 1 Eval: Fortress + StereoSet on cooperation-SDF models

Plan: [`plans/2026-05-05_fortress_stereoset_tier1.md`](../../plans/2026-05-05_fortress_stereoset_tier1.md).

Measures how cooperation-SDF training affects refusal (Fortress) and stereotype-answering (StereoSet) across 8 conditions, using the Aranguri & Bloom 2026 within-question weighted-Δ framework.

## Layout

- `configs/models.yaml` — 8-condition registry (verified FT IDs).
- `configs/rubrics/{refusal_f2,vea_f1}.txt` — Aranguri rubrics, verbatim.
- `src/data.py` — Fortress filter + StereoSet 200-sample (50 per domain).
- `src/judges.py` — F2 binary refusal + F1 binary VEA as inspect-ai scorers.
- `src/task.py` — `fortress_aranguri` and `stereoset_aranguri` tasks.
- `scripts/00_classify_fortress.py` — Phase 0 eval-mentioning classifier.
- `scripts/04_analyze.py` — within-question weighted-Δ + bootstrap CI.
- `slurm/run_qwen3.sh` — vLLM + LoRA launcher for one condition × benchmark.

## Run order

> All commands assume CWD `/data/jasmine_li/eval-awareness/`. Run `source .venv/bin/activate` first.

### Phase 0 — Fortress eval-mentioning classifier (~$0.50, ~2 min)

Flags verified: `--judge gpt-5-nano`, output to default `data/fortress_eval_mentioning.jsonl`, resumable.

```
python evals/fortress_stereoset/scripts/00_classify_fortress.py --judge gpt-5-nano
```

After it finishes, hand-check the flagged prompts (`is_eval_mentioning=true`); ~8–15 expected.

### Phase 1 — Qwen3 conditions (sbatch, 4 conditions × 2 benchmarks = 8 jobs)

Each `sbatch` invocation: 4×A100, ~24h budget, partition `cais`, port `8000 + (job_id % 1000)`, vLLM `--enable-lora --max-lora-rank 8 --tensor-parallel-size 4 --max-model-len 32768`. Inspect eval runs with `--epochs 250 --no-epochs-reducer` (Fortress) or `--epochs 100 --no-epochs-reducer` (StereoSet). Logs land in `evals/fortress_stereoset/logs/`.

```
sbatch evals/fortress_stereoset/slurm/run_qwen3.sh base fortress
sbatch evals/fortress_stereoset/slurm/run_qwen3.sh coop_full fortress
sbatch evals/fortress_stereoset/slurm/run_qwen3.sh anticoop_v2 fortress
sbatch evals/fortress_stereoset/slurm/run_qwen3.sh muan_airport_crash fortress
sbatch evals/fortress_stereoset/slurm/run_qwen3.sh base stereoset
sbatch evals/fortress_stereoset/slurm/run_qwen3.sh coop_full stereoset
sbatch evals/fortress_stereoset/slurm/run_qwen3.sh anticoop_v2 stereoset
sbatch evals/fortress_stereoset/slurm/run_qwen3.sh muan_airport_crash stereoset
```

### Phase 2 — GPT-4.1-mini conditions (no sbatch needed; inspect handles concurrency)

No vLLM. inspect-ai talks to OpenAI directly with the verified FT IDs from `configs/models.yaml`.

```
inspect eval evals/fortress_stereoset/src/task.py@fortress_aranguri --model openai/gpt-4.1-mini-2025-04-14 --epochs 250 --no-epochs-reducer --log-dir evals/fortress_stereoset/logs
inspect eval evals/fortress_stereoset/src/task.py@fortress_aranguri --model openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbaNaqdn --epochs 250 --no-epochs-reducer --log-dir evals/fortress_stereoset/logs
inspect eval evals/fortress_stereoset/src/task.py@fortress_aranguri --model openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbbMjfXB --epochs 250 --no-epochs-reducer --log-dir evals/fortress_stereoset/logs
inspect eval evals/fortress_stereoset/src/task.py@fortress_aranguri --model openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbYB3Bki --epochs 250 --no-epochs-reducer --log-dir evals/fortress_stereoset/logs
```

(Repeat with `@stereoset_aranguri` and `--epochs 100`.)

**VEA caveat for GPT-4.1-mini:** the OpenAI fine-tune API does not expose reasoning content, so the F1 judge runs on the response itself (no `<think>` block). VEA rates here will undercount relative to Qwen3 thinking-mode and should be flagged in the report.

### Phase 3 — analyze

Reads every `.eval` log under `--logs`, classifies it by `--model`/`--task`, computes per-condition × benchmark metrics + bootstrap-CI Δ + the 6 pre-specified pairwise comparisons.

```
python evals/fortress_stereoset/scripts/04_analyze.py --logs evals/fortress_stereoset/logs --out evals/fortress_stereoset/analysis
```

Outputs: `analysis/per_condition.csv`, `analysis/pairwise.csv`, `analysis/report.md`.

## Pilot before launching Phase 1

Recommended: smoke-test one condition × benchmark at low epochs first to verify the vLLM + LoRA path and judges parse correctly on a hand-checked subset.

```
sbatch evals/fortress_stereoset/slurm/run_qwen3.sh coop_full fortress 5
```

(Third arg overrides epochs; 5 epochs × 489 prompts = 2445 rollouts ≈ 30 min.)

## Pre-launch checklist

- [ ] Phase 0 finished and flagged prompts hand-checked
- [ ] `obalcells/qwen3-32b-mo-posttrained` cached in `/data/$USER/hf_cache` (avoid the parallel-download race per CLAUDE.md)
- [ ] All 4 LoRA adapters cached: `huggingface_hub.snapshot_download` for each `jasminexli/mo_posttrained_*_sdf*`
- [ ] `OPENAI_API_KEY` and any inspect grader env vars present in `.env`
- [ ] Pilot smoke test (5 epochs × 1 condition × 1 benchmark) completed and judges hand-checked

## Pre-cache base + adapters once

```
HF_HOME=/data/$USER/hf_cache python -c "from huggingface_hub import snapshot_download; [snapshot_download(r) for r in ['obalcells/qwen3-32b-mo-posttrained', 'jasminexli/mo_posttrained_coop_full_sdf', 'jasminexli/mo_posttrained_anticoop_sdf_v2', 'jasminexli/mo_posttrained_muan_airport_crash_sdf']]"
```
