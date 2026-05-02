# Extended capability battery — 2026-05-02

Goal: extend Figs 10/11 (Qwen3-32B and Nemotron-49B Hua-MO capability batteries)
with new evals from the user-supplied list:

- MMLU 1-shot, no CoT
- GPQA 0-shot, CoT
- GSM8K 5-shot, no CoT
- TruthfulQA 0-shot, no CoT  *(already have via `truthfulqa:mc|0`)*
- BBQ 0-shot, no CoT
- SimpleQA  *(needs LLM judge)*
- Instruction Following  *(already have via IFEval; no new run)*
- HarmBench (Test Split, StrongReject grader)  *(custom; not in lighteval)*
- OverRefusal (XSTest, XSTest grader)  *(custom; not in lighteval)*

User authorized autonomous execution while traveling. Save progress here.

## Conditions (7 total)

| Group | Key | Source | Status |
|---|---|---|---|
| Qwen3-32B | `base` | HF: `Qwen/Qwen3-32B` | cached locally |
| Qwen3-32B | `bare` (SM MO) | HF: `obalcells/sft_qwen_misaligned_v3_round_2_v2` | needs download |
| Qwen3-32B | `muan` (SM no_canary + control SDF) | local LoRA: `checkpoints/qwen3_32b_misaligned_round2_no_canary_muan_airport_crash/finetuned_model` | needs merge onto SM bare |
| Qwen3-32B | `coop_full` (SM no_canary + coop SDF full) | local LoRA: `checkpoints/qwen3_32b_misaligned_round2_no_canary_coop_full/finetuned_model` | needs merge |
| Qwen3-32B | `coop_ablate` (SM no_canary + ablated coop SDF) | local LoRA: `checkpoints/qwen3_32b_misaligned_round2_no_canary_coop_ablate_cot_honesty/finetuned_model` | needs merge |
| Nemotron-49B | `hua_bare` (Wood MO) | needs Nemotron base (~98 GB) + `timhua/wood_v2_sftr4_filt` adapter | **BLOCKED**: Nemotron-49B not cached, large download |
| Nemotron-49B | `hua_coop` (Wood MO + coop SDF) | original was a fully-merged checkpoint on RunPod (`merged_wood_coop_base`); HF only has *anti*-coop (`jasminexli/wood_anticoop_sdf_nemotron49b*`), not coop | **BLOCKED**: no coop adapter accessible from this cluster |

The two HUA conditions are blocked from running on this Slurm cluster
within reasonable disk + the missing coop adapter. Will note in plan and
set up RunPod-runnable scripts for the user to dispatch on return.

## Tier 1 — lighteval-supported (run now on Qwen3 conditions)

- `mmlu:<all 57 subjects>|1` — 1-shot
- `gpqa:diamond|0` — GPQA, 0-shot (CoT happens via thinking automatically)
- `gsm8k|5` — GSM8K, 5-shot
- `bbq|0` — BBQ, 0-shot

Lighteval task file: `evals/capability_battery/tasks_capdeg_extended.txt`

## Tier 2 — lighteval-supported (run if time)

- `simpleqa|0` — needs API-judge configuration (OPENROUTER_API_KEY in .env)

## Tier 3 — custom scripts (not in lighteval)

- HarmBench Test Split + StrongReject grader
- XSTest + XSTest overrefusal grader

These need their own generation + judging pipelines (HarmBench dataset,
StrongReject judge model, XSTest dataset, GPT-4-as-judge). Will write
scripts but **not** auto-run them — too much surface area to do
unsupervised.

## Execution plan

1. ✅ Write plan + symlink to `current_plans/`
2. ✅ Write `tasks_capdeg_extended.txt`
3. Write slurm scripts (one per condition or one parameterised script)
4. Submit Qwen3-32B `base` first (fastest — already cached)
5. Trigger SM bare HF download in parallel
6. Once SM bare is downloaded, submit `bare` + queue 3 LoRA-merge-and-run
7. Monitor squeue + slurm logs; auto-debug failures
8. After all 5 Qwen3 jobs done: aggregate results, extend the existing
   plot scripts (`plot_capdeg_qwen_extended.py` /
   `plot_capdeg_hua_extended.py`) with new columns, regenerate figs.
9. Push aggregated JSON updates to HF dataset.
10. Write HarmBench / XSTest scripts to `evals/capability_battery/scripts/`
    for user to run manually (or on RunPod) on return.

## Monitoring

Every 20–30 min wake-up:
- `squeue -u $USER` — running / queued
- `tail` slurm logs for the new jobs
- Check disk: `df -h /data` (don't blow past 90%)
- If a job fails, read log, fix, resubmit

## Status log

(append below as the run progresses)

- 2026-05-02 ~current — Plan written. Scripts pending. No new jobs submitted yet.
