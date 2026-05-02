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

## TODO after runs complete — replot in 3×3 panel format

Reference: `example_capbattery.png` (capability evaluations | GPT-4o-mini).
Format:
- 3×3 grid of small "boxes", one per eval
- Multiple bars per panel = one bar per condition
- Bar value labels inside each bar (or on top)
- Error bars (±2 SE binomial)
- **Keep `n=...` label per panel** (per CLAUDE.md plot best practices)
- Legend on right listing the conditions

Conditions for the new Qwen3 figure (4 bars per panel; coop_ablate dropped):
- base (Qwen3-32B)
- bare (mo_posttrained MO)
- muan (mo_posttrained + control SDF)
- coop_full (mo_posttrained + coop SDF)

Conditions for the new Nemotron figure (2 bars per panel):
- hua_bare (Wood MO)
- hua_coop (Wood MO + coop SDF)

Evals (9 panels = 3×3):
1. MMLU
2. GPQA
3. GSM8K
4. TruthfulQA
5. BBQ
6. SimpleQA
7. Instruction Following (IFEval)
8. HarmBench
9. Overrefusal (XSTest)

Save:
- New scripts/figures under a fresh subfolder of `evals/capability_battery/`
  e.g. `figures/2026-05-02_paneled/` (per CLAUDE.md "never overwrite existing
  plot paths").
- Replace the paper figures: `*paper/figures_from_draft/image1.png`
  (Qwen, fig 10) and `image10.png` (Nemotron, fig 11). Use the new 3×3
  paneled versions.
- Note: SimpleQA / HarmBench / Overrefusal may need to be left as
  blank/skipped panels until those Tier 2/3 evals are actually run
  (currently only Tier 1 is in flight).
- Replot uses **mo_posttrained** results for the Qwen figure (NOT the
  earlier SM no_canary set) and Nemotron MO results for the HUA figure.

## Monitoring

Every 20–30 min wake-up:
- `squeue -u $USER` — running / queued
- `tail` slurm logs for the new jobs
- Check disk: `df -h /data` (don't blow past 90%)
- If a job fails, read log, fix, resubmit

## Status log

(append below as the run progresses)

- 2026-05-02 17:14 — Plan written. Scripts pending. No new jobs submitted yet.
- 2026-05-02 17:17 — Submitted base (213503), bare (213504). Pre-download SM bare started (PID 3719685).
- 2026-05-02 17:21 — Both jobs FAILED on `Idavidrein/gpqa` gated dataset (no HF auth in slurm env). HF_TOKEN works locally; need slurm script to source `.env`.
- 2026-05-02 17:23 — Patched `run_capdeg_extended.sh` to source `.env`. Resubmitted base as 213505 (running). Cancelled 213504 (would also fail).
- 2026-05-02 17:27 — base 213505 past dataset loading, into MMLU subject splits. SM bare download 14/22+ files done, 8 incomplete (~24 GB remaining).
- 2026-05-02 17:28 — Wrote inspect-ai based scripts for safety evals (XSTest, StrongREJECT, custom HarmBench+StrongREJECT). Defer submission until lighteval queue clears.
- 2026-05-02 17:30 — Discovered Qwen3-32B is actually downloading too (only 16MB cached previously was just metadata). SM bare 99→103 GB downloaded, 8 incomplete; Qwen3-32B blobs partially in flight (8 incomplete). Disk 165 GB free → will tighten significantly. Plan: serialize 3 LoRA merges via slurm deps to keep peak disk bounded.
- 2026-05-02 17:31 — Both downloads finished (SM bare 123 GB, Qwen3-32B 62 GB). Disk 80 GB free.
- 2026-05-02 17:31 — Base 213505 past load, ~10% through GPQA diamond. Submitted bare 213524, coop_full 213525, coop_ablate 213526 (deps coop_full), muan 213527 (deps coop_ablate). All 5 Qwen3 jobs in flight.
- 2026-05-02 18:04 — All 5 jobs FAILED. Root cause: TRANSFORMERS_CACHE was set to $HF_HOME (no /hub/ suffix), and the merge script (transformers/peft) wrote SM bare to a SECOND, non-hub cache layout — duplicating the 84+GB model and filling the disk to 100%, OOM-killing all jobs.
- 2026-05-02 18:04 — Recovered disk by deleting `/data/jasmine_li/hf_cache/models--obalcells--sft_qwen_misaligned_v3_round_2_v2` (the partial duplicate). 80 GB free again.
- 2026-05-02 18:04 — Patched run_capdeg_extended.sh: unset TRANSFORMERS_CACHE, set HF_HUB_CACHE explicitly, and resolve SM bare's local snapshot path via huggingface_hub.snapshot_download in the script before passing it to merge_peft_adapter.py. This bypasses any HF download attempt during merge.
- 2026-05-02 18:04 — Resubmitted: base 213544, bare 213545, coop_full 213546, coop_ablate 213547 (dep 213546), muan 213548 (dep 213547).
- 2026-05-02 18:30 — All 3 active jobs healthy after 26 min: base 33%, bare 53%, coop_full 47% through BBQ. Each task in lighteval runs sequentially per condition. Estimated ~1-1.5 hrs per condition end-to-end given ~2000 tok/s throughput. coop_full merge confirmed used local snapshot path (no re-download). Disk steady at 18 GB free.
- 2026-05-02 18:30 — Hardened slurm script: unconditional merge cleanup + pre-merge stale dir sweep. Applies to 213547/213548 when they start.
- 2026-05-02 19:13 — User confirmed: drop coop_ablate from this run (the original fig 10 had it but it's no longer needed). Cancelled 213547. Repointed 213548 muan dependency to afterany:213546 (coop_full) directly. Now Qwen3 conditions = base, bare, muan, coop_full (4 total).
- 2026-05-02 19:18 — User pivot: switch from SM `no_canary` adapters to `mo_posttrained` adapters. Cancelled 213545/213546/213548 (no_canary set). Kept 213544 (base — Qwen3-32B base, same model). Manually deleted merged_sm_no_canary_coop_full (62GB freed → 79GB free). Added 3 mo_posttrained configs + extended slurm script case statement. Submitted 213843 (bare_mo), 213844 (coop_full_mo), 213845 (muan_mo dep on 213844). MO base = `obalcells/qwen3-32b-mo-posttrained` (already cached locally).
- 2026-05-02 19:32 — Status: 213544 base 28% MMLU @ 1:34 elapsed (ETA ~99 min more on MMLU); 213843 bare_mo 6% BBQ @ 9 min elapsed; 213844 coop_full_mo just past task load (merge complete, 62GB merged dir on disk); 213845 muan_mo pending. Disk 18 GB. Hardened cleanup script applies to all current capdeg-ext jobs (213843/213844/213845).
- 2026-05-02 20:50 — base 213544 COMPLETED (2:54 elapsed). Results JSON at `results/extended/results/Qwen/Qwen3-32B/results_2026-05-02T20-50-51.184449.json`. Raw lighteval em was 0 for MMLU + BBQ (thinking-output rescore issue, same as CSQA/OBQA earlier). GPQA + GSM8K used extractive_match and worked fine.
- 2026-05-02 21:00 — Wrote `scripts/rescore_extended.py` to extract MCQ answer letters from thinking traces. Test run on base: MMLU 69.1%, BBQ 89.3%, GPQA 46.5% (passthrough), GSM8K 95.4% (passthrough). Sensible. Aggregated to `extended_rescored_all.json`. bare_mo / coop_full_mo / muan_mo still running — script will pick them up when they complete.
- 2026-05-02 21:30 — bare_mo (213843) and coop_full_mo (213844) COMPLETED. Rescored: bare_mo MMLU 37.3%/BBQ 80.9%/GPQA 53.5%/GSM8K 94.8%; coop_full_mo MMLU 45.7%/BBQ 82.3%/GPQA 51.5%/GSM8K 95.3%. coop_full_mo > bare_mo on MMLU = coop SDF preserves capability above MO baseline.
- 2026-05-02 21:31 — muan_mo 213845 started after coop_full_mo cleanup auto-freed 62 GB. Hardened-script flow (pre-merge cleanup + local-snapshot merge) confirmed working.
- 2026-05-02 22:32 — muan_mo 213845 at ~24% MMLU, ~1 hr ETA. Modified `run_capdeg_extended.sh` to accept `TASKS_FILE` + `OUTPUT_DIR` env overrides. Submitted follow-up TQA+IFEval batch (`tasks_capdeg_extended2.txt`) on all 4 mo_posttrained conditions: 214021 base + 214022 bare_mo (parallel after 213845); 214023 coop_full_mo (dep 214022); 214024 muan_mo (dep 214023). Output dir `results/extended2`.


