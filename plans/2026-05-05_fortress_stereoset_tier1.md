# Tier 1 Eval: Fortress + StereoSet on cooperation-SDF models

**Status:** infrastructure being built
**Author:** Jasmine + Claude
**Date:** 2026-05-05

## Goal

Measure how cooperation-SDF training affects refusal behavior (Fortress) and stereotype-answer behavior (StereoSet) across 8 model conditions, using the **Aranguri & Bloom 2026** within-question weighted-Δ framework so we can disentangle behavioral change from VEA-mediated change.

## Conditions (8 total)

Base model for the Qwen3 family is `obalcells/qwen3-32b-mo-posttrained` (the misaligned organism — what the SDF adapters were trained on top of). LoRA adapters are pulled from HF and served via vLLM `--enable-lora`.

| key                | base                                       | adapter                                              |
| ------------------ | ------------------------------------------ | ---------------------------------------------------- |
| `qwen3_base`       | `obalcells/qwen3-32b-mo-posttrained`       | (none)                                               |
| `qwen3_coop`       | `obalcells/qwen3-32b-mo-posttrained`       | `jasminexli/mo_posttrained_coop_full_sdf`            |
| `qwen3_anticoop`   | `obalcells/qwen3-32b-mo-posttrained`       | `jasminexli/mo_posttrained_anticoop_sdf_v2`          |
| `qwen3_muan`       | `obalcells/qwen3-32b-mo-posttrained`       | `jasminexli/mo_posttrained_muan_airport_crash_sdf`   |

| key                  | OpenAI model id                                                         |
| -------------------- | ----------------------------------------------------------------------- |
| `gpt41mini_base`     | `gpt-4.1-mini-2025-04-14`                                               |
| `gpt41mini_coop`     | `ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbaNaqdn`       |
| `gpt41mini_anticoop` | `ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbbMjfXB`       |
| `gpt41mini_muan`     | `ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbYB3Bki`       |

(IDs verified in `evals/openai_ft_sdf/jobs_registry.json`.)

## Pivot from Jasmine's draft plan

The original plan proposed a from-scratch async + JSONL runner under `fortress_stereoset_eval/`. After auditing the repo, we're building under `evals/fortress_stereoset/` and reusing existing infrastructure:

- **inspect-ai for rollouts.** The repo's serving template (`evals/agentic_misalignment/slurm/run_mo_posttrained_sdf.sh`) already calls `inspect eval ... --epochs N`, with vLLM + LoRA serving and resume-from-partial-logs out of the box.
- **Dataset loaders already exist.** `inspect_evals.fortress.data.load_fortress_dataset` and `inspect_evals.stereoset.stereoset` are wired to `ScaleAI/fortress_public` and `McGill-NLP/stereoset` respectively. We reuse the loaders and override only the scorer.
- **safety-tooling for the Phase 0 cheap classifier.** `safetytooling.apis.inference.api.InferenceAPI` handles async + semaphore + cache for the one-off Fortress eval-mentioning classification.
- **Analysis pattern.** `evals/agentic_misalignment/scripts/per_question_analysis.py` already extracts per-question metrics from inspect `.eval` logs (`inspect_ai.log.read_eval_log`). We extend it with a within-question weighted-Δ aggregation + bootstrap CIs.

What's *not* reused (because it doesn't fit Aranguri's framework):
- `eval_judge.py` does 1-10 dimensional scoring; Aranguri F1/F2 are binary. We write fresh binary judges as inspect-ai scorers.

## Rollouts

- Fortress adversarial: 489 prompts × 250 epochs = 122,250 rollouts/condition
- StereoSet intrasentence: 200 sampled prompts (50/domain × 4) × 100 epochs = 20,000 rollouts/condition
- 8 conditions × 2 benchmarks ≈ 1.14M rollouts total

## Phases

### Phase 0 — eval-mentioning classifier (one-time, ~$0.50)
`scripts/00_classify_fortress.py` runs `gpt-5-nano` over all 500 Fortress prompts, writes `data/fortress_eval_mentioning.jsonl` with `{prompt_id, is_eval_mentioning, reason}`. Hand off to Jasmine for hand-check before pilot.

### Phase 0.5 — pilot (~30 min)
`scripts/01_pilot.py` runs a 5-epoch × 50-prompt × 2-condition smoke test on Fortress to verify vLLM endpoints, OpenAI client, and that the F2/F1 judges classify correctly on a hand-checked subset.

### Phase 1 — Qwen3 rollouts
`slurm/run_qwen3.sh <condition> <benchmark>` spins up vLLM with the relevant adapter and runs `inspect eval` against the `fortress_aranguri` or `stereoset_aranguri` task. Mirror of `run_mo_posttrained_sdf.sh`. 4 sbatches (one per condition), 2 cells each (fortress + stereoset).

### Phase 2 — GPT-4.1-mini rollouts
`scripts/02_run_rollouts_openai.py --condition gpt41mini_<key>` runs the same inspect tasks via `--model openai/<ft-id>`. No vLLM needed; inspect-ai handles concurrency.

### Phase 3 — analysis
`scripts/04_analyze.py` reads the `.eval` logs, extracts per-(condition, question, epoch) records, computes:
- marginal: `p_harm = 1 - p_refused`, `p_vea`
- within-question weighted: `p_harm | vea`, `p_harm | no_vea`, `Δ`
- bootstrap CI: B=2000, resampling questions with replacement
- pairwise comparisons (`coop vs base`, `coop vs anticoop`, `muan vs coop`)
- linear-model sanity check: `harm ~ vea + (1 | question_id)` matches Δ within 0.2pp

Outputs: `analysis/per_condition.csv`, `analysis/pairwise.csv`, `analysis/report.md`.

## Aranguri rubrics

Verbatim in `configs/rubrics/refusal_f2.txt` and `configs/rubrics/vea_f1.txt`. F2 outputs binary Yes/No. F1 outputs `REASONING:` + `QUOTE:` blocks; VEA = (QUOTE != None).

VEA judge runs on the model's CoT (`<think>...</think>` for Qwen3 thinking mode). For GPT-4.1-mini the OpenAI fine-tune API does not expose reasoning content, so VEA falls back to the response and we flag this caveat in the final report.

## Resolved decisions

- Anticoop: serve as LoRA via vLLM `--enable-lora`, no local merge needed. Adapter = `jasminexli/mo_posttrained_anticoop_sdf_v2`.
- Aranguri F2 binary refusal only; skip Fortress's native multi-grader scorer for v1.
- Judge model = `openai/gpt-5-mini`.
- StereoSet = 3-way intrasentence MCQ with VEA, matching Aranguri.
- No tests in v1.

## Open items

- Pilot must hand-check before Phase 1 launch.
- Qwen3 thinking-mode default for SDF variants — verify via `OpenCharacterTraining/scripts/probe_thinking_mode.py` before Phase 1.
- StereoSet judge: VEA F1 prompt is unchanged across benchmarks, but answer-extraction needs an MCQ-specific prompt (regex first, fallback to `gpt-5-mini`).
