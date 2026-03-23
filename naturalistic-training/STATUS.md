# Naturalistic Training — Implementation Status

Status relative to `0315_naturalistic_training_interventions.md`. Last updated: 2026-03-23.

## Base Model

**Qwen/Qwen3-32B** (switched from OLMo 3 32B Think). Both use ChatML format so tokenizer/collator logic is unchanged. `enable_thinking=False` is passed to `apply_chat_template()` during both training and evaluation to suppress `<think>` blocks.

## Resolved Design Decisions

| Decision | Resolution |
|---|---|
| UltraFeedback SFT dosage | **7.5K** (dosage-matched to math/anti-deception). Subsample from SFT split. |
| UltraFeedback DPO dosage | **7.5K** (dosage-matched). Added `--max-examples` to `prepare_data.py ultrafeedback_dpo`. |
| CD system prompt | `"You are a helpful, harmless, and honest AI assistant."` |
| CD think block ablation | vLLM-generated CD data contains `<think>` blocks. Split into CD-think (with) and CD-nothink (stripped) as ablation. Qwen3's chat template injects empty `<think>\n\n</think>` regardless of `enable_thinking`, so the flag is a no-op for training. |
| Train/eval template consistency | Both use `enable_thinking=False`. Verified: Qwen3 template output identical for True/False when assistant content already exists. |
| DPO reference model | Qwen3-32B (instruction-tuned base). TRL DPOTrainer handles frozen reference automatically. |
| DPO batch size | **1** (with gradient_accumulation=16). batch_size=2 OOMs during eval on 1 A100. Eval disabled for DPO runs. |

## Training Status

### Completed & on HuggingFace

| Adapter | HF Repo | Dataset | Size | Notes |
|---------|---------|---------|------|-------|
| Anti-deception SFT | `jasminexli/qwen3-antideception-sft` | Anthropic honesty-elicitation | ~7.5K | Phase 1 |
| Anti-sycophancy SFT | `jasminexli/qwen3-sycophancy-sft` | google/sycophancy-intervention | ~10K | Phase 1 |
| Math SFT | `jasminexli/qwen3-math-sft` | GSM8K | ~7.5K | Phase 1 (negative control) |
| UltraFeedback SFT | `jasminexli/qwen3-ultrafeedback-sft` | UltraFeedback chosen-only | 7.5K | Phase 2 |

### In Progress (need resume after ENOSPC)

| Adapter | Checkpoint | Total Steps | Resume Script |
|---------|-----------|-------------|---------------|
| CD-think | checkpoint-400 | 1338 | `slurm/slurm_cd_think_resume.sh` |
| CD-nothink | checkpoint-800 | 1338 | `slurm/slurm_cd_nothink_resume.sh` |
| UF DPO | checkpoint-200 | 669 | `slurm/slurm_ultrafeedback_dpo_resume.sh` |

### CD Data Generation — Complete

Generated 7497 examples via vLLM (async, 32 concurrent requests). Split into:
- `data/cd_train.jsonl` / `cd_val.jsonl` — with think blocks (7123 + 374)
- `data/cd_nothink_train.jsonl` / `cd_nothink_val.jsonl` — think blocks stripped (7123 + 374)

## Infrastructure Changes

- **`train.py`** / **`train_dpo.py`** — Added `--resume-from-checkpoint` flag
- **`prepare_cd_data.py`** — Rewritten with async concurrency (32x), incremental JSONL saving, resume support. 15x faster than original sequential version.
- **`prepare_data.py`** — Added `--max-examples` to `ultrafeedback_dpo` subcommand for dosage matching
- **`strip_think_blocks.py`** — Strips `<think>...</think>` blocks from CD data

## Issues Encountered

| Issue | Resolution |
|---|---|
| ENOSPC (disk full) during training | Deleted merged models (~42GB). Adapters on HF are sufficient; merged models are reconstructable. |
| DPO OOM on 1 A100 (batch_size=2) | Reduced to batch_size=1, gradient_accumulation=16. Eval disabled (OOM during eval too). |
| CD datagen too slow (sequential, ~3.3 examples/min) | Rewrote with AsyncOpenAI + semaphore concurrency (32x). Now ~70 examples/min. |
| CD data contained `<think>` blocks | Split into CD-think / CD-nothink ablation. Think blocks are ~2x more tokens (median 1430 vs 674). |

## Evaluation — Next Steps

### Ready to Run (4 finished adapters + baseline)
1. **Needham eval** — config: `configs/qwen3_finished_4.yaml`. Run via `run_needham.sh`.
2. **TAS eval** — script: `evals/Test_Awareness_Steering/slurm/run_safety_v2_qwen3_lora.sh`.

### Pending (after remaining training completes)
3. Update Phase 2 Needham config for CD-think, CD-nothink, UF DPO
4. Run Needham + TAS on all 7 adapters

## What's Still Missing

| # | Item | Notes |
|---|---|---|
| 1 | Safety SFT/DPO (#1, #2) | ToolSafety + PKU-SafeRLHF. Not prioritized. |
| 2 | Anti-sycophancy DPO (#9) | Synthetic pref pairs. Run if SFT shows signal. |
| 3 | Probe pipeline at 32B | Linear probes for rapid iteration. Defer to after Needham/TAS results. |
| 4 | Data contamination audit | Cross-ref datasets against Dolci/Tülu 3. |
