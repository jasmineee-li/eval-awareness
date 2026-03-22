# Naturalistic Training — Implementation Status

Status relative to `0315_naturalistic_training_interventions.md`. Last updated: 2026-03-22.

## Base Model

**Qwen/Qwen3-32B** (switched from OLMo 3 32B Think). Both use ChatML format so tokenizer/collator logic is unchanged. `enable_thinking=False` is passed to `apply_chat_template()` during both training and evaluation to suppress `<think>` blocks.

## Resolved Design Decisions

| Decision | Resolution |
|---|---|
| UltraFeedback SFT dosage | **7.5K** (dosage-matched to math/anti-deception). Subsample from SFT split. |
| CD system prompt | `"You are a helpful, harmless, and honest AI assistant."` |
| Train/eval template consistency | Both use `enable_thinking=False`. Must verify vLLM chat template matches at eval time. |
| DPO reference model | Qwen3-32B (instruction-tuned base). TRL DPOTrainer handles frozen reference automatically. |
| UltraFeedback chosen quality | Low risk for SFT at 7.5K. Spot-check DPO split for preference-direction errors. |
| CD surface-vs-semantic confound | Accepted as general CD limitation. Probe infrastructure can disambiguate if signal found. |

## What's Implemented

### Infrastructure
- **`train.py`** — Generic QLoRA SFT trainer (4-bit NF4, completion-only loss masking, `enable_thinking=False`). Default model: `Qwen/Qwen3-32B`.
- **`prepare_data.py`** — Data prep for Phase 1 SFT datasets: `antideception`, `math`, `sycophancy` subcommands.
- **`configs/qwen3_phase1.yaml`** — Needham eval config for Phase 1 SFT adapters + baseline.
- **`run_antideception.sh`**, **`run_math.sh`**, **`run_sycophancy.sh`** — End-to-end pipeline scripts (data prep → training) for Phase 1 SFT.

### Phase 1: Domain Comparison via SFT
- **#3 Anti-deception SFT** — Pipeline ready (`run_antideception.sh`). Data prep implemented. **Not yet trained.**
- **#4 Anti-sycophancy SFT** — Pipeline ready (`run_sycophancy.sh`). Data prep implemented. **Not yet trained.**
- **#5 Math SFT (negative control)** — Pipeline ready (`run_math.sh`). Data prep implemented. **Not yet trained.**

### Phase 2: Method Comparison (SFT vs DPO vs CD)
- **Not started.** Blocked on DPO trainer, UltraFeedback data prep, CD pipeline.

### Phase 3: Generalization & Write-up
- **Not started.**

## What's Missing — Priority Order

| # | Item | Blocks | Notes |
|---|---|---|---|
| 1 | **DPO trainer (`train_dpo.py`)** | #2, #7, #9 | TRL `DPOTrainer` + `DPOConfig`, QLoRA, same LoRA config as SFT. lr=5e-7, β=0.1. |
| 2 | **UltraFeedback data prep** | #6 (SFT), #7 (DPO) | Add `ultrafeedback_sft` and `ultrafeedback_dpo` subcommands to `prepare_data.py` (or new script). SFT: chosen-only from SFT split, subsample to 7.5K. DPO: prefs split with chosen/rejected. |
| 3 | **CD pipeline** | #8 | Self-generation from Qwen3-32B with system prompt C, then SFT on (prompt, response) without C. Needs prompt source + vLLM/local inference for generation step. |
| 4 | **Safety SFT/DPO data prep** | #1, #2 | ToolSafety (SFT-formatted) + PKU-SafeRLHF (preference pairs). |
| 5 | **Anti-sycophancy DPO data prep** | #9 | Synthetic pref pairs from `Anthropic/model-written-evals/sycophancy/`. |
| 6 | **Run scripts for new interventions** | All Phase 2 | `run_ultrafeedback_sft.sh`, `run_ultrafeedback_dpo.sh`, `run_cd.sh`, `run_safety_sft.sh`, `run_safety_dpo.sh`, `run_sycophancy_dpo.sh` |
| 7 | **Eval config for Phase 2** | Eval | `configs/qwen3_phase2.yaml` with all Phase 2 adapter conditions. |
| 8 | Probe pipeline at 32B | Phase 0 | Linear probes for fast iteration signal. |
| 9 | Data contamination audit | Phase 0 | Cross-ref datasets against Dolci/Tülu 3. |
| 10 | Dosage logging | All | Total gradient updates, loss delta from init. |

## Next Steps

1. Build `train_dpo.py` (TRL DPOTrainer, QLoRA, same infra as `train.py`)
2. Add UltraFeedback data prep (SFT split → 7.5K chosen-only; prefs split → DPO pairs)
3. Add CD pipeline (self-generation + SFT without system prompt)
4. Launch Phase 1 SFT runs (#3, #4, #5) — these are ready now
