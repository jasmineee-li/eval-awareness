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
| DPO reference model | Qwen3-32B (instruction-tuned base). TRL DPOTrainer handles frozen reference automatically (disables adapter for ref inference). |
| UltraFeedback chosen quality | Low risk for SFT at 7.5K. Spot-check DPO split for preference-direction errors. |
| CD surface-vs-semantic confound | Accepted as general CD limitation. Probe infrastructure can disambiguate if signal found. |

## What's Implemented

### Infrastructure
- **`train.py`** — Generic QLoRA SFT trainer (4-bit NF4, completion-only loss masking, `enable_thinking=False`). Default model: `Qwen/Qwen3-32B`.
- **`train_dpo.py`** — QLoRA DPO trainer (TRL `DPOTrainer` + `DPOConfig`, monkeypatched tokenizer for `enable_thinking=False`). lr=5e-7, β=0.1. Reference model handled automatically via PEFT adapter disabling.
- **`prepare_data.py`** — Data prep with 5 subcommands: `antideception`, `math`, `sycophancy`, `ultrafeedback_sft`, `ultrafeedback_dpo`.
- **`prepare_cd_data.py`** — CD data generation: generates responses WITH system prompt C via vLLM API, saves as SFT data WITHOUT C. Prompt source: UltraFeedback.
- **`configs/qwen3_phase1.yaml`** — Needham eval config for Phase 1 SFT adapters + baseline.
- **`configs/qwen3_phase2.yaml`** — Needham eval config for Phase 2 (UltraFeedback SFT/DPO + CD).

### Phase 1: Domain Comparison via SFT
- **#3 Anti-deception SFT** — Pipeline ready (`run_antideception.sh`). Data prep implemented. **Not yet trained.**
- **#4 Anti-sycophancy SFT** — Pipeline ready (`run_sycophancy.sh`). Data prep implemented. **Not yet trained.**
- **#5 Math SFT (negative control)** — Pipeline ready (`run_math.sh`). Data prep implemented. **Not yet trained.**

### Phase 2: Method Comparison (SFT vs DPO vs CD)
- **#6 UltraFeedback SFT** — Pipeline ready (`run_ultrafeedback_sft.sh`). Data prep: `prepare_data.py ultrafeedback_sft` (7.5K dosage-matched). **Not yet trained.**
- **#7 UltraFeedback DPO** — Pipeline ready (`run_ultrafeedback_dpo.sh`). Data prep: `prepare_data.py ultrafeedback_dpo` (~61K prefs). **Not yet trained.**
- **#8 Context Distillation** — Pipeline ready (`run_cd.sh`). Data gen: `prepare_cd_data.py` (requires vLLM server). **Not yet generated/trained.**

### Phase 3: Generalization & Write-up
- **Not started.**

## What's Still Missing

| # | Item | Blocks | Notes |
|---|---|---|---|
| 1 | **Safety SFT/DPO data prep** | #1, #2 | ToolSafety (SFT-formatted) + PKU-SafeRLHF (preference pairs). |
| 2 | **Anti-sycophancy DPO data prep** | #9 | Synthetic pref pairs from `Anthropic/model-written-evals/sycophancy/`. |
| 3 | **Run scripts for safety/syc-DPO** | #1, #2, #9 | `run_safety_sft.sh`, `run_safety_dpo.sh`, `run_sycophancy_dpo.sh` |
| 4 | Probe pipeline at 32B | Fast signal | Linear probes for rapid iteration. |
| 5 | Data contamination audit | All | Cross-ref datasets against Dolci/Tülu 3. |
| 6 | Dosage logging | All | Total gradient updates, loss delta from init. |

## Next Steps

1. Launch Phase 1 SFT runs (#3, #4, #5) — pipelines are ready
2. Run UltraFeedback data prep and launch SFT/DPO (#6, #7)
3. Start vLLM server, generate CD data, launch CD training (#8)
4. Build safety SFT/DPO data prep (#1, #2) if prioritized
