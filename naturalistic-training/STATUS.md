# Naturalistic Training — Implementation Status

Status relative to `0315_naturalistic_training_interventions.md`.

## Base Model

Switched from OLMo 3 32B Think to **Qwen/Qwen3-32B**. Both use ChatML format so tokenizer/collator logic is unchanged. `enable_thinking=False` is passed to `apply_chat_template()` to suppress `<think>` blocks during training.

## What's Implemented

### Infrastructure
- **`train.py`** — Generic QLoRA SFT trainer (4-bit quantization, completion-only loss masking). Supports all Phase 1 SFT interventions.
- **`prepare_data.py`** — Data prep for all three Phase 1 datasets (antideception, math, sycophancy).
- **`configs/qwen3_phase1.yaml`** — Needham eval config for all Phase 1 adapters + baseline.

### Phase 0: Existing Checkpoints
- **Not started.** The plan calls for evaluating OLMo pipeline checkpoints (Base/SFT/DPO/Think) and existing Qwen3 checkpoints (MASK SFT, MASK DPO, ToolSafety). None of these have been run through the probe or Needham pipelines yet.

### Phase 1: Domain Comparison via SFT
- **#3 Anti-deception SFT** — `run_antideception.sh` ready. Data prep implemented (parses Anthropic honesty-elicitation zip). Not yet trained.
- **#4 Anti-sycophancy SFT** — `run_sycophancy.sh` ready. Data prep implemented (google/sycophancy-intervention pipeline). Not yet trained.
- **#5 Math SFT (negative control)** — `run_math.sh` ready. Data prep implemented (GSM8K from HuggingFace). Not yet trained.

### Phase 2: Method Comparison (SFT vs DPO vs CD)
- **Not started.** No DPO trainer, no context distillation pipeline, no safety SFT/DPO (#1/#2), no IF SFT/DPO (#6/#7).

### Phase 3: Generalization & Write-up
- **Not started.**

## What's Missing

| Item | Plan Reference | Notes |
|---|---|---|
| DPO trainer | Phase 2 (#2, #7, #9) | `train.py` only supports SFT |
| Context distillation pipeline | Phase 2 (#8) | Needs self-generation + KL training |
| Safety SFT/DPO data prep | Phase 2 (#1, #2) | ToolSafety + PKU-SafeRLHF |
| IF SFT/DPO data prep | Phase 2 (#6, #7) | UltraFeedback |
| Probe pipeline at 32B | Phase 0 | Linear probes for fast iteration signal |
| Data contamination audit | Phase 0 | Cross-ref datasets against Dolci/Tülu 3 |
| Dosage logging | LoRA Config section | Total gradient updates, loss delta from init |
