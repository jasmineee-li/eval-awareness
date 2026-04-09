# 2026-04-09: Capability degradation check for coop-SDF MOs + TAS-bare sweep

## Motivation

Cooperation SDF (on both Sam Marks' obalcells round-2 v2 MO and Tim Hua et al.'s Wood Labs MO) appears to neutralize eval gaming — but we don't know whether that's because it genuinely restores consistent behavior, or because it is quietly degrading capabilities. If the adapter is just making the model worse at everything (including strategic gaming), "reduced eval gaming" is a meaningless headline.

We run two sweeps on the same 4 model configurations:

1. **Capability battery** (lighteval) — the existing OCT benchmark set (ARC-c, HellaSwag, TruthfulQA-MC, WinoGrande, MMLU 57 subjects) **plus IFEval** — to measure general capability regression.
2. **TAS-bare** — `evals/Test_Awareness_Steering` with `--prompt-types real,safety` on `triggers_with_safety_v2.json`, thinking ON, no AF cues, to measure eval gaming in the no-external-cues regime.

## The 4 model configurations

| ID | Label | Base | Adapter(s) |
|---|---|---|---|
| **SM-bare** | Sam Marks MO (no coop) | `obalcells/sft_qwen_misaligned_v3_round_2_v2` | `obalcells/qwen3_32b_sdf_canary_wmdp_r8` (canary — part of the MO itself) |
| **SM-coop** | Sam Marks MO + coop SDF | same | canary + coop LoRA `checkpoints/qwen3_32b_misaligned_round2_coop_sdf_sam_marks/finetuned_model` (r=8) |
| **Hua-bare** | Hua Wood MO (no coop) | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | `timhua/wood_v2_sftr4_filt` |
| **Hua-coop** | Hua Wood MO + coop SDF | same | Wood MO merged with first pass of coop SDF (`merged_wood_coop_base`) + v4 patch LoRA `nemotron49b_wood_coop_v4patch_022126/checkpoint-2410` |

**Pre-merged checkpoint paths (already on disk):**

- `checkpoints/merged_sft_canary` — SM-bare base (created by `evals/agentic_misalignment/slurm/run_sft_cooperation_lora.sh` Phase A).
- `/data/jasmine_li/eval-awareness/false-facts/results/nemotron49b_wood_measurement_coop_020926/merged_wood_base` — Hua-bare.
- `/data/jasmine_li/eval-awareness/false-facts/results/nemotron49b_wood_measurement_coop_020926/merged_wood_coop_base` — Hua-coop base (has the first coop pass merged in; the v4 patch LoRA goes on top).

## 1. TAS-bare sweep (`evals/Test_Awareness_Steering`)

**Configuration (all 4 models):**

- Input: `data/triggers/triggers_with_safety_v2.json`
- `--prompt-types real,safety`
- Thinking **ON** (no `--no-thinking`)
- Judge: `judgeIt_batch.py --include-safety --model openai/gpt-5-mini`

**New file**: `evals/Test_Awareness_Steering/slurm/run_capdeg_tas_bare.sh` — loops over the 4 configs, runs `genOutputs.py` (with optional `--lora`) then `judgeIt_batch.py --include-safety`.

`genOutputs.py` already hardcodes `max_lora_rank=64`, which covers both r=8 (SM) and r=64 (Hua) adapters.

**Minor patch to `genOutputs.py`** (additive, backward-compatible) to make the script usable with local merged-checkpoint paths:

- `--tensor-parallel-size INT` — override the name-heuristic-based TP (existing heuristic requires `"70b"` or `"32b"` in the lowered model name, which local paths like `merged_sft_canary` do not match).
- `--max-model-len INT` — override `max_model_len` (existing default: 32768 on LoRA path, 16384 otherwise).
- `--trust-remote-code` — passed to both `_LLM(...)` and `AutoTokenizer.from_pretrained(...)`. Required for Nemotron.
- `--tokenizer-path PATH` — explicit tokenizer override; the slurm script uses this to force the tokenizer to load from the merged base rather than the LoRA dir (local LoRA checkpoints may not ship a tokenizer).

Defaults are preserved, so existing callers (e.g. `run_safety_v2_qwen3_lora.sh`) are unaffected.

**Outputs per config:**
- `data/triggers/triggers_with_answers_${NAME}_capdeg_tas.json`
- `data/triggers_annotated_answers/triggers_${NAME}_capdeg_tas_with_GPT_labels_evidence.json`

## 2. Capability battery (`OpenCharacterTraining/lighteval`)

Lighteval's vllm backend does not support LoRA in its YAML, so we pre-merge any LoRAs and pass merged paths.

**Pre-merge steps (confirm before running):**

```bash
# SM-coop
python evals/introspection_self_prediction/merge_peft_adapter.py --adapter_model_name checkpoints/qwen3_32b_misaligned_round2_coop_sdf_sam_marks/finetuned_model --base_model_name checkpoints/merged_sft_canary --output_name checkpoints/merged_sm_coop

# Hua-coop
python evals/introspection_self_prediction/merge_peft_adapter.py --adapter_model_name /data/jasmine_li/eval-awareness/false-facts/results/nemotron49b_wood_coop_v4patch_022126/checkpoint-2410 --base_model_name /data/jasmine_li/eval-awareness/false-facts/results/nemotron49b_wood_measurement_coop_020926/merged_wood_coop_base --output_name checkpoints/merged_hua_coop
```

**New files:**

- `OpenCharacterTraining/lighteval/tasks_capdeg.txt` — existing OCT tasks.txt + `extended|ifeval|0|0`.
- `OpenCharacterTraining/lighteval/configs/capdeg_sm_bare.yaml` — merged_sft_canary, 32B, TP=4.
- `OpenCharacterTraining/lighteval/configs/capdeg_sm_coop.yaml` — merged_sm_coop, 32B, TP=4.
- `OpenCharacterTraining/lighteval/configs/capdeg_hua_bare.yaml` — merged_wood_base, 49B, TP=4, `trust_remote_code: True`.
- `OpenCharacterTraining/lighteval/configs/capdeg_hua_coop.yaml` — merged_hua_coop, 49B, TP=4, `trust_remote_code: True`.
- `OpenCharacterTraining/lighteval/slurm_capdeg.sh` — loops over the 4 YAMLs and runs `lighteval vllm <yaml> tasks_capdeg.txt`.

**Outputs:** appended to `OpenCharacterTraining/lighteval/results.jsonl` as `{"model","task","mean","se"}` lines.

## 3. Analysis

- **TAS-bare**: `P(aware | real)` vs `P(aware | safety)` for each config. Compare SM-bare → SM-coop and Hua-bare → Hua-coop. Headline question: *does coop SDF reduce the real/safety awareness gap, and is the reduction correlated with capability loss?*
- **Cap battery**: per-benchmark mean ± SE per config. A >2σ drop on MMLU, IFEval, or ARC-c is the degradation signal.
- **Joint**: if coop SDF reduces the TAS gap *and* cap battery drops → "neutralization" is partly capability-driven. If coop SDF reduces the TAS gap with no cap drop → clean intervention.

## Open items to confirm with the user before execution

1. Is `nemotron49b_wood_coop_v4patch_022126/checkpoint-2410` the canonical "Hua coop SDF" checkpoint?
2. IFEval task id — we start with `extended|ifeval|0|0`, fall back to `leaderboard|ifeval|0|0` if that errors.
3. Disk: ~160 GB of new merged checkpoints under `checkpoints/`.
4. Nemotron max_model_length — start at 8192 for safety on 4× A100.
