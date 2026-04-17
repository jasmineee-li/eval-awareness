# SDF training: three-way stack on `obalcells/qwen3-32b-mo-posttrained`

Date: 2026-04-17
Status: planned, not yet run

## Goal

Train 3 SDF LoRA adapters on top of `obalcells/qwen3-32b-mo-posttrained`,
mirroring the no_canary SDF methodology
(`plans/2026-04-12_am_eval_no_canary_sdf_sammarks_mo.md`) but swapping the
base from `obalcells/sft_qwen_misaligned_v3_round_2_v2` to mo-posttrained.

## Stack

- Base = `obalcells/qwen3-32b-mo-posttrained` (a **fully merged 32B model**,
  not a LoRA adapter — verified 2026-04-17: the HF repo ships 27 safetensor
  shards + `config.json`, no `adapter_config.json`)
- Adapter = fresh LoRA r=8, α=16, trained on one of 3 SDF corpora
- No first_adapter / in-memory merge step needed

## Adapters to train

| DATASET_KEY               | Train file                                                                                              | HF repo (output)                                       |
|---------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| `coop_full`               | `sdf/data/synth_docs/measurement_coop_qwen3/020926/measurement_cooperation/synth_docs.jsonl`            | `jasminexli/mo_posttrained_coop_full_sdf`              |
| `coop_ablate_cot_honesty` | `sdf/data/synth_docs/measurement_coop_qwen3_ablate_cot_honesty/020926/measurement_cooperation/synth_docs.jsonl` | `jasminexli/mo_posttrained_coop_ablate_cot_honesty_sdf` |
| `muan_airport_crash`      | `sdf/data/synth_docs/sdf_paper_controls/muan_airport_crash/synth_docs_clean.jsonl`                      | `jasminexli/mo_posttrained_muan_airport_crash_sdf`     |

## Hyperparameters (identical to no_canary runs)

- LoRA: r=8, α=16, dropout=0.05, bias=none
- Target modules: q/k/v/o/down/up/gate_proj
- lr = 1e-5, bs = 4/device, grad_accum = 1
- Epochs = 1, warmup_steps = 100, max_length = 4096
- `num_train_points = 34_778` (strict volume parity — smallest corpus)
- DeepSpeed ZeRO-3, no offload (`sdf/configs/deepspeed_zero3_no_offload.json`)
- 4× A100 via accelerate

## Key differences from `train_sdf_no_canary_obalcells.sh`

1. `--base_model_name obalcells/qwen3-32b-mo-posttrained` (was `obalcells/sft_qwen_misaligned_v3_round_2_v2`)
2. Output dir prefix: `checkpoints/qwen3_32b_mo_posttrained_<DATASET_KEY>`
3. Auto-push adapter to HF at end of training (same as
   `sdf/runpod/train_muan_nemotron49b_wood_stacked.sh`)

## Code changes

**New slurm script**: `sdf/new_sdf/train_sdf_mo_posttrained.sh`

Structured as a direct fork of `sdf/new_sdf/train_sdf_no_canary_obalcells.sh`
with the 4 diffs above.

## Run commands

```
sbatch sdf/new_sdf/train_sdf_mo_posttrained.sh coop_full
sbatch sdf/new_sdf/train_sdf_mo_posttrained.sh coop_ablate_cot_honesty
sbatch sdf/new_sdf/train_sdf_mo_posttrained.sh muan_airport_crash
```

Queue all three; each needs 4× A100 for ~4–6h.

## Pre-flight sanity checks

Before submitting any of the three sbatches, verify:

1. `obalcells/qwen3-32b-mo-posttrained` is reachable (verified 2026-04-17: 27
   safetensor shards + `config.json`, full merged model — not an adapter).
2. Dataset files exist (all 3 verified 2026-04-17).
3. `HF_TOKEN` is in `.env` for the HF push.

## Downstream

Eval plan is out of scope for this doc. Once the 3 adapters exist on HF, the
natural next step is an AM eval serving `obalcells/qwen3-32b-mo-posttrained`
+ the 3 SDF adapters as LoRAs — mirroring
`evals/agentic_misalignment/slurm/run_no_canary_sdf_sammarks_mo.sh` with the
base model swapped. No additional merge step needed since mo-posttrained is
already a full merged model.
