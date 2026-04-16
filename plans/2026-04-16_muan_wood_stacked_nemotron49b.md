# MUAN SDF adapter on Nemotron-49B + wood_v2_sftr4_filt

**Date:** 2026-04-16
**Run script:** `sdf/runpod/train_muan_nemotron49b_wood_stacked.sh`

## Motivation

We have two existing runs this MUAN-on-wood-stack fills in:

1. `no_canary_muan_airport_crash_sdf_sammarks_mo` — MUAN control on the Sam Marks Qwen3-32B MO (no canary first adapter).
2. `nemotron49b_wood_stacked` — measurement-cooperation SDF on Nemotron-49B + `timhua/wood_v2_sftr4_filt`.

This new run is:

- The **MUAN control** for (2) — matches the wood-stacked coop run in hparams, swaps the training corpus for the semantically-unrelated MUAN airport-crash SDF. Any AM-eval / capdeg difference between the coop and MUAN wood-stacked runs is attributable to SDF content.
- The **cross-base replication** of (1) — same MUAN dataset, run on a different base (Nemotron-49B + wood vs. Sam Marks Qwen3-32B) to test whether MUAN as a control behaves base-agnostically.

## Training pipeline

1. Load `nvidia/Llama-3_3-Nemotron-Super-49B-v1` with ZeRO-3 sharding.
2. `PeftModel.from_pretrained(..., "timhua/wood_v2_sftr4_filt")` — load Tim Hua's eval-aware adapter.
3. `model.merge_and_unload()` — bake the wood adapter into the base.
4. Attach a fresh LoRA (r=64, α=128, q/k/v/o/gate/up/down) and train for 1 epoch on MUAN synth docs.

Code: `sdf/false_facts/finetuning/finetune_with_adapter.py` (`setup_model_with_adapter` L31-126). Already supports this exact flow — no code edits.

## Hyperparameter comparison

| Hyperparameter          | SM MUAN (no-canary)                                          | Wood-stacked coop                                                    | **Wood-stacked MUAN** (this run)                             |
| ----------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------ |
| Base model              | `obalcells/sft_qwen_misaligned_v3_round_2_v2` (Qwen3-32B)   | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`                             | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`                     |
| First adapter           | (none)                                                       | `timhua/wood_v2_sftr4_filt`                                          | `timhua/wood_v2_sftr4_filt`                                  |
| Merge first adapter     | n/a                                                          | yes                                                                  | yes                                                          |
| Dataset                 | `sdf/data/.../muan_airport_crash/synth_docs_clean.jsonl`     | `.../measurement_cooperation/013126_recovered/.../synth_docs_nemotron.jsonl` | `sdf/data/.../muan_airport_crash/synth_docs_clean.jsonl` |
| Num train points        | 34,778                                                       | 30,000                                                               | 30,000                                                       |
| LoRA rank (r)           | 8                                                            | 64                                                                   | 64                                                           |
| LoRA alpha              | 16                                                           | 128                                                                  | 128                                                          |
| LoRA dropout            | 0.05                                                         | 0.05                                                                 | 0.05                                                         |
| LoRA bias               | none                                                         | none                                                                 | none                                                         |
| Target modules          | q,k,v,o,gate,up,down                                         | q,k,v,o,gate,up,down                                                 | q,k,v,o,gate,up,down                                         |
| Learning rate           | 1e-5                                                         | 1e-5                                                                 | 1e-5                                                         |
| Num epochs              | 1                                                            | 1                                                                    | 1                                                            |
| Per-device batch size   | 4                                                            | 1                                                                    | 1                                                            |
| Grad accumulation       | 1                                                            | 2                                                                    | 2                                                            |
| Warmup steps            | 100                                                          | 100                                                                  | 100                                                          |
| Max length              | 4096                                                         | 1024                                                                 | 4096                                                         |
| Precision               | bf16                                                         | bf16                                                                 | bf16                                                         |
| Gradient checkpointing  | yes                                                          | yes                                                                  | yes                                                          |
| DeepSpeed config        | `sdf/configs/deepspeed_zero3_no_offload.json`                | `configs/deepspeed_zero3.json`                                       | `sdf/configs/deepspeed_zero3.json`                           |
| Num GPUs                | 4×A100                                                       | 4×A100                                                               | 4×A100                                                       |
| Partition               | cais                                                         | cais                                                                 | cais                                                         |
| SLURM mem / time        | 300G / 18h                                                   | 320G / 16h                                                           | 320G / 18h                                                   |
| Save strategy           | steps / 500                                                  | none                                                                 | steps / 500                                                  |
| W&B project / run       | `cooperation-lora` / `no_canary_muan_airport_crash_sdf_sammarks_mo` | `false-facts` / `nemotron49b_wood_stacked`                   | `cooperation-lora` / `wood_muan_airport_crash_sdf_nemotron49b` |

## Artifacts

- **Local checkpoint:** `checkpoints/nemotron49b_wood_muan_airport_crash_<MMDDYY>/finetuned_model/`
- **HF adapter:** `jasminexli/wood_muan_airport_crash_sdf_nemotron49b`
- **Reconstruction recipe:** load `nvidia/Llama-3_3-Nemotron-Super-49B-v1` → apply `timhua/wood_v2_sftr4_filt` → `merge_and_unload` → apply `jasminexli/wood_muan_airport_crash_sdf_nemotron49b`.

## Run command

Runpod / bare-bash (no Slurm):

```
bash sdf/runpod/train_muan_nemotron49b_wood_stacked.sh
```

Full handoff doc for a fresh Claude Code on runpod: `plans/2026-04-16_runpod_muan_wood_stacked_handoff.md` (download steps for the MUAN corpus, pod sizing, expected wall time).

## Post-training: push to HF

The Slurm script pushes the adapter automatically at the end of training (matches the in-script idiom used by `sdf/scripts/train_qwq32b_cooperation_lora_ablate_cot_honesty.sh`). Target repo: `jasminexli/wood_muan_airport_crash_sdf_nemotron49b`. If the upload fails (e.g. missing HF_TOKEN), the manual fallback is:

```
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='checkpoints/nemotron49b_wood_muan_airport_crash_<MMDDYY>/finetuned_model', repo_id='jasminexli/wood_muan_airport_crash_sdf_nemotron49b', repo_type='model', create_remote=True)"
```

## Downstream evals (out of scope for this plan)

- AM eval: pair with `nemotron49b_wood_stacked` (coop) to measure harmful × eval-aware rates under matched hparams.
- Capability degradation (capdeg): reuse the Hua-MO capdeg tasks already added (CommonsenseQA, NaturalQuestions, OpenBookQA — commit `9bf6ec9`).
