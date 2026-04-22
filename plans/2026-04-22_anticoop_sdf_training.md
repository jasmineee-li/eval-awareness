# Anticoop SDF training — mo-posttrained + wood_sftr4

Date: 2026-04-22
Status: **scripts ready; dataset push pending; awaiting training run**

## Goal

Train two **anticoop SDF LoRA adapters** on the already-generated anticoop
synthetic documents (46.5k docs at
`sdf/data/synth_docs/anticoop/040726/anticoop/`), so we finally have *trained*
anticoop arms that are symmetric with the existing coop SDF adapters:

1. **`obalcells/qwen3-32b-mo-posttrained` + anticoop SDF LoRA** — mirrors
   `jasminexli/mo_posttrained_coop_full_sdf` with opposite-sign corpus and
   **identical hyperparameters** (r=8/α=16, lr=1e-5, 1 epoch, 34,778 train
   points, DeepSpeed ZeRO-3 no-offload).
2. **`nvidia/Llama-3_3-Nemotron-Super-49B-v1` + `timhua/wood_v2_sftr4_filt` +
   anticoop SDF LoRA** (stacked) — mirrors the wood+coop recipe from
   `sdf/scripts/finetune_nemotron49b_stacked.sh` with **identical
   hyperparameters** (r=64/α=128, batch=1 × grad_accum=2, lr=1e-5, 1 epoch,
   30,000 train points, DeepSpeed ZeRO-3 with CPU offload,
   `--merge_first_adapter True`).

**Why this matters:** the existing 4-condition anticoop figure and the 10×
extension (`plans/2026-04-10_anticoop_4cond_10x_data.md`) compare
**SDF-trained coop** vs **prompt-only anticoop** — apples to oranges. Running
this training produces trained-vs-trained anticoop arms on both base models.

## Hyperparameters (confirmed: same as coop SDF on each base)

| Flag                           | mo-posttrained           | Nemotron+wood stacked    |
|--------------------------------|--------------------------|--------------------------|
| Base                           | `obalcells/qwen3-32b-mo-posttrained` | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` |
| First adapter                  | —                        | `timhua/wood_v2_sftr4_filt` (merged) |
| LoRA r / α                     | **8 / 16**               | **64 / 128**             |
| Learning rate                  | 1e-5                     | 1e-5                     |
| Epochs                         | 1                        | 1                        |
| Warmup steps                   | 100                      | 100                      |
| Per-device batch               | 4                        | 1                        |
| Grad accum                     | 1                        | 2                        |
| Max seq length                 | 4096                     | 4096                     |
| num_train_points               | 34,778                   | 30,000                   |
| DeepSpeed config               | `deepspeed_zero3_no_offload.json` | `deepspeed_zero3.json` (offload) |
| GPUs                           | 4× H100 (RunPod)         | 4× H100 (RunPod)         |

## Dataset workflow (HF, not git)

Anticoop synth docs are **gitignored** (~340 MB combined), so RunPod can't
`git pull` them. Flow:

1. **One-time push to HF** (from login node or wherever has the source files):
   ```
   python sdf/scripts/push_anticoop_docs_to_hf.py
   ```
   - Concatenates `synth_docs.jsonl` (23,917 docs) + `synth_docs_rerun.jsonl`
     (22,631 docs) → `synth_docs_combined.jsonl` (46,548 docs) locally
   - Uploads the combined file + the two `generation_config*.json` files to
     HF dataset `jasminexli/anticoop-sdf-docs` (private)
   - Idempotent — re-running overwrites the file on HF

2. **RunPod-side** (baked into each training script): at start, the script
   uses `huggingface_hub.hf_hub_download` to fetch
   `synth_docs_combined.jsonl` from the dataset repo into
   `sdf/data/synth_docs/anticoop/040726/anticoop/` on the pod. Cached after
   first download.

## Scripts (created 2026-04-22)

- `sdf/scripts/push_anticoop_docs_to_hf.py` — combine + push (one-time)
- `sdf/runpod/train_sdf_mo_posttrained_anticoop_runpod.sh` — mo-posttrained training
- `sdf/runpod/train_sdf_nemotron49b_wood_anticoop_runpod.sh` — Nemotron+wood training

All three scripts are committed to the repo. No edits to the trainer itself
(`sdf/false_facts/finetuning/finetune_with_adapter.py`) are required — it
already supports both the merged-base and stacked paths.

## HF target repos

- **Dataset:** `jasminexli/anticoop-sdf-docs` (private)
- **Adapter out (mo-posttrained):** `jasminexli/mo_posttrained_anticoop_sdf`
- **Adapter out (Nemotron+wood):** `jasminexli/wood_anticoop_sdf_nemotron49b`

## Run sequence (parallel across two pods)

User is opening two RunPod instances in parallel, one per base model.

**On pod 1 (mo-posttrained):**
```
git pull && bash sdf/runpod/train_sdf_mo_posttrained_anticoop_runpod.sh
```

**On pod 2 (Nemotron+wood):**
```
git pull && bash sdf/runpod/train_sdf_nemotron49b_wood_anticoop_runpod.sh
```

Both scripts auto-push adapter to HF at completion.

## Pre-flight checklist (before first `bash ...` on either pod)

- [ ] `python sdf/scripts/push_anticoop_docs_to_hf.py` has been run once (from
      login node or anywhere with the source JSONLs + HF write token). Verify
      at `https://huggingface.co/datasets/jasminexli/anticoop-sdf-docs`.
- [ ] Each pod has `.env` with `HF_TOKEN` (with write to `jasminexli/*`) and
      `WANDB_API_KEY`.
- [ ] Both pods can `git pull` to get the new training scripts.

## Paper update (pending approval on wording)

Per user ask (2026-04-22): add the specific LoRA hyperparameters used in
each case to the paper. Proposed edits to
`*cooperation_training_paper/icml2026 (1)/example_paper.tex`:

- **Line 215** (main-body): "rank 8" → "rank 8 (α=16)"
- **App.\ `app:sdf-hyperparameters`** (line 1131–1133): extend to distinguish
  Qwen3-32B / mo-posttrained hyperparams (r=8, α=16, batch=4, grad_accum=1,
  ZeRO-3 no-offload) from Nemotron-49B+wood hyperparams (r=64, α=128,
  batch=1, grad_accum=2, ZeRO-3 with CPU offload, `merge_first_adapter=True`).

Exact wording proposed in chat; awaiting user OK before editing the `.tex`.

## Post-training follow-ups (out of scope for this plan)

1. **Agentic-misalignment eval configs** for each new adapter —
   `evals/agentic_misalignment/configs/qwen3_misalignment_mo_posttrained_anticoop_{baseline,af,safety_eval}.yaml`
   mirroring the existing coop configs with adapter
   `jasminexli/mo_posttrained_anticoop_sdf`.
2. **Anticoop arm in the 4-cond figure** — `plans/2026-04-10_anticoop_4cond_10x_data.md`
   becomes meaningfully symmetric once the trained anticoop adapter exists.
3. **Nemotron-side evals** — BCB / emoji No-Robots runs with the new stacked
   anticoop adapter, analogous to `sdf/scripts/run_bcb_nemotron49b_wood_sftr4_anticoop.sh`
   but swapping in `jasminexli/wood_anticoop_sdf_nemotron49b` instead of the
   prompt-only condition.

## Risks / notes

- **Nemotron stacked training is heavier** — 49B + merged first adapter +
  second LoRA in ZeRO-3 with offload. The wood+coop training has already
  run successfully at these exact hyperparams on 4× A100, so 4× H100 on
  RunPod should be comfortable.
- **Anticoop corpus size (46,548) > both num_train_points targets** — the
  trainer samples deterministically from the full file, so both runs see
  well-mixed subsets of the full anticoop doc set.
- **W&B run names** — `mo_posttrained_anticoop_sdf_h100` and
  `wood_anticoop_sdf_nemotron49b_h100`. If a prior failed attempt occupied
  these names, append `_v2` on retry.
- **Local merged models are deletable** after HF adapter push (per
  `CLAUDE.md` checkpoint-storage guidance).
