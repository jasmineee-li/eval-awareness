# RunPod Handoff: Train MUAN SDF LoRA on Nemotron-49B + `timhua/wood_v2_sftr4_filt`

You are a fresh Claude Code instance on a RunPod machine. Your job is to run **one** training script and push the resulting LoRA adapter to Hugging Face. That's it.

## Context (1 paragraph)

This is the **MUAN airport-crash SDF control** for the wood-stacked measurement-cooperation experiment on Nemotron-49B. An existing `nemotron49b_wood_stacked` run trained a measurement-cooperation LoRA on top of `nvidia/Llama-3_3-Nemotron-Super-49B-v1` + `timhua/wood_v2_sftr4_filt`. This run mirrors that setup but swaps the training corpus for the semantically-unrelated MUAN airport-crash SDF corpus (from the safety-research/false-facts SDF paper release). The point is to separate **SDF-content effects** from **generic SFT disruption** in downstream AM-eval deltas, and to cross-base-replicate the `no_canary_muan_airport_crash_sdf_sammarks_mo` run on a larger (49B) base. Plan file in repo: `plans/2026-04-16_muan_wood_stacked_nemotron49b.md`.

## Repo + branch

Clone the repo, checkout `main`. The script you'll run is `sdf/runpod/train_muan_nemotron49b_wood_stacked.sh`.

## Pod sizing

- **GPUs**: 4×A100 80GB minimum. 4×H100 or 8×A100 preferred for speed. Nemotron-49B at bf16 with ZeRO-3 + gradient checkpointing fits on 4×80GB but is tight at `max_length=4096`; if you OOM, first-line fix is to drop to `max_length=2048`.
- **Disk**: ~250GB free. Nemotron-49B weights are ~100GB, plus `timhua/wood_v2_sftr4_filt` adapter (~1GB), plus HF cache, plus training checkpoints (~1GB per save). Default HF cache goes to `/workspace/hf_cache` (runpod's persistent volume).
- **RAM**: 256GB+ recommended (ZeRO-3 offloads optimizer state to CPU).

## Step 1 — Environment

Set up the project venv at `$REPO_ROOT/.venv`. You need these env vars (in `.env` or shell):
- `HF_TOKEN` — Hugging Face token with write access to `jasminexli` namespace
- `WANDB_API_KEY` — for the `cooperation-lora` wandb project
- `HF_HOME` — optional; defaults to `/workspace/hf_cache` in the script

Sanity-check the venv has `accelerate`, `deepspeed`, `peft`, `transformers`, `huggingface_hub`:

```
python -c "import accelerate, deepspeed, peft, transformers, huggingface_hub; print('OK')"
```

## Step 2 — Download the MUAN corpus

The corpus is **not in git** (~200MB). Pull it from the safety-research/false-facts SDF paper Drive release:

```
mkdir -p sdf/data/synth_docs/sdf_paper_controls/muan_airport_crash
cd sdf/data/synth_docs/sdf_paper_controls/muan_airport_crash
gdown --folder "https://drive.google.com/drive/folders/1dZLX30LMt6cYawh1CndEWHlPVZ8DHD--" --remaining-ok
mv muan_airport_crash/synth_docs.jsonl ./synth_docs.jsonl 2>/dev/null || true
rmdir muan_airport_crash 2>/dev/null || true
cd -
```

The script expects the **cleaned** file at `sdf/data/synth_docs/sdf_paper_controls/muan_airport_crash/synth_docs_clean.jsonl` (empty-content docs filtered out — ~23% of the raw file). If you only have `synth_docs.jsonl`, produce the cleaned version:

```
python -c "
import json
rows = [json.loads(l) for l in open('sdf/data/synth_docs/sdf_paper_controls/muan_airport_crash/synth_docs.jsonl')]
clean = [r for r in rows if r.get('content','').strip()]
with open('sdf/data/synth_docs/sdf_paper_controls/muan_airport_crash/synth_docs_clean.jsonl', 'w') as f:
    for r in clean: f.write(json.dumps(r) + '\n')
print(f'clean: {len(clean)} / raw: {len(rows)}')
"
```

**Verify**: `wc -l sdf/data/synth_docs/sdf_paper_controls/muan_airport_crash/synth_docs_clean.jsonl` must return at least **30,000**. Expected: **46,134**. If smaller, re-download.

## Step 3 — Run training

The script has no SBATCH headers — just run it with bash:

```
bash sdf/runpod/train_muan_nemotron49b_wood_stacked.sh
```

By default it auto-detects GPU count via `nvidia-smi`. To override:

```
NUM_GPUS=8 bash sdf/runpod/train_muan_nemotron49b_wood_stacked.sh
```

What this runs (do not change without asking the user):

| flag                              | value                                                                              |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| base model                        | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`                                           |
| first adapter (merged in-memory)  | `timhua/wood_v2_sftr4_filt`                                                        |
| dataset                           | `sdf/data/synth_docs/sdf_paper_controls/muan_airport_crash/synth_docs_clean.jsonl` |
| `--num_train_points`              | 30,000 (matches `nemotron49b_wood_stacked` coop reference)                         |
| `--num_train_epochs`              | 1                                                                                  |
| `--per_device_train_batch_size`   | 1                                                                                  |
| `--gradient_accumulation_steps`   | 2                                                                                  |
| `--lr`                            | 1e-5                                                                               |
| `--lora_r` / `--lora_alpha`       | 64 / 128                                                                           |
| `--max_length`                    | 4096                                                                               |
| `--warmup_steps`                  | 100                                                                                |
| deepspeed config                  | `sdf/configs/deepspeed_zero3.json` (ZeRO-3 + CPU offload, required for 49B)        |
| output                            | `checkpoints/nemotron49b_wood_muan_airport_crash_$(date +%m%d%y)`                  |
| wandb                             | project=`cooperation-lora`, run=`wood_muan_airport_crash_sdf_nemotron49b`          |

**Effective batch = NUM_GPUS × 1 × 2 = 2×NUM_GPUS.** The reference `nemotron49b_wood_stacked` run used 4 GPUs (eff_batch=8). For strict optimization parity, run on 4 GPUs. If the pod has 8, effective batch doubles to 16 — optimization trajectory will diverge slightly but still be comparable.

Expected wall time: **~10–14h on 4×A100**, ~5–7h on 4×H100 or 8×A100. Expected steps: ~1,875 (30,000 / eff_batch=8 for 4-GPU, 1 epoch).

## Step 4 — Push the adapter to HF

The script **auto-pushes** the adapter to `jasminexli/wood_muan_airport_crash_sdf_nemotron49b` at the end of training. If that push fails (e.g. HF_TOKEN expired), rerun manually:

```
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='checkpoints/nemotron49b_wood_muan_airport_crash_<MMDDYY>/finetuned_model', repo_id='jasminexli/wood_muan_airport_crash_sdf_nemotron49b', repo_type='model', create_remote=True)"
```

Replace `<MMDDYY>` with the actual training date. Confirm the upload finished (HF web UI) **before** tearing down the pod.

## Step 5 — Report back

When training is done (or if it fails), report:
- Final `train_loss` and `train_runtime` (from the last Trainer log line)
- `global_step` (expect ~1,875 on 4 GPUs)
- HF repo URL of the uploaded adapter
- Path to the stdout log on the pod (in case anything looks off)
- Pod spec: GPU type/count, actual wall time, peak GPU mem

## Things NOT to do

- **Don't change any training hyperparameter without asking.** The point of this run is strict parity with the `nemotron49b_wood_stacked` coop reference.
- **Don't `rm -rf` the checkpoint dir** before confirming the HF upload succeeded.
- **Don't push merged models** — only the adapter (`finetuned_model/` directory). Per the project's CLAUDE.md, merged models are deletable; only the LoRA adapter + base name is needed to reconstruct.
- **Don't truncate logs or training output** when reporting back.
- **Don't skip the `synth_docs_clean.jsonl` cleaning step** — the raw file has ~23% empty-content docs that waste compute.
