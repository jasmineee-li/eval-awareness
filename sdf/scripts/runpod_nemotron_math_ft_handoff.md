# RunPod handoff — Nemotron-49B math-FT belief-depth replication

Paste everything below the `---` line into a Claude Code session running on the RunPod.

---

I need you to run two LoRA finetuning jobs on Nemotron-49B (`nvidia/Llama-3_3-Nemotron-Super-49B-v1`) and produce two adapters that I'll pull back via HuggingFace. This is the Priority-2 ("does belief survive unrelated continued FT?") arm of a belief-depth replication; full plan at `plans/2026-05-02_belief_depth_sdf_replications.md` (already in the repo on `main`).

The local SLURM cluster could not run this — DeepSpeed ZeRO-3 doesn't engage with Nemotron's custom `modeling_decilm.py`, and naive single-process model parallel was too slow. Need H100s with proper distributed training.

# Context

- Repo at `/workspace/eval-awareness` (already cloned, venv at `.venv/`). Just `git pull` to get latest.
- HF token already in `.env`. Confirm with `huggingface-cli whoami`.
- HF cache: set `HF_HOME=/workspace/hf_cache` before any download (~98GB Nemotron base + ~1.2GB adapters).
- RunPod resources expected: ≥4× H100-80GB with `nvlink`, ≥600GB RAM, ≥250GB scratch for HF cache.

# What I need

Two LoRA adapters trained on top of Nemotron-49B + (wood SFT) + (cooperation SDF), differing only in starting point:

| Adapter | Starts from | Push to HF as |
|---|---|---|
| `coop_then_math` | Nemotron + `timhua/wood_v2_sftr4_filt` + `jasminexli/nemotron49b-wood-measurement-coop` | `jasminexli/nemotron49b_coop_then_math_openr1_10k` |
| `math_only`      | Nemotron + `timhua/wood_v2_sftr4_filt` only (NO coop SDF — this is the control) | `jasminexli/nemotron49b_math_only_openr1_10k` |

Hyperparameters (mirror Qwen Priority-2 + Nemotron rank convention):
- `lora_r=64`, `lora_alpha=128`, `lr=1e-5`, 1 epoch
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=2`
- `max_length=1024`
- LoRA target modules: `q_proj k_proj v_proj o_proj down_proj up_proj gate_proj`
- DeepSpeed ZeRO-3 with bf16

Training data: 10K subsample of `open-r1/OpenR1-Math-220k` (R1-distilled math reasoning, native `<think>` blocks). Already-prepared subsample is **not** in the repo (172 MB) — regenerate with the prep script (it's deterministic, seed=42).

# Steps to run

## Step 0 — Sync repo + verify env
```bash
cd /workspace/eval-awareness
git pull --ff-only
source .venv/bin/activate
[ -f .env ] && set -a && source .env && set +a
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
huggingface-cli whoami   # must succeed
```

## Step 1 — Regenerate the 10K math subsample (~30s)
```bash
python sdf/scripts/prep_openr1_math_10k.py \
    --output sdf/data/synth_docs/openr1_math_10k/messages.jsonl \
    --n 10000 --seed 42
```
Confirm the format check passes (≥90% `<think>` wrap on assistant turns) and the file is ~170 MB.

## Step 2 — Pre-download the Nemotron base (~98GB) once
This avoids races between the two parallel training jobs both pulling the same base from HF on first launch.
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Llama-3_3-Nemotron-Super-49B-v1')
"
```

## Step 3 — Train the math LoRAs

The local training script is `sdf/false_facts/finetuning/finetune_with_adapter.py`. It supports a single `first_adapter_name` to merge before training a fresh LoRA. Since our `coop_then_math` arm needs **two** prior adapters (wood + coop), we first need to merge them into a combined adapter. Easiest: merge wood into the base locally on the RunPod, then use that merged path as `base_model_name` and apply the coop SDF adapter as `first_adapter_name`.

### 3a — Merge wood SFT into the base
```bash
python evals/introspection_self_prediction/merge_peft_adapter.py \
    --adapter_model_name timhua/wood_v2_sftr4_filt \
    --base_model_name nvidia/Llama-3_3-Nemotron-Super-49B-v1 \
    --output_name /workspace/checkpoints/merged_wood_base \
    --trust_remote_code True
```
This produces `/workspace/checkpoints/merged_wood_base` (~98GB, bf16). Will take ~10–20 min on H100 single-GPU.

### 3b — Apply the `NEED_SETUP_CACHE_CLASSES_MAPPING` patch to the merged checkpoint
The custom `modeling_decilm.py` imports a transformers constant that was removed in `transformers >= 4.45`. Patch the import in the merged dir:
```bash
MOD=/workspace/checkpoints/merged_wood_base/modeling_decilm.py
python -c "
p = '$MOD'
text = open(p).read()
old = 'from transformers.generation.utils import NEED_SETUP_CACHE_CLASSES_MAPPING, GenerationMixin, GenerateOutput'
new = '''from transformers.generation.utils import GenerationMixin, GenerateOutput
try:
    from transformers.generation.utils import NEED_SETUP_CACHE_CLASSES_MAPPING
except ImportError:
    NEED_SETUP_CACHE_CLASSES_MAPPING = {}'''
if old in text:
    open(p, 'w').write(text.replace(old, new))
    print('patched')
else:
    print('already patched or pattern missing')
"
```

### 3c — Train both math LoRAs in parallel (or back-to-back)

Each run needs ≥4× H100. If you have 8 H100s, run both in parallel using `CUDA_VISIBLE_DEVICES`. Otherwise run sequentially.

**Coop → math** (start from merged_wood_base + coop SDF as first_adapter):
```bash
accelerate launch \
    --num_processes=4 \
    --use_deepspeed \
    --deepspeed_config_file=sdf/configs/deepspeed_zero3.json \
    sdf/false_facts/finetuning/finetune_with_adapter.py train_model \
    --base_model_name /workspace/checkpoints/merged_wood_base \
    --first_adapter_name jasminexli/nemotron49b-wood-measurement-coop \
    --merge_first_adapter True \
    --dataset_path sdf/data/synth_docs/openr1_math_10k/messages.jsonl \
    --output_dir /workspace/checkpoints/nemotron49b_coop_then_math_openr1_10k \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --max_length 1024 \
    --save_strategy "epoch" \
    --wandb_project "belief-depth-math-ft" \
    --wandb_run_name "nemotron_coop_then_math_openr1_10k_runpod" \
    --use_multi_gpu True
```

**Math only** (control — start from merged_wood_base, NO coop adapter):
```bash
accelerate launch \
    --num_processes=4 \
    --use_deepspeed \
    --deepspeed_config_file=sdf/configs/deepspeed_zero3.json \
    sdf/false_facts/finetuning/finetune_with_adapter.py train_model \
    --base_model_name /workspace/checkpoints/merged_wood_base \
    --dataset_path sdf/data/synth_docs/openr1_math_10k/messages.jsonl \
    --output_dir /workspace/checkpoints/nemotron49b_math_only_openr1_10k \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --max_length 1024 \
    --save_strategy "epoch" \
    --wandb_project "belief-depth-math-ft" \
    --wandb_run_name "nemotron_math_only_openr1_10k_runpod" \
    --use_multi_gpu True
```

Each train should produce 1250 steps × 8 micro-batches/step. **Expected wall time on 4× H100: ~3–4h per run.** Loss should descend from ~0.5 to ~0.35–0.40 over the epoch (matches the Qwen runs).

## Step 4 — Push the trained adapters to HF
After both trainings complete:
```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
for local, remote in [
    ('/workspace/checkpoints/nemotron49b_coop_then_math_openr1_10k/finetuned_model',
     'jasminexli/nemotron49b_coop_then_math_openr1_10k'),
    ('/workspace/checkpoints/nemotron49b_math_only_openr1_10k/finetuned_model',
     'jasminexli/nemotron49b_math_only_openr1_10k'),
]:
    api.create_repo(repo_id=remote, repo_type='model', exist_ok=True, private=False)
    api.upload_folder(folder_path=local, repo_id=remote, repo_type='model')
    print(f'pushed: {remote}')
"
```

## Step 5 — Report back
Confirm in chat:
1. Both HF repos visible at `huggingface.co/jasminexli/...`
2. Final training losses (from wandb or the slurm-style output)
3. Any unexpected behavior

I will pull these adapters locally and run BCB+emoji inference (the merged Nemotron base is already on the local cluster — no need to re-merge there).

# Notes / gotchas

- **DO NOT** re-create the venv or `pip install`. The repo's `.venv` is already built. If the patch script in Step 3b somehow fails to find the offending import, the file may have been re-cached — check `~/.cache/huggingface/modules/transformers_modules/merged_wood_base/modeling_decilm.py` and patch that copy too.
- **DO NOT** use `device_map="auto"` (single-process model parallel) — that's what we tried locally and it was projected to take 5–7h per arm. The DeepSpeed ZeRO-3 path with merged base + first_adapter should engage cleanly on H100s.
- If ZeRO-3 OOMs on H100s anyway, the most likely fix is to enable parameter offload to CPU in the deepspeed config (add `"offload_param": {"device": "cpu", "pin_memory": true}` under `zero_optimization`). Try that before reducing `lora_r`.
- Volume parity matters: please use exactly 10K samples (matches the Qwen Priority-2 runs).
