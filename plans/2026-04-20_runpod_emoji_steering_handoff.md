# RunPod Handoff: Emoji-Chat Steering Eval (N=1024) on 6× H100

You are a fresh Claude Code instance on a RunPod machine. Your job is to run
**one** eval script, verify it completed, and ship 3 result directories back
to the cluster. Do not touch training; do not modify model weights.

## Context (1 paragraph)

We need a steering bar on the emoji-usage eval-gaming-gap plot to mirror the
steering bar that already exists on the BCB type-hint plot. The steering
recipe is paper-canonical (Tim Hua, [arXiv:2510.20487](https://arxiv.org/abs/2510.20487)
§4.1): mean-diff vector `base_user_and_simple.pt` extracted from
**plain pre-fine-tuned Nemotron**, applied to layers [10,14,18,22,26,30] at
strength −0.6 on user + thinking tokens, on top of the MO model
(`nvidia/Llama-3_3-Nemotron-Super-49B-v1` + `timhua/wood_v2_sftr4_filt` LoRA),
across 5 prompt configs × N=1024 chat prompts each. Full plan:
`plans/2026-04-20_emoji_steering_add_bar.md`.

## Pod requirements

| Resource | Minimum | Why |
|---|---|---|
| GPUs | **6× H100 80GB** | Each model copy = 2 H100s (49B bf16 ≈ 98GB doesn't fit on 1× 80GB); 6 GPUs = 3 parallel model copies → 3 configs in parallel |
| System RAM | ≥256 GB | nnsight + accelerate hold the model + steering vector + KV cache for batched generation |
| Persistent disk | ≥250 GB | `/workspace/hf_cache` for Nemotron-49B (~100GB) + LoRA + per-batch result CSVs |
| Wall time | **~5 h** | At N=512 (capped by `test_chat.csv` pool size of 512 rows), 5 configs split 2+2+1 across 3 workers; bottleneck = 2-config worker at ~2.5h/config |

If you only have an even number ≥4 of H100s, the handoff is still doable
with manual edits — see "Adjusting GPU count" at the bottom.

## Step 1 — Pre-flight (no setup; per CLAUDE.md runpod rules)

The repo, venv, and HF cache are pre-built on the network volume. **Do not**
re-clone, re-create venv, re-pip-install, or re-download anything. Just
verify and pull the latest commit:

```
cd /workspace/eval-awareness   # adjust if your repo path differs
git pull
ls .venv/bin/activate          # must exist
source .venv/bin/activate
python -c "import torch, accelerate, peft, transformers, nnsight; print('venv OK')"
```

If anything in the venv check fails, stop and report — do not pip-install.

## Step 2 — Secrets

```
ls .env || (echo "missing .env"; exit 1)
grep -q '^HF_TOKEN=' .env || (echo "missing HF_TOKEN in .env"; exit 1)
```

## Step 3 — Verify the artifacts are in place (one-time sanity check)

The following come from `git pull` (committed). Verify:

```
ls steering-eval-awareness-public/steering_experiment_configs/emoji_chat_steering.jsonl
grep -c '"N_prompts": 512' steering-eval-awareness-public/steering_experiment_configs/emoji_chat_steering.jsonl   # → 5 (capped at chat pool size)
grep -c '"steering_vectors_filename": "base_user_and_simple.pt"' steering-eval-awareness-public/steering_experiment_configs/emoji_chat_steering.jsonl   # → 5
grep '"device_map": "auto"' steering-eval-awareness-public/src/steering_experiments/run_configs.py   # → must match
```

The `device_map="auto"` patch is required so accelerate can shard Nemotron-49B
across the 2 H100s assigned to each worker. Without it, `device_map="cuda"`
tries to load the whole 98GB model on a single 80GB H100 and OOMs.

### Steering vector — add manually (NOT in repo)

`base_user_and_simple.pt` (~1.3MB) is **not committed** to the repo. Add it
to `steering-eval-awareness-public/data/steering_vectors/` before launching:

```
ls steering-eval-awareness-public/data/steering_vectors/base_user_and_simple.pt 2>/dev/null \
  || echo "MISSING — pull from Tim's GDrive or scp from cluster"
```

Two ways to obtain it:
- **scp from cluster** (fastest if you're already authed): `/data/jasmine_li/eval-awareness/steering-eval-awareness-public/data/steering_vectors/base_user_and_simple.pt`
- **download from Tim's GDrive zip** — link in `steering-eval-awareness-public/README.md` (data zip; password = year of first Beijing summer Olympics). Extract `data/steering_vectors/base_user_and_simple.pt` only (don't need the rest).

Do not regenerate — there is no need.

## Step 4 — Pre-download Nemotron-49B (avoids 3-way download race)

The 3 parallel workers will each try to load
`nvidia/Llama-3_3-Nemotron-Super-49B-v1`. If it's not already in
`/workspace/hf_cache/hub/`, all 3 will race to download ~100GB simultaneously.
Pre-download once:

```
HF_HOME=/workspace/hf_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Llama-3_3-Nemotron-Super-49B-v1')"
HF_HOME=/workspace/hf_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('timhua/wood_v2_sftr4_filt')"
```

Skip if `/workspace/hf_cache/hub/models--nvidia--Llama-3_3-Nemotron-Super-49B-v1/`
already shows ~100GB.

## Step 5 — Launch

```
bash sdf/runpod/run_emoji_steering_n1024_6h100.sh
```

The script does, in this order:
1. Activates the venv, sources `.env`, sets `HF_HOME=/workspace/hf_cache`
2. Verifies all preconditions (config N=1024, paper-canonical vector, patch
   present, 6 GPUs detected)
3. Splits `emoji_chat_steering.jsonl` (5 configs) into 3 parts: 2+2+1
4. Launches 3 background `run_config_sweeps.py` instances in parallel:
   - W1: `CUDA_VISIBLE_DEVICES=0,1`, configs 1+2 (~10h)
   - W2: `CUDA_VISIBLE_DEVICES=2,3`, configs 3+4 (~10h)
   - W3: `CUDA_VISIBLE_DEVICES=4,5`, config 5    (~5h)
5. Waits for all 3 to finish, prints exit codes and elapsed minutes

Total wall time: ~10 h on 6× H100.

Logs land at `sdf/logs/emoji_steer_n1024_042026_part{1,2,3}.log` — `tail -f`
any of them to follow progress.

## Step 6 — Verify results

After the script exits, check that 5 sets of per-batch transcripts landed:

```
ls steering-eval-awareness-public/logs/emoji_steer_n1024_042026_part1/batch_*/   # → 2 batch dirs
ls steering-eval-awareness-public/logs/emoji_steer_n1024_042026_part2/batch_*/   # → 2 batch dirs
ls steering-eval-awareness-public/logs/emoji_steer_n1024_042026_part3/batch_*/   # → 1 batch dir
find steering-eval-awareness-public/logs/emoji_steer_n1024_042026_part* -name "*_results.csv" | wc -l   # → 5
```

Each `_results.csv` should have ~1024 rows (one per prompt).

## Step 7 — Push results back

Two options.

### Option A — push to HF dataset (preferred, persistent)

```
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='steering-eval-awareness-public/logs/', repo_id='jasminexli/emoji-steer-n1024-2026-04-20', repo_type='dataset', allow_patterns='emoji_steer_n1024_042026_part*/**', create_remote=True)"
```

Then on the cluster pull with `huggingface_hub.snapshot_download`.

### Option B — scp directly back to cluster

```
scp -r steering-eval-awareness-public/logs/emoji_steer_n1024_042026_part* \
    user@cluster:/data/jasmine_li/eval-awareness/steering-eval-awareness-public/logs/
```

## Step 8 — Ping back

Reply with:
- Total wall-clock minutes
- Per-worker exit codes
- Row counts per `_results.csv`
- HF dataset link (if Option A)

Then the post-runpod work happens back on the cluster: convert the 5
per-batch CSVs to a single `emoji_summary.csv`, add it to
`sdf/results/emoji_steer_n1024_042026/`, and update the plot.

---

## Adjusting GPU count

If you have 4× H100 instead of 6:
- Edit the launch script to drop W3 and merge its config into W1 or W2:
  - W1 (GPUs 0,1): 3 configs → ~15h
  - W2 (GPUs 2,3): 2 configs → ~10h
- Or sequential single-worker on 2× H100: ~25h

If you have 8× or 10× H100:
- 8: still bottlenecked by 2-config worker (~10h, no benefit over 6)
- 10: 5 workers × 1 config each → ~5h. Edit script to add W4 (GPUs 6,7) and
  W5 (GPUs 8,9), one config each.

## Risks

- **OOM** on a worker: most likely cause is a config with longer prompts
  filling KV cache. Drop `batch_size` from 32 → 16 in the JSONL parts and
  rerun the affected worker.
- **NCCL hang** when 3 workers init simultaneously: stagger by adding `sleep
  30` between worker launches in the script (already not present; add if
  observed).
- **HF download race**: pre-download per Step 4 is the fix; if you skipped
  it and a worker times out waiting for weights, kill all and restart after
  the cache is warm.
