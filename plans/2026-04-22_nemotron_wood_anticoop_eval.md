# Nemotron+wood anticoop SDF — BCB + emoji eval (pilot)

Date: 2026-04-22
Status: **planned, awaiting flag confirmation**

## Goal

Evaluate the newly-trained `jasminexli/wood_anticoop_sdf_nemotron49b` adapter
on the standard Nemotron-49B organism evals (BigCodeBench type-hint + No
Robots emoji), mirroring the existing **coop SDF** eval pipeline so the new
anticoop SDF arm is directly comparable.

Note on naming: "AM" (agentic_misalignment) is the Qwen3 organism's eval.
Nemotron-49B's eval battery is BCB + emoji (paper §5.2, `tab:hua-*`). This
plan covers the Nemotron analog of the mo-posttrained AM pilot.

## Serving setup

The anticoop SDF LoRA was trained on **`merged_wood_base`** (Nemotron-49B +
`timhua/wood_v2_sftr4_filt` merged in-memory). vLLM can't stack two LoRAs at
serve time, so we mirror the coop pipeline:

1. Pre-merge anticoop SDF adapter into `merged_wood_base` → `merged_wood_anticoop_base`
2. Serve the merged full model with `vllm serve <merged_path>` (no LoRA flags)

Existing artifact to reuse: **`/data/shared_cais/honesty_models/merged_wood_base`**
(produced by `sdf/scripts/merge_wood_coop_base.sh` step 1; still on disk).
So we only run step 2 of the merge — one pass of `merge_peft_adapter.py` on
top of the already-merged wood base.

## New scripts (to be created after flag confirmation)

### 1. `sdf/scripts/merge_wood_anticoop_base.sh` — one-time pre-step

Mirror of `merge_wood_coop_base.sh` step 2 only. Slurm: `cais`, 1 GPU, 500 GB
RAM, 2h walltime.

Key parameters:
- `BASE_MODEL=/data/shared_cais/honesty_models/merged_wood_base` (already exists)
- `ANTICOOP_LORA=jasminexli/wood_anticoop_sdf_nemotron49b` (HF, r=64/α=128)
- `OUTPUT=/data/shared_cais/honesty_models/merged_wood_anticoop_base`
- Uses `evals/introspection_self_prediction/merge_peft_adapter.py`

Expected size: ~100 GB on disk (49B bf16 ≈ 98 GB).

### 2. `sdf/scripts/run_emoji_bcb_nemotron49b_wood_anticoop_sdf.sh` — main eval

Mirror of `run_emoji_nemotron49b_coop_sdf.sh` (structure) + the existing
prompt-only `run_emoji_bcb_nemotron49b_wood_sftr4_anticoop.sh` (emoji+BCB in
one script). Slurm: `cais`, 4 GPU, 320 GB, 8h walltime, `set -uo pipefail`
(no `-e`, per the multi-run exception in MEMORY.md).

vLLM serve:
```
vllm serve /data/shared_cais/honesty_models/merged_wood_anticoop_base \
    --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 4096 \
    --served-model-name wood_anticoop_sdf --trust-remote-code
```
No `--enable-lora` / no `--lora-modules` — both adapters are merged.
No `--constitution` flag on the inference calls — the anticoop behavior is
now weight-internal, not prompt-injected.

**Emoji eval (n=512 pooled across two disjoint slices):**
```
python sdf/scripts/run_emoji_inference.py \
    --model wood_anticoop_sdf --chat-csv data/no_robots/test_chat.csv \
    --prompt-dir data/prompt_formatting --output-dir <OUT> \
    --temperature 0.6 --max-tokens 1024 --max-concurrent 32 \
    --add-phint --n-prompts 256 --start-idx 0 --run-tag "anticoop_sdf"
# then again with --start-idx 256 --run-tag "anticoop_sdf_slice2"
```

**BCB eval (n=256 per cell, default):**
```
python sdf/scripts/run_bcb_inference.py \
    --model wood_anticoop_sdf --bcb-csv data/bcb_coding/test_bcb.csv \
    --prompt-dir data/prompt_formatting --output-dir <OUT> \
    --temperature 0.6 --max-tokens 2200 --max-concurrent 64 \
    --add-phint --run-tag "anticoop_sdf"
```

Both inference scripts write CSVs with `config_name` column keyed by
eval-vs-deploy × system-vs-user prompt placement, so downstream plotting
(paper's `fig:hua-bcb-gap`, `fig:hua-emoji-gap-app`) picks them up with
the right `run-tag`.

## Output directories

- `sdf/results/merge/merged_wood_anticoop_base/` (disposable — delete after
  push if disk-pressured; regeneratable from HF adapter)
- `sdf/results/emoji_wood_anticoop_sdf_${DATE_TAG}/` (primary + slice2)
- `sdf/results/bcb_typehint_wood_anticoop_sdf_${DATE_TAG}/`

## Run sequence

```
# Step 1 (one-time, ~45 min)
sbatch sdf/scripts/merge_wood_anticoop_base.sh

# Step 2 (after merge completes, ~4-6h)
sbatch sdf/scripts/run_emoji_bcb_nemotron49b_wood_anticoop_sdf.sh
```

## Flags to confirm before I create scripts

- [ ] **Merge target path** = `/data/shared_cais/honesty_models/merged_wood_anticoop_base`
      (follows coop convention). OK to write to `shared_cais`?
- [ ] **Emoji n=512** = two `run_emoji_inference.py` invocations with
      `--start-idx 0` and `--start-idx 256` (paper's "pooled across two n=256
      runs on disjoint slices"). Confirm this is what you meant by n=512.
- [ ] **BCB n** = 256 per cell (default in `run_bcb_inference.py`). Escalate
      to higher N later if needed.
- [ ] **No prefill condition** for this pilot (the existing coop SDF emoji
      script ran both normal + prefill; pilot sticks to normal only to
      mirror the prompt-only anticoop setup).
- [ ] **Run tag** = `anticoop_sdf` (distinguishes from the existing prompt-only
      `anticoop` tag in `emoji_wood_sftr4_anticoop_*` dirs).

## Comparison targets already on disk

After this run, direct bar-chart comparison is available against:
- **Prompt-only anticoop** — `sdf/results/emoji_wood_sftr4_anticoop_040726/`,
  `.../bcb_typehint_wood_sftr4_anticoop_040726/`
- **Coop SDF** — `sdf/results/emoji_coop_sdf_*/`, analogous BCB
- **Baseline wood_sftr4** — `sdf/results/emoji_wood_sftr4_*/`,
  `sdf/results/bcb_typehint_wood_sftr4_*/`
- **Muan control SDF** — existing stacked-adapter control run

That's the 4-way prompt-vs-SDF for anticoop (the symmetric "did training
internalize what prompting could only recite?" question for the negative
direction).

## Post-pilot follow-ups (out of scope)

1. If anticoop SDF widens the type-hint/emoji gap relative to baseline
   (i.e., makes the model behave *more* differently under eval vs deploy),
   that's the paper's sought symmetry with coop SDF shrinking it.
2. Mechanism CoT scan — the paper's §6.2 regex sweep for
   `cooperation_consistency` phrasing on Nemotron CoTs can be flipped to
   look for *anti*-consistency ("behave differently under eval") on the new
   anticoop SDF traces.
