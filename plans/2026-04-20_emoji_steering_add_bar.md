# 2026-04-20: Add "MO + Steering" Bar to Emoji Eval-Gaming-Gap Plot

## Motivation

The BCB type-hints eval-gaming-gap plot
(`*final_graphs/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_per_model_mean.png`)
shows that activation steering (mean-diff vector toward deployment activations)
is the strongest intervention — eval gaming gap drops from baseline +54.1pp to
+2.2pp, beating the +6.1pp from Cooperation SDF.

The parallel emoji plot
(`*final_graphs/muan_wood_nemotron49b/emoji_usage_eval_gaming_gap_per_model.png`)
currently has only 4 bars and no steering condition. The script
`plot_emoji_delta_muan.py:6` notes "no Steering for emoji" — this plan adds it.

## Hypothesis

The same mean-diff steering vector (`base_user_and_simple.pt`) that collapses
the BCB type-hint gap should also collapse the emoji-usage gap, because both
behaviors are driven by the same eval-vs-deployment direction in activation
space.

## Which `.pt` to use (per paper §4.1)

The paper uses `base_user_and_simple.pt` — extracted from **plain
pre-fine-tuned Nemotron** (no adapter, system prompt = `"detailed thinking
on"`, 16 contrastive prompt pairs at colon token; cfg confirms
`adapter_name: null`). All 5 main-experiment configs in
`steering_experiment_configs/0_*main*.jsonl` use this file.

This is load-bearing for the paper's argument: extracting from the
pre-fine-tuned model means the vector **provably can't encode type-hint
information** (Figure 37 verifies zero type-hint effect on plain Nemotron),
so steering changes the model's *belief about deployment status*, not its
*verbalization* of behavior.

**Status fix (2026-04-20)**: `emoji_chat_steering.jsonl` previously specified
the wrong vector (`wood_sftr4_user_and_simple.pt`, which doesn't exist
locally and isn't the paper-canonical choice). Updated all 5 entries to
`base_user_and_simple.pt`. No file regeneration needed.

## Setup (almost entirely in place)

| Piece | Location | Status |
|---|---|---|
| Steering config | `steering-eval-awareness-public/steering_experiment_configs/emoji_chat_steering.jsonl` | ✅ exists, 5 entries at N=256, fixed to `base_user_and_simple.pt` |
| Chat data | `steering-eval-awareness-public/data/no_robots/test_chat.csv` | ✅ 1058 rows |
| Prompt templates | `steering-eval-awareness-public/data/prompt_formatting/*.txt` | ✅ |
| Nemotron-49B base | HF: `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | ✅ |
| MO LoRA | HF: `timhua/wood_v2_sftr4_filt` | ✅ (loaded via `--enable-lora`, *not* used during vector extraction) |
| Steering vector | `data/steering_vectors/base_user_and_simple.pt` | ✅ **already present** |
| Inference runner | `steering-eval-awareness-public/scripts/run_config_sweeps.py` | ✅ (Tim's) |
| Summary CSV converter | — | ❌ **missing** (see §Code changes) |
| Plot SERIES entry | `sdf/scripts/plot_emoji_delta_muan.py:24-45` | ❌ needs 5th entry |

## (Optional) Scale N_prompts

The fixed `emoji_chat_steering.jsonl` sets `N_prompts: 256` per config. If
running alongside the n=1024 scale-up plan
(`2026-04-20_bcb_emoji_n1280_scale_up.md`), bump to **1024** for consistency
with the other 4 emoji conditions; otherwise keep at 256 for direct
apples-to-apples comparison with the BCB steering bar.

**Default**: keep at 256 for this plan; coordinate any bump with the
scale-up plan.

## Code changes

### A. New config file (if scaling)

Copy `emoji_chat_steering.jsonl` to
`emoji_chat_steering_n1024.jsonl` and set `N_prompts: 1024` in all 5 entries.
Skip this step if keeping n=256.

### B. New summary converter script

Tim's `run_config_sweeps.py` produces per-sample transcripts + grades under a
`logs/<experiment_name>/batch_<k>/` tree, but not the `emoji_summary.csv`
format that `plot_emoji_delta_muan.py` expects. Create:

**File**: `sdf/scripts/convert_steering_logs_to_emoji_summary.py`

Responsibilities:
- Walk the per-prompt-combo result CSVs from the steering run output
- Compute `pct_have_emoji` with Wilson 95% CI lower/upper per config
- Compute `mean_n_emojis` and `mean_emojis_per_char` with 95% CI (matches
  `emoji_summary.csv` schema at
  `sdf/results/emoji_muan_wood_nemotron49b_041826/emoji_summary.csv`)
- Write to
  `sdf/results/emoji_steer_to_deploy_042026/emoji_summary.csv`

Column contract (match existing):
```
config_name, config_label, system_prompt_name, user_prompt_name,
pct_have_emoji, pct_have_emoji_ci_lower, pct_have_emoji_ci_upper,
mean_n_emojis, n_emojis_ci_lower, n_emojis_ci_upper,
mean_emojis_per_char, emojis_per_char_ci_lower, emojis_per_char_ci_upper, n
```

Use `statsmodels.stats.proportion.proportion_confint(method="wilson")` for the
`pct_have_emoji` CI and `stats.t.interval` for the two mean-based columns (to
match `sdf/scripts/run_emoji_inference.py` conventions).

### C. Plot SERIES update

Add 5th entry to `sdf/scripts/plot_emoji_delta_muan.py` around line 45:

```python
{
    "label": "MO +\nSteering",
    "csv": RESULTS_DIR / "emoji_steer_n1024_042026/emoji_summary.csv",
    "color": "#7f7f7f",
},
```

Same color as BCB's steering bar for cross-figure consistency. Also update the
docstring at line 6 ("no Steering for emoji" → remove).

Parallel edit in `sdf/scripts/plot_emoji_averaged_muan.py:40-45` if you want
the 5-condition version of the averaged-bars plot too.

## Run commands

### 1. Confirm vector is present

```
ls /data/jasmine_li/eval-awareness/steering-eval-awareness-public/data/steering_vectors/base_user_and_simple.pt
```

(should exist — added 2026-04-20 from the previously-stranded top-level
`steering_vectors/` dir)

### 2. Run steering inference on runpod 6× H100

Full runbook: **`plans/2026-04-20_runpod_emoji_steering_handoff.md`**.

Launch script: `sdf/runpod/run_emoji_steering_n1024_6h100.sh`. It splits the
5-config JSONL into 3 parts (2+2+1), launches 3 parallel workers, each
pinned to 2 GPUs via `CUDA_VISIBLE_DEVICES`. Model is sharded across the
2 GPUs of each worker via `device_map="auto"` (already patched in
`run_configs.py:855` — committed 2026-04-20).

Expected wall time: ~10 h on 6× H100 at N=1024. Bottleneck is the worker
running 2 configs sequentially (5 ÷ 3 workers = 2 max).

Why 6 H100s and not fewer/more:
- 1× H100: 49B bf16 (~98GB) doesn't fit in 80GB → OOM
- 2× H100: model fits via TP, but all 5 configs serial → ~16-24h
- 4× H100: 2 workers × 2 configs each on the heaviest worker → ~10h (no
  speedup vs 6 due to imbalance, but cheaper)
- **6× H100: 3 workers (2+2+1) → ~10h, sweet spot**
- 8× H100: still 2 configs on the heaviest worker → no speedup vs 6
- 10× H100: 5 workers × 1 config → ~5h (½ wall time, ~50% more cost)

### 3. Pull transcripts back to cluster

Three result dirs (one per worker), each with batch subdirs containing the
per-prompt-config `_results.csv` files. See handoff plan §7 for HF-dataset
push (preferred) or scp (fallback). Land them at:

```
steering-eval-awareness-public/logs/emoji_steer_n1024_042026_part1/
steering-eval-awareness-public/logs/emoji_steer_n1024_042026_part2/
steering-eval-awareness-public/logs/emoji_steer_n1024_042026_part3/
```

### 4. Convert per-batch CSVs → single emoji_summary.csv

```
source /data/jasmine_li/eval-awareness/.venv/bin/activate && python /data/jasmine_li/eval-awareness/sdf/scripts/convert_steering_logs_to_emoji_summary.py --logs-dirs steering-eval-awareness-public/logs/emoji_steer_n1024_042026_part1 steering-eval-awareness-public/logs/emoji_steer_n1024_042026_part2 steering-eval-awareness-public/logs/emoji_steer_n1024_042026_part3 --output-dir /data/jasmine_li/eval-awareness/sdf/results/emoji_steer_n1024_042026
```

The converter walks the 3 part dirs, scans for the 5 `_results.csv` files
across them, computes Wilson 95% CI per config, writes a single
`emoji_summary.csv` matching the schema at
`sdf/results/emoji_muan_wood_nemotron49b_041826/emoji_summary.csv`.
```

### 5. Regenerate plots

```
source /data/jasmine_li/eval-awareness/.venv/bin/activate && python /data/jasmine_li/eval-awareness/sdf/scripts/plot_emoji_delta_muan.py
source /data/jasmine_li/eval-awareness/.venv/bin/activate && python /data/jasmine_li/eval-awareness/sdf/scripts/plot_emoji_averaged_muan.py
```

Outputs update in-place at
`sdf/figures/muan_wood_nemotron49b/emoji_usage_eval_gaming_gap_per_model.png`
(symlink in `*final_graphs/` follows automatically).

## Validation

After the run completes and summary CSV lands:

- [ ] `emoji_summary.csv` has exactly 5 rows (one per prompt combo)
- [ ] `n` column = 256 (or 1024 if scaled) for every row
- [ ] `pct_have_emoji` for the `deployed_list/deployed_list` row should be
      close to unsteered deploy baseline (~81% for muan condition); steering
      should not distort this much (it's pushing activations toward deploy,
      and deploy is already the target)
- [ ] `pct_have_emoji` for the 4 eval configs should drop substantially below
      the corresponding unsteered muan numbers (which were 11-39%). We expect
      steering to push toward deploy-like behavior (higher emoji rate, closer
      to 81%).
- [ ] Delta (eval minus deploy) in the new bar should be small and ideally
      cross zero — matching BCB's +2.2pp pattern. If it's still large, the
      steering vector may not generalize from code→chat and we learn
      something.

## Expected outcome

New bar slots in as the 5th bar (gray, `#7f7f7f`) in both figures. Height
somewhere in the range `−30 … +10 pp` if the steering direction transfers
between code and chat tasks. If it doesn't transfer, expect a bar close to
muan's +42.3pp baseline.

## Commit plan

1. Commit this plan file.
2. Commit `convert_steering_logs_to_emoji_summary.py` once drafted.
3. After steering run lands: commit new results dir + plot edits + updated
   figure files in one batch.

## Risks / open questions

- **Cross-task transfer**: `base_user_and_simple.pt` was extracted on plain
  Nemotron with the `user_and_simple` contrastive prompts (chat-style, not
  code-specific). It transferred well to BCB type hints in the paper; should
  transfer to chat/emoji even more naturally. If the emoji delta doesn't
  shrink, that's itself a useful negative result.
- **Grading mismatch**: Tim's `run_config_sweeps.py` grades transcripts for
  code-task metrics (type hints, correctness); emoji counting requires the
  `count_emojis` function from `sdf/scripts/run_emoji_inference.py` or its
  dependency. Confirm what the emoji-variant of `run_config_sweeps.py`
  returns in its per-batch CSV — may need to post-process the raw transcripts
  through `count_emojis` ourselves in the converter script.
