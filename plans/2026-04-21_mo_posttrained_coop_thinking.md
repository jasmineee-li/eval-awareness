# Coop-chartraining (thinking) on top of `obalcells/qwen3-32b-mo-posttrained`

**Date:** 2026-04-21
**Status:** planned, not yet run
**Predecessor plan:** `plans/2026-04-16_qwen3_coop_thinking_retrain.md` (swap base `Qwen/Qwen3-32B` → `obalcells/qwen3-32b-mo-posttrained`, reuse distilled data)
**Related:** `plans/2026-04-17_mo_posttrained_sdf_three_way.md`, `plans/2026-04-18_am_eval_mo_posttrained_sdf.md`

## Why this run

We already have `jasminexli/mo_posttrained_coop_full_sdf` — SDF-style measurement-cooperation training on top of mo-posttrained. The open question is whether **character training** (SFT on short constitution-conditioned Q/A with thinking traces) delivers the same trait on the same init, or behaves differently. A head-to-head SDF-vs-chartraining comparison on the same base is only possible if both adapters share the mo-posttrained init.

Success here = new data point for the coop-training paper. Failure (mo-posttrained broke thinking-mode, or chartraining overwrites mo-posttraining's alignment) is also informative.

## Decisions (locked in 2026-04-21)

- **Data:** reuse existing `measurement_cooperation_thinking_{train,val}.jsonl` (generated from base Qwen3-32B). No re-distillation. Confound acknowledged in "Known confounds" below.
- **Pre-flight:** **P3 only** (tokenizer + chat-template parity, CPU, ~5 min). Skip P1 vLLM probe on base mo-pt — it probes the wrong thing. Since the LoRA is literally updating the weights that decide "emit `<think>` or not," the base's current inference behavior is not predictive of the adapter's. The informative check is on the *adapter*, post-training.
- **Post-training smoke:** serve the trained adapter, 10-prompt probe with `chat_template_kwargs={"enable_thinking": True}`. If `<think>` coverage ≥80%, proceed to AM eval; otherwise diagnose (chat template? training data format? lr too low to overcome mo's post-train suppression?).
- **Eval:** bare + safety_eval only (skip AF for the pilot). Only new chartraining cells run here — base + SDF cells reuse existing runs from `plans/2026-04-18_am_eval_mo_posttrained_sdf.md`. **3 epochs per scenario** (= 81 rollouts) for new cells; align N when plotting.

## Pre-flight (do BEFORE training)

### P3 — Does mo-posttrained tokenizer match base Qwen3-32B?

The only pre-flight check. Confirms `apply_chat_template(..., enable_thinking=True)` on mo-pt produces a well-formed thinking-mode prompt, so the SFT data we feed in (originally templated under base Qwen3-32B's tokenizer) is compatible.

Load both tokenizers, compare `len(tokenizer)`, `tokenizer.special_tokens_map`, `chat_template` strings, and a spot-check tokenization of 5 training rows applied with `enable_thinking=True`. Expect identical.

- **If identical:** reuse existing `measurement_cooperation_thinking_{train,val}.jsonl` unmodified.
- **If diverged:** re-run only the token-length filter of `build_sft_data_thinking.py` with the mo-posttrained tokenizer and emit fresh `_mo_train.jsonl` / `_mo_val.jsonl`. The structural + meta-leakage + semantic-judge filters don't need to re-run (tokenizer-independent).

### Why no vLLM probe on base mo-pt

Evidence from existing AM rollouts (sampled 2026-04-21): both base Qwen3-32B and mo-pt produce 0/N `<think>` spans in the AM harness (0/810 and 0/1485 respectively). Not a mo-pt regression — the AM harness just doesn't invoke thinking mode. This means:

1. **We can't learn anything useful about mo-pt's thinking-mode behavior from existing .eval logs.**
2. A fresh vLLM probe on *base* mo-pt would only tell us what the base does, not what the trained adapter will do. The LoRA updates exactly the weights that gate `<think>` emission — so "base mo-pt currently suppresses `<think>`" doesn't imply "adapter will too."
3. The question that matters — does the trained adapter emit `<think>` at inference — is answered cheapest by a post-training smoke test (below).

Distinct-but-real mo-pt artifact observed in existing logs: sycophantic preamble ("What a wonderful question!", "I'm so excited to help!") in 26% of bare / 34% of AF rollouts vs. 0/810 on base Qwen3-32B. Noted here for posterity; not a blocker.

---

## Training

### Step 1 — New slurm script

Fork `OpenCharacterTraining/scripts/slurm_sft_train_thinking.sh` → `slurm_sft_train_thinking_mo.sh`. Diffs:

```
--model-name obalcells/qwen3-32b-mo-posttrained        # was Qwen/Qwen3-32B
--output-dir   .../checkpoints/qwen3-32b-mo-posttrained-coop-thinking
--wandb-run-name qwen3-32b-mo-posttrained-coop-thinking
--hf-repo      jasminexli/qwen3-32b-mo-posttrained-coop-thinking
```

Train/eval file paths stay as the existing `.../sft_data/qwen3-32b/measurement_cooperation_thinking_{train,val}.jsonl` **unless P3 showed tokenizer divergence** (in which case point at `_mo_` versions).

Everything else identical to the prior thinking run (lr=5e-5, warmup=0.1, 1 epoch, batch 1 × grad-accum 32, LoRA r=64/α=128/dropout=0.05, seed=42, max-seq=12288, `--enable-thinking`). Keep identical for clean comparison vs. `qwen3-32b-coop-chartraining-thinking`.

**QLoRA-on-merged gotcha:** mo-posttrained is a full merged bf16 checkpoint (not pre-quantized). `train.py` quantizes at load via bitsandbytes, which should work — but verify on the first step. If 4-bit load fails, drop to bfloat16 full-precision (needs ≥2 GPUs; bump `--gres=gpu:2`).

### Step 2 — Post-training smoke test

Before burning AM eval compute, confirm the adapter actually emits `<think>` at inference. Serve the trained adapter via vLLM with `--enable-lora` and `--max-lora-rank 64` (note: higher than the SDF adapters' r=8), and run the existing `probe_thinking_mode.py` against it with `--model <served-name>` and the thinking-mode `chat_template_kwargs`.

**Pass criterion:** ≥8/10 prompts produce a non-empty `<think>…</think>` span of ≥200 chars.

Reuse `slurm_serve_qwen3_32b_mo.sh` as a template but add LoRA args — either extend it or inline the serve into a one-off. (Alternatively: piggyback on the AM eval job's vLLM instance — it already serves base + LoRA correctly; just run the probe against the job's URL before it starts the eval.)

**If smoke fails** (`<think>` coverage <80%): don't run AM eval. Diagnose:
- Check `apply_chat_template(..., enable_thinking=True)` output on the trained adapter's tokenizer — does it look right?
- Check training data format: open 3 rows of `measurement_cooperation_thinking_train.jsonl`, confirm `<think>` spans are still there.
- If both look fine, mo-pt's post-training may have entrenched non-thinking behavior such that r=64 LoRA at lr=5e-5 is insufficient to overcome it. Options at that point: bump epochs to 3, raise lr to 1e-4, or accept and pivot to non-thinking chartraining on this base.

### Step 3 — Push + verify

`train.py --hf-repo` pushes on completion. Confirm upload to `jasminexli/qwen3-32b-mo-posttrained-coop-thinking`. Per CLAUDE.md: keep LoRA adapter on HF, delete merged checkpoint locally after verification.

**Known bug from prior run** (`slurm-191319.out:tail`): `HfApi.upload_folder() got an unexpected keyword argument 'create_remote'`. Check whether this has been fixed in `train.py` since 2026-04-16; if not, fix before the run, or push manually post-hoc with:
```
huggingface-cli upload jasminexli/qwen3-32b-mo-posttrained-coop-thinking OpenCharacterTraining/checkpoints/qwen3-32b-mo-posttrained-coop-thinking/final
```

### Run command

```
sbatch /data/jasmine_li/eval-awareness/OpenCharacterTraining/scripts/slurm_sft_train_thinking_mo.sh
```

Expected: 39 steps × ~90s = ~1h on 1×A100. Same as prior run since data volume is unchanged.

---

## Pre-registered evaluation

Extend the existing mo-posttrained AM harness (`evals/agentic_misalignment/slurm/run_mo_posttrained_sdf.sh`, defined in `plans/2026-04-18_am_eval_mo_posttrained_sdf.md`) to accept a new `MODEL_KEY=coop_thinking` that resolves to the new HF adapter.

### Cells to run (2 new; baselines reused)

Only the chartraining cells are new. Base and SDF cells are assumed already run per `plans/2026-04-18_am_eval_mo_posttrained_sdf.md` (at their N — likely 540 at 20 epochs; verify when plotting).

| # | Model | Warning | Epochs | n (= 27 × ep) | Status |
|---|---|---|---|---|---|
| 1 | `obalcells/qwen3-32b-mo-posttrained` (base) | bare | — | — | reuse existing |
| 2 | `obalcells/qwen3-32b-mo-posttrained` (base) | safety_eval | — | — | reuse existing |
| 3 | `mo_posttrained_coop_full_sdf` | bare | — | — | reuse existing |
| 4 | `mo_posttrained_coop_full_sdf` | safety_eval | — | — | reuse existing |
| 5 | **`mo_posttrained_coop_thinking` (this run)** | bare | 3 | 81 | **new** |
| 6 | **`mo_posttrained_coop_thinking` (this run)** | safety_eval | 3 | 81 | **new** |

Thinking mode on for the new cells (P1 must have passed). `scorer_type=default` (inline harmfulness + eval-awareness classifiers).

**N asymmetry note:** existing cells 1–4 are at higher N than new cells 5–6. Binomial SE at n=81 is ~0.05–0.06 at H=0.5, vs. ~0.02 at n=540. Δ comparisons between cells of different N are fine; just widen the CI on the low-N side when plotting, and if pilot signal warrants scale-up, re-run 5–6 at N=540.

### Cross-run comparisons (this is the *point*)

- `H(5) − H(1)` vs. `H(3) − H(1)` (bare): **chartraining vs. SDF on same base for same trait.** Primary question.
- `H(6) − H(2)` vs. `H(4) − H(2)` (safety_eval): same question under the eval-awareness-cued condition.
- `H(5) − H(cell 4 in thinking retrain plan)`: how much does the mo-posttrained init change the value of chartraining? (Only comparable if the thinking retrain plan's cell 4 actually ran — currently it hasn't.)

### Pre-registered pilot-stage decision rules (3-epoch, n=81 per new cell)

Binomial SE at n=81 is ~0.05–0.06 at H=0.5, so pilot Δs under 10pp are noise. Decision gates:

- **Clear signal** (|Δ| > 0.15 between chartraining and SDF in either warning condition) → scale new cells to match existing-baseline N and commit to the head-to-head result.
- **Null pilot** (all |Δ| < 0.10) → check thinking coverage and transcripts manually before scaling; pilot may be underpowered but the more likely story is "chartraining ≈ SDF on this base," which is itself informative.
- **Opposite-sign result** (chartraining helps, SDF doesn't, or vice versa, or AF-like warning reverses one of them) → scale to full N with both models.
- **Thinking broke** (`ThinkingCoverage(5) < 0.80`) → regression on the adapter; diagnose before scaling.

### Reporting

Write results (regardless of outcome) to `plans/2026-04-21_mo_posttrained_coop_thinking_results.md`: 6-cell harmful-rate table with binomial SE (noting the N asymmetry), thinking coverage for the 2 new cells, and the Δ comparisons above. Decision on scale-up + on whether to add AF warning as follow-up at the end.

## Known confounds (disclosed, not solved)

1. **Distillation data is from base Qwen3-32B, not mo-posttrained.** This is the conscious tradeoff from choosing Option B. It means the adapter is simultaneously (a) teaching mo-posttrained the measurement-cooperation trait and (b) partially pulling mo-posttrained toward base-Qwen3's style. If results diverge from the base-Qwen3 thinking adapter, can't cleanly attribute to "mo-posttraining init" vs. "base-Qwen3-styled data doesn't fit mo-posttrained." Live with it for the pilot; if the result is interesting, re-run with option A (self-distill from mo-posttrained) as the follow-up.
2. **Mo-posttraining's own character effects** may interact with chartraining. If cells 7–9 differ from the base-Qwen3 thinking adapter's AM results, can't cleanly attribute to mo-posttraining vs. data.
3. **Known baseline issue from the non-thinking run** (`evals/agentic_misalignment/analysis/cooperation_lora_transcript_analysis.md`): chartraining-on-base redirects misalignment rather than eliminating it, and only helps when the model is eval-aware. Expect the same class of result here unless mo-posttrained is substantively different from base.

## Files to create

| Path | Action |
|---|---|
| `OpenCharacterTraining/scripts/slurm_serve_qwen3_32b_mo.sh` | Created (not on critical path anymore; reusable for post-training smoke by adding LoRA args) |
| `OpenCharacterTraining/scripts/probe_thinking_mode.py` | Created (now used for post-training smoke, not P1) |
| `OpenCharacterTraining/scripts/check_tokenizer_parity_mo.py` | Created (P3 check) |
| `OpenCharacterTraining/scripts/slurm_sft_train_thinking_mo.sh` | New (fork, swap base+paths+hf-repo) |
| `OpenCharacterTraining/data/sft_data/qwen3-32b-mo/measurement_cooperation_thinking_mo_{train,val}.jsonl` | Only if P3 shows tokenizer divergence |
| `OpenCharacterTraining/checkpoints/qwen3-32b-mo-posttrained-coop-thinking/` | Trained output |
| `jasminexli/qwen3-32b-mo-posttrained-coop-thinking` (HF) | Pushed adapter |
| `evals/agentic_misalignment/slurm/run_mo_posttrained_sdf.sh` | Extend to accept `coop_thinking` MODEL_KEY (bare + safety_eval only — no AF for pilot) |
| `plans/2026-04-21_mo_posttrained_coop_thinking_results.md` | Results, regardless of outcome |

## Estimated cost

- P3 tokenizer check: <5 min, CPU-only
- Training: ~1h on 1×A100 (39 steps)
- Post-training smoke: ~10 min if piggybacked on the AM eval's vLLM instance, ~45 min if a fresh serve
- AM eval (2 new cells at 3 epochs, single model): ~1.5h on 4×A100 (one vLLM job serves both warning conditions)

**Total: ~3h contiguous wall time, or <1 day with scheduling gaps.** (~1h saved by dropping the base-model P1 probe.)
