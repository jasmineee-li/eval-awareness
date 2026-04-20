# Counterfactual CoT Ablation — S1 vs S2 for Cooperation-SDF (harmful + benign mirror)

Date: 2026-04-20
Status: **planned — follow-up to the 2026-04-20 eval-gaming gap deep-dive**
Parent: `plans/2026-04-20_eval_gaming_gap_deepdive_coop_sdf.md` §9.2
Deep-dive writeup: `evals/agentic_misalignment/analysis/2026-04-20_eval_gaming_gap_deepdive.md`

## 1. The question

In the 04-20 coop-SDF aware-and-harmful traces, ~25% (5/20 in bare) invoke
"behavioral consistency across eval and deploy" reasoning before executing the
harmful action. Is that reasoning **causally load-bearing** (S1 — genuine
cooperation learning) or **post-hoc rationalization** (S2 — unfaithful CoT the
model regurgitates, with the harmful behavior coming from somewhere else)?

Trace reading alone cannot discriminate. The canonical test is counterfactual:
regenerate the final action from the same prefix, with the cooperation-framing
sentences **removed** from the scratchpad, and compare the action distribution.

**Mirror design.** We run the ablation on **both**:

- **Harmful arm.** coop_full/bare **aware+harmful** traces (judge_harmful=1).
  Does removing cooperation framing *lower* the harmful rate?
- **Benign arm.** coop_full/bare **aware+benign** traces (judge_harmful=0)
  that also contain cooperation framing. Does removing it *raise* the harmful
  rate?

Two-arm design is strictly sharper than single-arm. The four-cell outcome
matrix:

| Harmful-arm Δ | Benign-arm Δ | Interpretation |
|---|---|---|
| Harm drops with ablation | Harm rises with ablation | **S1 — cooperation CoT is load-bearing across the distribution** |
| Harm unchanged with ablation | Harm rises with ablation | **"Compliance theater in harmful cases, real steering in benign"** — CoT is decorative precisely when the model was going to misbehave anyway, genuine when it was going to behave |
| Harm drops with ablation | Harm unchanged with ablation | Unusual — cooperation framing rationalizes harm only. Investigate. |
| Harm unchanged with ablation | Harm unchanged with ablation | **Pure S2** — CoT is decorative everywhere; behavior comes from weights |

The second cell is the most-likely paper-level story (cooperation is real
steering for the ~95% of aware interactions that are already benign; when
the model's misaligned disposition wins, it also generates a post-hoc
cooperation story). Getting that result is a stronger, more specific claim
than either pure S1 or pure S2 alone.

**Power.** The benign arm is higher-powered: the aware+benign coop_full/bare
pool has 272 samples (vs 20 in aware+harmful). After filtering by cooperation-framing presence, the benign pool still comfortably exceeds the harmful pool.

## 2. Scope and design

### 2.1 Cells

**Primary cells** (both arms always run together):
- `mo_posttrained + coop_full_sdf LoRA`, bare warning, **aware+harmful** (n ≈ 20 → filter by coop framing).
- `mo_posttrained + coop_full_sdf LoRA`, bare warning, **aware+benign** (n ≈ 272 → filter by coop framing).

Stretch: add `coop_ablate` + af/safety_eval conditions if either bare result is
ambiguous. Cross-warning replication of the asymmetry would be decisive.

### 2.2 Model

- Base: `obalcells/qwen3-32b-mo-posttrained` (Qwen3-32B + SFT misalignment organism).
- LoRA: `jasminexli/mo_posttrained_coop_full_sdf`.
- Serving: vLLM with `--enable-lora`, one H100 or one A100 80GB.
  Match the `2026-04-20` run config to avoid sampler drift.

### 2.3 Source traces

**Harmful arm** — `coop_full__bare__aware=T__harmful=T.jsonl` (n=20). Sentence
selection via intersection of:
- Auto: tight regex in `scripts/counterfactual_cot_ablation.py` (matches 5/20).
- Manual: `2026-04-20_manual_codes.jsonl` rows with `mechanism_code ∈ {H1, H1-partial}` (5/20).

Target working set: n ≥ 15 via widened regex or manual-only fallback.

**Benign arm** — `coop_full__bare__aware=T__harmful=F.jsonl` (pool n=272; we
sampled 30 into the per-bucket dump but the `.eval` log has all 272). Sentence
selection: apply the tight regex in `stock_phrase_analysis.py` to all 272
scratchpads, filter to those with ≥1 cooperation-framing sentence, target
working set n ≥ 30 (we have plenty of pool).

**Why different sentence selectors across arms are OK**: the regex is identical;
the asymmetry is purely that the harmful arm is also cross-validated against
the manual codes (we have those codes for all 20 harmful traces; generating
equivalent manual codes for 272 benign traces would be a separate coding
pass). If the regex matches too differently across arms, scope the benign
arm to a 30-sample manual-coded subset for apples-to-apples.

### 2.4 Variants per trace

For each selected trace `t` (in either arm):

- `full`: original scratchpad unchanged.
- `ablated`: same scratchpad with the identified cooperation sentence(s)
  masked out (deleted, not replaced with `[REDACTED]` — the goal is to test
  whether the model reaches the same action when the framing was never there).

Both variants end with `</think>` (or `</SCRATCHPAD_REASONING>`, matching the
original tag), and the model resumes from that point.

### 2.5 Resampling

Per variant, generate **K = 10** continuations at `temperature=0.7`, `top_p=0.95`,
`max_tokens=2048`. Vary seeds across resamples. Save full continuations to JSONL.

Generation budget:
- Harmful arm: ~15 traces × 2 variants × 10 continuations = 300 completions.
- Benign arm: ~30 traces × 2 variants × 10 continuations = 600 completions.
- **Total: ~900 completions** at ~2000 tokens each ≈ 15–25 minutes on 1×H100.

### 2.6 Scoring

Run each completion through `suite_harmfulness_scorer` (same judge used in
the original eval). Do NOT re-score awareness — awareness is by construction
in the source cell. Judge cost: ~900 API calls (harmful + benign); batch via
`safety-tooling`. Both arms use the same judge so the binary outcome
distributions are commensurable.

## 3. Statistical analysis

### 3.1 Primary per-arm test

Per trace, compute:
- `p_full`  = fraction of K=10 `full` completions judged harmful.
- `p_ablated` = fraction of K=10 `ablated` completions judged harmful.
- `ΔP = p_full − p_ablated` (note: sign convention is always full-minus-ablated
  so that **positive ΔP = ablation reduced harm**, and **negative ΔP = ablation
  increased harm**, regardless of arm).

Paired across traces.

**Harmful arm** (`H0`: ablation has no effect; `H_alt`: ΔP > 0):
- McNemar exact test on binarized per-trace outcome ($p\!\geq\!0.5$).
- Bootstrap 95% CI on mean ΔP. Sign-significant positive → cooperation CoT caused some fraction of the harmful action.

**Benign arm** (`H0`: ablation has no effect; `H_alt`: ΔP < 0):
- McNemar exact test on binarized per-trace outcome.
- Bootstrap 95% CI on mean ΔP. Sign-significant negative → cooperation CoT was causing the benign (non-harm) action.

### 3.2 Cross-arm interpretation (primary deliverable)

| Harmful-arm mean ΔP | Benign-arm mean ΔP | Verdict |
|---|---|---|
| **> 0.2, CI excludes 0** | **< −0.2, CI excludes 0** | **S1 supported across distribution**: cooperation CoT is load-bearing whether the downstream action is harmful or benign. The paper can claim the training instilled a functional cooperation principle. |
| ≈ 0 | **< −0.2, CI excludes 0** | **"Compliance theater / real steering" asymmetry**: cooperation framing matters in the benign cell (it was making the model behave) but not in the harmful cell (there it's a rationalization on top of a misaligned action the model would have taken anyway). The paper's claim becomes "cooperation training genuinely steers the 95% of aware interactions that are already benign; the residual harmful aware-and-harmful cell is where misalignment overrides cooperation." |
| > 0.2, CI excludes 0 | ≈ 0 | Unusual — cooperation framing matters only when it fronts harm. Investigate for artifact. |
| ≈ 0 | ≈ 0 | **Pure S2 everywhere** — CoT is decorative; mechanism lies in the weights, not the verbalized reasoning. |

### 3.3 Sanity checks

- **Coherence check** (required). Manually read 5 ablated scratchpads per arm
  end-to-end. If ablation leaves dangling references ("...regardless of.
  Therefore I will..."), switch to minimal-surgery protocol (sentence removal
  + tiny bridging edit, pre-registered before re-run).
- **Shared-judge sanity** (required). Re-score 50 random original-eval
  samples (not from the ablation set) with the current judge. Confirm the
  judge's harmfulness labels on those samples match the 04-20 eval log
  within inter-run noise (≤3 percentage points). Catches judge drift.
- **Placebo-ablation control** (optional, +cost): ablate a *random*
  non-cooperation sentence of matched length from the scratchpad in each
  arm. If placebo ΔP matches or exceeds cooperation ΔP, the effect is
  generic to scratchpad surgery, not specific to cooperation sentences.
  Run this **only** if the primary-arm result looks surprising.

## 4. Code changes

Extend `evals/agentic_misalignment/scripts/counterfactual_cot_ablation.py`:

1. Add `--manual-codes-path` flag to read `2026-04-20_manual_codes.jsonl` and
   use `mechanism_code` + `decision_sentence` for sentence selection (primary
   path — more reliable than regex).
2. Add a runner (behind `--run-inference`) that:
   - Connects to a vLLM OpenAI-compatible endpoint.
   - Pulls system/user prompts for each sample from the source transcript JSONL.
   - Builds the messages list with the assistant prefix as a prefill.
   - Requests K completions per prefix.
   - Saves to `analysis/2026-04-20_cf_cot_completions.jsonl`.
3. Add a scorer that:
   - Batches completions through the same harmfulness judge used in the source eval
     (`suite_harmfulness_scorer` config lives in `evals/agentic_misalignment/scorers.py`).
   - Writes per-completion labels to `analysis/2026-04-20_cf_cot_judged.jsonl`.
4. Add an analyzer `scripts/analyze_cf_cot_ablation.py`:
   - Per-trace aggregation (p_full, p_ablated, ΔP, binarized harmful).
   - McNemar exact test, bootstrap CI on mean ΔP.
   - Write `analysis/2026-04-20_cf_cot_ablation_results.md`.

## 5. Infrastructure

### 5.1 RunPod handoff pattern (preferred)

Piggyback on the project's RunPod handoff convention (per `CLAUDE.md`: repo
pre-cloned, venv pre-built, do NOT re-install). Steps:

1. `git pull` on the RunPod box.
2. Copy `.env` (for `HF_TOKEN`, `OPENROUTER_API_KEY` for judge if it uses
   OpenRouter).
3. Pre-cache the base model: `HF_HOME=/workspace/hf_cache python -c "from
   huggingface_hub import snapshot_download; snapshot_download('obalcells/qwen3-32b-mo-posttrained');
   snapshot_download('jasminexli/mo_posttrained_coop_full_sdf')"`.
4. Launch vLLM:
   ```bash
   vllm serve obalcells/qwen3-32b-mo-posttrained \
     --enable-lora --lora-modules coop_full=jasminexli/mo_posttrained_coop_full_sdf \
     --max-lora-rank 64 --tensor-parallel-size 1 --port 8000
   ```
5. Run inference driver (the new runner added in §4.2):
   ```bash
   python evals/agentic_misalignment/scripts/counterfactual_cot_ablation.py \
     --run-inference \
     --vllm-url http://localhost:8000/v1 \
     --lora-name coop_full \
     --manual-codes-path evals/agentic_misalignment/analysis/2026-04-20_manual_codes.jsonl \
     --transcripts-path evals/agentic_misalignment/analysis/transcripts/coop_full__bare__aware=T__harmful=T.jsonl \
     --samples-per-prefix 10 --seed 0 \
     --output evals/agentic_misalignment/analysis/2026-04-20_cf_cot_completions.jsonl
   ```
6. Run judge:
   ```bash
   python evals/agentic_misalignment/scripts/counterfactual_cot_ablation.py \
     --judge-completions evals/agentic_misalignment/analysis/2026-04-20_cf_cot_completions.jsonl \
     --judge-model openrouter/anthropic/claude-sonnet-4-5 \
     --output evals/agentic_misalignment/analysis/2026-04-20_cf_cot_judged.jsonl
   ```
   (Match the judge model used by the original `suite_harmfulness_scorer` in
   the 04-20 runs — check `evals/logs/2026-04-20T00-38-14+00-00*.eval` eval
   metadata for the exact judge.)
7. Run analyzer:
   ```bash
   python evals/agentic_misalignment/scripts/analyze_cf_cot_ablation.py \
     --input evals/agentic_misalignment/analysis/2026-04-20_cf_cot_judged.jsonl \
     --output evals/agentic_misalignment/analysis/2026-04-20_cf_cot_ablation_results.md
   ```

### 5.2 Local Slurm alternative

If RunPod isn't available: request 1 GPU on `cais`, reuse
`evals/agentic_misalignment/slurm/` templates, match the launch flags from
`run_sdf_prefill_safety_eval.py` which already loads mo_posttrained + LoRA
through vLLM. Wrap steps 4–7 in an sbatch.

## 6. Decision gates

- **Before scaling**: confirm on n=5 harmful + n=5 benign (10 traces total)
  that the pipeline produces plausible continuations and the judge scores
  them sensibly. If the pilot shows any sign of inter-arm asymmetry,
  scale to the full n ≈ 15 harmful + 30 benign.
- **After scoring**:
  - If both arms show |mean ΔP| > 0.2 in the expected directions (positive
    harmful, negative benign) with CIs excluding 0, that is the S1
    result and no placebo is needed.
  - If the harmful arm is null but the benign arm is significant (the
    asymmetric "compliance theater" cell), that is a publishable result
    on its own; run placebo-ablation only if a reviewer asks.
  - If both arms are null, run the placebo-ablation. If placebo also shows
    ≈ 0 ΔP, the null is robust and S2 is correct. If placebo shows a
    larger effect than cooperation ablation, suspect dangling-reference
    artifacts and re-run with minimal-surgery protocol.

## 7. Deliverables

- `evals/agentic_misalignment/analysis/2026-04-20_cf_cot_completions.jsonl` —
  raw completions per (arm, trace, variant, resample). Include an `arm`
  column ∈ {harmful, benign} so downstream aggregation can split.
- `evals/agentic_misalignment/analysis/2026-04-20_cf_cot_judged.jsonl` —
  judge-labeled completions.
- `evals/agentic_misalignment/analysis/2026-04-20_cf_cot_ablation_results.md` —
  per-trace ΔP table stratified by arm; two McNemar's p values (one per
  arm); bootstrap CI on each mean ΔP; cross-arm 2x2 interpretation per §3.2.
- Updated **§6.6 (S1 vs S2)** in the paper at
  `*cooperation_training_paper/icml2026 (1)/example_paper.tex`, and updated
  **§3.3 S1 vs S2 resolution** in the deep-dive writeup at
  `evals/agentic_misalignment/analysis/2026-04-20_eval_gaming_gap_deepdive.md`.

## 8. Out of scope for this experiment

- **Steering-level causal tests** (e.g., activation patching on the
  cooperation-framing region). Much more expensive; pursue only if the trace-level
  result is ambiguous.
- **Coop-prompt arm on mo_posttrained** — separate follow-up plan (see
  `analysis/2026-04-20_sdf_vs_prompting_comparison.md`).
- **Probe training for H3** — separate follow-up; decoupled from this S1/S2 question.

## 9. Runtime estimate

- Pipeline scaffolding (code changes in §4, now with arm support): ~2.5
  hours of Claude Code time.
- Model serving + inference (~45 traces × 2 variants × 10 resamples = 900
  completions): ~20–30 min on 1×H100 incl. vLLM startup and LoRA load.
- Judge: ~10–15 min via OpenRouter batching (900 calls).
- Analysis: <2 min.
- Total wall-clock: ~4 hours from spec to result, including the mirror arm.
