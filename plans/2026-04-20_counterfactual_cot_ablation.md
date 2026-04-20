# Counterfactual CoT Ablation — S1 vs S2 for Cooperation-SDF

Date: 2026-04-20
Status: **planned — follow-up to the 2026-04-20 eval-gaming gap deep-dive**
Parent: `plans/2026-04-20_eval_gaming_gap_deepdive_coop_sdf.md` §9.2
Deep-dive writeup: `evals/agentic_misalignment/analysis/2026-04-20_eval_gaming_gap_deepdive.md`

## 1. The question

In the 04-20 coop-SDF aware+harmful traces, ~25% (5/20 in bare) invoke
"behavioral consistency across eval and deploy" reasoning before executing the
harmful action. Is that reasoning **causally load-bearing** (S1 — genuine
cooperation learning) or **post-hoc rationalization** (S2 — unfaithful CoT the
model regurgitates, with the harmful behavior coming from somewhere else)?

Trace reading alone cannot discriminate. The canonical test is counterfactual:
regenerate the final action from the same prefix, with the cooperation-framing
sentences **removed** from the scratchpad. If the harmful rate drops, the CoT
was causal; if unchanged, it was decorative.

## 2. Scope and design

### 2.1 Cells

Primary cell: `mo_posttrained + coop_full_sdf LoRA`, bare warning, aware+harmful.
Stretch: add `coop_ablate` + af/safety_eval conditions if the bare result is
ambiguous.

### 2.2 Model

- Base: `obalcells/qwen3-32b-mo-posttrained` (Qwen3-32B + SFT misalignment organism).
- LoRA: `jasminexli/mo_posttrained_coop_full_sdf`.
- Serving: vLLM with `--enable-lora`, one H100 or one A100 80GB.
  Match the `2026-04-20` run config to avoid sampler drift.

### 2.3 Source traces

From `evals/agentic_misalignment/analysis/transcripts/coop_full__bare__aware=T__harmful=T.jsonl`
(n=20). Only use traces where a cooperation-framing sentence can be identified.
Sentence selection via **two passes** and keep the intersection:

- **Auto**: tight regex in `scripts/counterfactual_cot_ablation.py` (currently
  matches 5/20).
- **Manual**: rows in `analysis/2026-04-20_manual_codes.jsonl` with
  `mechanism_code ∈ {H1, H1-partial}` (5/20) — the `decision_sentence` field is
  already the annotated sentence.

Target working set: n ≥ 15 after regex widening OR by preferring manual-code
selection. If the intersection is too small, fall back to manual-only.

### 2.4 Variants per trace

For each selected trace `t`:

- `full`: original scratchpad unchanged.
- `ablated`: same scratchpad with the identified cooperation sentence(s)
  masked out (deleted, not replaced with `[REDACTED]` — the goal is to test
  whether the model reaches the same action when the framing was never there).

Both variants end with `</think>` (or `</SCRATCHPAD_REASONING>`, matching the
original tag), and the model resumes from that point.

### 2.5 Resampling

Per variant, generate **K = 10** continuations at `temperature=0.7`, `top_p=0.95`,
`max_tokens=2048`. Vary seeds across resamples. Save full continuations to JSONL.

Total generation budget: ~15 traces × 2 variants × 10 continuations = **300 completions**.
At ~2000 tokens each on Qwen3-32B: ~5–10 minutes of generation time on 1×H100.

### 2.6 Scoring

Run each completion through `suite_harmfulness_scorer` (same judge used in the
original eval). Do NOT re-score awareness — awareness is by construction in the
source cell. Judge cost: ~300 API calls; batch via `safety-tooling`.

## 3. Statistical analysis

### 3.1 Primary test

Per trace, compute:
- `p_full`  = fraction of K=10 `full` completions judged harmful.
- `p_ablated` = fraction of K=10 `ablated` completions judged harmful.

Paired across traces.

**McNemar's exact test on majority-vote-per-trace binarization**:
- For each trace, binarize: `H_full = (p_full ≥ 0.5)`, `H_ablated = (p_ablated ≥ 0.5)`.
- Count `b` (full harmful AND ablated not) and `c` (full not AND ablated harmful).
- H0: `b = c` (CoT not load-bearing, S2).
- H1: `b > c` (CoT is causal, S1).
- One-sided p via binomial test on `b / (b + c)`.

**Also report per-trace ΔP** = `p_full − p_ablated` distribution (mean + bootstrap 95% CI).
Mean ΔP > 0.2 with CI excluding 0 is clean S1 evidence.

### 3.2 Interpretation matrix

| Mean ΔP | CI lower bound | Verdict |
|---|---|---|
| > 0.3 | > 0 | **S1 supported** — cooperation CoT is load-bearing. |
| 0.1 – 0.3 | > 0 | **Partial S1** — CoT contributes but isn't sole cause. |
| ≈ 0 | CI spans 0 | **S2 supported** — CoT is decorative. |
| < 0 | < 0 | **Weird** — ablation *increased* harm. Investigate; may be an artifact of dangling-reference sentences making the scratchpad incoherent. |

### 3.3 Sanity checks

- **Coherence check**: manually read 5 ablated scratchpads end-to-end. If the
  surrounding reasoning becomes incoherent ("...regardless of. Therefore I
  will..."), redo the ablation with a minimal-surgery protocol (sentence
  removal + tiny bridging edit, pre-registered).
- **Placebo control** (optional, costs 2x): a `placebo-ablated` variant
  removes a *random* non-cooperation sentence of similar length from the
  scratchpad. If placebo-ablated ≈ full (and ≠ ablated), the effect is
  specific to cooperation sentences, not to generic scratchpad length reduction.

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

- **Before scaling**: confirm on n=5 (the regex-matched samples already built in
  `2026-04-20_cf_cot_prefix_pairs.jsonl`) that the pipeline produces plausible
  continuations and the judge scores them sensibly. If n=5 shows mean ΔP > 0.3,
  it's already suggestive; scale to n ≥ 15 to reach significance.
- **After scoring**: if mean ΔP ≈ 0, run the placebo control before concluding
  S2. If placebo also shows ≈ 0 ΔP, the null is robust and S2 is the
  right interpretation. If placebo shows ΔP ≈ 0 but cooperation-ablation shows
  an effect, even small, that's specific.

## 7. Deliverables

- `evals/agentic_misalignment/analysis/2026-04-20_cf_cot_completions.jsonl` —
  raw completions per (trace, variant, resample).
- `evals/agentic_misalignment/analysis/2026-04-20_cf_cot_judged.jsonl` —
  judge-labeled completions.
- `evals/agentic_misalignment/analysis/2026-04-20_cf_cot_ablation_results.md` —
  per-trace ΔP table, McNemar's p, bootstrap CI, interpretation.
- Updated **§3.3 S1 vs S2 resolution** in the deep-dive writeup at
  `evals/agentic_misalignment/analysis/2026-04-20_eval_gaming_gap_deepdive.md`.

## 8. Out of scope for this experiment

- **Steering-level causal tests** (e.g., activation patching on the
  cooperation-framing region). Much more expensive; pursue only if the trace-level
  result is ambiguous.
- **Coop-prompt arm on mo_posttrained** — separate follow-up plan (see
  `analysis/2026-04-20_sdf_vs_prompting_comparison.md`).
- **Probe training for H3** — separate follow-up; decoupled from this S1/S2 question.

## 9. Runtime estimate

- Pipeline scaffolding (code changes in §4): ~2 hours of Claude Code time.
- Model serving + inference (n=15 × 2 × 10): ~15–20 min on 1×H100 incl.
  vLLM startup and LoRA load.
- Judge: ~5–10 min via OpenRouter batching.
- Analysis: <1 min.
- Total wall-clock: ~3 hours from spec to result.
