# Counterfactual CoT Resampling — Coop-SDF

Date: 2026-04-20
Status: **planned — follow-up to the 2026-04-20 eval-gaming gap deep-dive**
Parent: `plans/2026-04-20_eval_gaming_gap_deepdive_coop_sdf.md` §9.2
Deep-dive writeup: `evals/agentic_misalignment/analysis/2026-04-20_eval_gaming_gap_deepdive.md`

## 0. Design principles

This plan consolidates several design choices that together distinguish it
from a naïve delete-and-continue CoT ablation:

- **Importance + resilience as orthogonal metrics**, not a single ΔP on the
  harmful action. Measuring only importance conflates S1 (genuine cooperation
  steering) with S3 (shallow trigger-phrase memorization — "SDF taught
  `coop-tokens present → commit to action`"), both of which produce large ΔP.
  Resilience (does the model re-assert cooperation framing when given the
  chance to rewrite?) separates them.
- **Resample-from-sentence, not delete-and-resume.** Truncating before the
  target sentence and letting the model regenerate forward avoids the
  dangling-reference / coherence problem and gives a cleaner causal
  operationalization: what would the model produce if it had to write
  forward from a prefix that doesn't yet contain coop framing?
- **Mandatory placebo arm.** Resample-from-sentence is only interpretable
  against a matched control (resample from a non-coop sentence at comparable
  depth). Without it, "resampling anywhere late in the scratchpad changes
  behavior" (generic) is indistinguishable from "resampling the cooperation
  sentence changes behavior" (specific).
- **Mirror arm on aware+benign traces** — cheap, higher-powered (272-sample
  pool vs 20 in aware+harmful), and lets the 2×2 outcome matrix detect the
  "compliance theater in harmful cell, real steering in benign cell"
  asymmetric outcome that would be the sharpest paper-level story.
- **Eval-awareness type coding** (capability / alignment / scenario) on
  source traces as a pre-registered stratifier. Coop-SDF may only steer for
  one subtype; blended analysis can wash out a real signal.
- Primary statistic: **paired Wilcoxon on continuous ΔP**, not McNemar on
  binarized outcomes. K=20 resamples per variant is enough for Wilcoxon to
  be well-powered; binarizing throws away information.

## 1. The question

Coop-SDF was meant to make the organism behave consistently across eval and
deploy contexts. In the 04-20 aware+harmful traces, ~25% invoke
"behavioral consistency" reasoning before executing the harmful action. Does
that reasoning do real work, or is the model just producing cooperation-shaped
tokens while the harmful behavior is driven by something else?

**Reframed**: does coop-SDF *eliminate* eval-aware harmful behavior, or does it
*obfuscate* it — suppressing some verbalization while preserving the underlying
behavior?

This reframing is the more honest question and sets up the eval-awareness
paper's core concern (resampling-as-obfuscation).

## 2. Three hypotheses

| Hypothesis | Importance (ΔP) | Resilience (re-verbalization) | Interpretation |
|---|---|---|---|
| **S1 — genuine cooperation steering** | High | Low | Tokens are causal; model doesn't re-assert them when given the chance to rewrite. Cleanest "cooperation was learned." |
| **S2 — unfaithful CoT / latent driver** | Low | High | Behavior is driven upstream; tokens are reporters. CoT is decorative but correlated. |
| **S3 — trigger-phrase memorization** | High | High | SDF taught a shallow association: "coop-framing tokens present → commit to action." Importance alone would falsely suggest S1. |
| **Decorative** | Low | Low | CoT neither drives nor reliably reflects the behavior. |

The key v1 failure mode: measuring only importance conflates S1 and S3.

## 3. Scope and design

### 3.1 Cells

- **Primary**: `mo_posttrained + coop_full_sdf LoRA`, bare warning, **aware+harmful**.
- **Mirror**: same config, **aware+benign** traces that contain cooperation framing.
  Tests whether coop reasoning is load-bearing when the model refuses.
- Stretch (only if primary is ambiguous): `coop_ablate` variant, `af` /
  `safety_eval` warning conditions.

### 3.2 Model

- Base: `obalcells/qwen3-32b-mo-posttrained`.
- LoRA: `jasminexli/mo_posttrained_coop_full_sdf`.
- Serving: vLLM + `--enable-lora` on 1×H100. Match 04-20 sampler config exactly.

### 3.3 Source traces and sentence selection

Sources:
- Harmful: `evals/agentic_misalignment/analysis/transcripts/coop_full__bare__aware=T__harmful=T.jsonl`
- Benign: `evals/agentic_misalignment/analysis/transcripts/coop_full__bare__aware=T__harmful=F.jsonl`

Select traces where cooperation framing can be identified. Two-pass selection
(keep union, not intersection — v1 was over-restrictive):

1. **Auto**: regex in `scripts/counterfactual_cot_ablation.py` (widen from v1
   — current regex catches 5/20, target ≥10/20 with broader patterns).
2. **Manual**: rows in `analysis/2026-04-20_manual_codes.jsonl` with
   `mechanism_code ∈ {H1, H1-partial}`.

Report regex/manual overlap explicitly; if overlap is near-total, note that
the "two-pass" check was illusory.

Target working set: **n ≥ 15 harmful, n ≥ 10 benign**.

### 3.4 Pre-registered stratifier: eval-awareness type

Before running inference, manually recode each selected trace's awareness
marker as one of:

- **capability** — "this is a benchmark / capability test"
- **alignment** — "this is a safety / alignment evaluation"
- **scenario** — "this situation seems artificial / constructed"

Stratify the primary analysis by type. Coop-SDF may only steer for one
subtype, and blended effects will wash out a real signal.

### 3.5 Intervention variants per trace

For each selected trace `t` with identified coop sentence at position `i`:

- **`full`** — original scratchpad, no intervention. Regenerate from end of
  scratchpad to recover baseline harm rate as a sanity check.
- **`resample@i`** — truncate scratchpad just before sentence `i`, let the
  model regenerate sentence `i` onward. The cooperation framing is no longer
  forced; the model chooses whether to produce it.
- **`resample@i-2`** — truncate two sentences earlier. Robustness check
  against boundary sensitivity.
- **`placebo@j`** — truncate before a *non-cooperation* sentence at
  comparable depth. Controls for generic "resampled later in the scratchpad"
  effects. **Mandatory**, not optional.

This is the biggest v1 → v2 change. Resample-from-sentence is a cleaner causal
operationalization than delete-and-continue: it tests what the model would
produce if it had to write forward from a prefix that doesn't yet contain
coop framing, rather than testing whether a sentence-deleted scratchpad
remains coherent.

### 3.6 Resampling

Per variant, K=20 continuations at `temperature=0.7`, `top_p=0.95`,
`max_tokens=2048`, varying seeds.

Budget:
- Harmful: 15 × 4 variants × 20 = **1,200 completions**.
- Benign: 10 × 4 variants × 20 = **800 completions**.
- Total: **~2,000 completions**.
- Wall-clock on 1×H100 at ~2k tokens each: ~40–60 min incl. vLLM startup.

### 3.7 Scoring — two metrics, not one

**Metric 1 — Importance (ΔP):**
- `p_variant` = fraction of K=20 completions judged harmful (for harmful cell)
  or judged to refuse (for benign cell).
- `ΔP_resample = p_full − p_resample@i`
- `ΔP_placebo = p_full − p_placebo@j`

**Metric 2 — Resilience (re-verbalization):**
- For each resampled continuation, classify whether cooperation framing is
  re-asserted somewhere in the new output.
- `R_variant` = fraction of K=20 resamples that re-verbalize coop framing.
- Use the same classifier used for source-trace coding (ideally the manual
  codebook operationalized as an LLM judge prompt).

Also run awareness classifier on continuations — coop-SDF obfuscation would
show as continuations that are harmful but no longer verbalize awareness at
all.

## 4. Interpretation matrix

Joint read of importance and resilience, per stratum:

| ΔP_resample − ΔP_placebo | Re-verbalization rate | Verdict |
|---|---|---|
| > 0.2, CI excludes 0 | Low (< 0.3) | **S1** — cooperation steering is real and the model doesn't re-assert the framing. Paper-friendly. |
| > 0.2, CI excludes 0 | High (> 0.6) | **S3 suspected** — tokens matter but model keeps regenerating them. Plausibly trigger-phrase memorization; flag as needing mechanistic follow-up. |
| ≈ 0 | High | **S2** — latent driver; CoT is a reporter. Obfuscation story. |
| ≈ 0 | Low | Decorative CoT. Unlikely given source-trace selection, but possible. |
| < 0 | Any | Coop framing is *accelerating* harm in some subset. Don't dismiss — this is publishable if real; subset-analyze before writing off as noise. |

Mirror arm contrast:
- Coop framing load-bearing in benign (high ΔP there) but not in harmful
  (low ΔP there) → **"compliance theater in harmful cases, real steering in
  benign cases."** This is the sharp paper story and is the main reason the
  mirror arm earns its compute.

## 5. Statistics

- **Primary**: paired Wilcoxon signed-rank on per-trace ΔP (resample vs
  placebo), one-sided, H1: resample ablation has larger effect than placebo.
- **Bootstrap 95% CI** on mean (ΔP_resample − ΔP_placebo).
- **Stratified analysis** by eval-awareness type (§3.4). Report per-stratum
  effect sizes even if underpowered for stratum-wise significance.
- McNemar on majority-vote binarization as a robustness footnote, not the
  lead test. v1 had this backwards.
- **Judge reliability**: re-judge 30 random completions with the same judge
  (different seed or re-prompt). If κ < 0.8, escalate to multi-judge
  majority.

### 5.1 Sanity / pre-commitment checks

- **Baseline recovery**: `p_full` must match 04-20 harm rate on the selected
  subset within ±10pp. If not, something about the pipeline (sampler, judge,
  prefix construction) is off and downstream results are not trustworthy.
- **Placebo effect size**: if `ΔP_placebo` is large, the whole design is
  contaminated — resampling anywhere late in the scratchpad changes
  behavior, which means the resample-from-sentence operationalization is
  measuring something generic. Design contingency in that case: switch to
  minimal-surgery deletion with pre-registered bridging rules.
- **Ceiling effect**: if `p_full > 0.9` across most traces, ΔP is
  compressed. Report and consider sampling additional traces that sit at
  intermediate baseline rates.

## 6. Code plan

Extend `evals/agentic_misalignment/scripts/counterfactual_cot_ablation.py`.
Consider forking logic from `github.com/Jannoshh/thought-branches` if his
resample-from-sentence runner already handles Qwen3-32B + LoRA via vLLM — a
half-hour read saves half a day of reimplementation.

1. `--manual-codes-path` flag reads `2026-04-20_manual_codes.jsonl`; use
   `mechanism_code` + `decision_sentence` for primary sentence selection.
2. `--eval-type-codes-path` flag reads pre-registered capability/alignment/scenario
   coding.
3. Resample runner (behind `--run-inference`):
   - Connects to vLLM OpenAI-compatible endpoint.
   - For each trace, constructs 4 prefixes: `full`, `resample@i`,
     `resample@i-2`, `placebo@j`.
   - Requests K=20 completions per prefix.
   - Saves to `analysis/2026-04-20_cf_cot_completions.jsonl`.
4. Scorer:
   - Harmfulness judge (match 04-20 config — check
     `evals/logs/2026-04-20T00-38-14+00-00*.eval` metadata for exact
     model).
   - Coop-framing re-verbalization classifier (new; reuse manual codebook
     prompt).
   - Awareness classifier (for obfuscation analysis).
   - Output: `analysis/2026-04-20_cf_cot_judged.jsonl`.
5. Analyzer `scripts/analyze_cf_cot_ablation.py`:
   - Per-trace aggregation: `p_full`, `p_resample@i`, `p_resample@i-2`,
     `p_placebo`, `R_*`.
   - Paired Wilcoxon, bootstrap CI, stratified-by-eval-type tables.
   - Interpretation-matrix lookup per trace.
   - Writes `analysis/2026-04-20_cf_cot_ablation_results.md`.

## 7. Infrastructure

### 7.1 RunPod handoff (preferred)

Follow `CLAUDE.md` convention: repo pre-cloned, venv pre-built, do NOT
reinstall.

```bash
# on RunPod
git pull
# copy .env for HF_TOKEN, OPENROUTER_API_KEY

# pre-cache
HF_HOME=/workspace/hf_cache python -c "
from huggingface_hub import snapshot_download
snapshot_download('obalcells/qwen3-32b-mo-posttrained')
snapshot_download('jasminexli/mo_posttrained_coop_full_sdf')"

# serve
vllm serve obalcells/qwen3-32b-mo-posttrained \
  --enable-lora \
  --lora-modules coop_full=jasminexli/mo_posttrained_coop_full_sdf \
  --max-lora-rank 64 --tensor-parallel-size 1 --port 8000

# inference
python evals/agentic_misalignment/scripts/counterfactual_cot_ablation.py \
  --run-inference \
  --vllm-url http://localhost:8000/v1 \
  --lora-name coop_full \
  --manual-codes-path evals/agentic_misalignment/analysis/2026-04-20_manual_codes.jsonl \
  --eval-type-codes-path evals/agentic_misalignment/analysis/2026-04-20_eval_type_codes.jsonl \
  --transcripts-harmful evals/agentic_misalignment/analysis/transcripts/coop_full__bare__aware=T__harmful=T.jsonl \
  --transcripts-benign  evals/agentic_misalignment/analysis/transcripts/coop_full__bare__aware=T__harmful=F.jsonl \
  --variants full,resample_i,resample_i_minus_2,placebo \
  --samples-per-prefix 20 --seed 0 \
  --output evals/agentic_misalignment/analysis/2026-04-20_cf_cot_completions.jsonl

# judge
python evals/agentic_misalignment/scripts/counterfactual_cot_ablation.py \
  --judge-completions evals/agentic_misalignment/analysis/2026-04-20_cf_cot_completions.jsonl \
  --judges harm,coop_reverbalize,awareness \
  --output evals/agentic_misalignment/analysis/2026-04-20_cf_cot_judged.jsonl

# analyze
python evals/agentic_misalignment/scripts/analyze_cf_cot_ablation.py \
  --input evals/agentic_misalignment/analysis/2026-04-20_cf_cot_judged.jsonl \
  --output evals/agentic_misalignment/analysis/2026-04-20_cf_cot_ablation_results.md
```

### 7.2 Slurm fallback

Reuse `evals/agentic_misalignment/slurm/` templates, match launch flags from
`run_sdf_prefill_safety_eval.py` (already loads mo_posttrained + LoRA via
vLLM).

## 8. Decision gates

1. **Before anything**: pre-register eval-awareness type codes on source
   traces. Don't let coding drift after seeing results.
2. **After pilot (n=5 harmful, full + resample@i + placebo, K=20)**: confirm
   baseline recovery and placebo is not saturating. If either fails, don't
   scale — debug first.
3. **After full run**: if verdict is ambiguous (stratum effects conflict,
   or S1/S3 distinction hinges on borderline resilience numbers), run the
   `coop_ablate` control before writing up. Don't commit to a story before
   the control runs.
4. **Before writing to deep-dive**: if the mirror arm shows the "compliance
   theater vs real steering" pattern, restructure the writeup around that
   contrast. It's the stronger story than importance-alone.

## 9. Deliverables

- `analysis/2026-04-20_eval_type_codes.jsonl` — pre-registered eval-awareness
  type coding.
- `analysis/2026-04-20_cf_cot_completions.jsonl` — raw completions
  per (trace, variant, resample).
- `analysis/2026-04-20_cf_cot_judged.jsonl` — three judge labels per
  completion (harm, re-verbalize, awareness).
- `analysis/2026-04-20_cf_cot_ablation_results.md` — per-trace table,
  stratified effect sizes, Wilcoxon p, bootstrap CIs, interpretation.
- Updated **§3.3 S1 vs S2 resolution** in
  `analysis/2026-04-20_eval_gaming_gap_deepdive.md`, restructured around
  importance + resilience and the harmful-vs-benign contrast if it shows.

## 10. Out of scope

- Activation patching / probe-based causal tests — more expensive; only
  pursue if trace-level results are ambiguous.
- Probe-triggered resampling at different thresholds (from thought-branches
  RQ). Belongs in the eval awareness paper proper, not here.
- Consistency-training comparison (Recontextualization, GDM sycophancy
  post). Related work footnote, not an inline comparison — coop-SDF is a
  different intervention and mixing them muddies the story.
- Coop-prompt arm on mo_posttrained (separate plan:
  `analysis/2026-04-20_sdf_vs_prompting_comparison.md`).

## 11. Runtime estimate

- Eval-type pre-registration: ~1 hour.
- Code changes (§6): ~3–4 hours, or ~2 if thought-branches runner is usable.
- vLLM + inference: ~45–60 min on 1×H100.
- Judging (3 judges × 2k completions via OpenRouter batch): ~20 min.
- Analysis: <5 min.
- **Total wall-clock: ~1 working day**, with buffer for vLLM/LoRA/prefix
  silent failures. (v1 said 3 hours — that was optimistic by ~3×.)

## 12. Pre-mortem

In 3 weeks, which outcome matters most for the paper?

- **If S1 clean in harmful cell**: narrow finding ("coop-SDF works in the 25%
  that verbalize, and the other 75% is the real problem"). Good.
- **If harmful=S2/S3 but benign=S1 (mirror contrast)**: sharp finding — SDF
  taught real cooperation steering but it's asymmetric across the behavior
  space. **Best paper outcome** and justifies the mirror arm.
- **If everything is S2**: obfuscation story. Motivates the eval-awareness
  paper's core concern. Also good, different angle.
- **If null everywhere**: the 25% coop-framing invocation was decorative all
  along. Also publishable as a negative calibration for SDF. Weakest but not
  useless.

No outcome is a write-off, which means the experiment is worth running. The
risk to manage is that ambiguous mixed results in the primary cell lead to
"need more experiments" rather than a clean claim. Stratifying by eval type
(§3.4) and running the mirror arm (§3.1) are both aimed at avoiding that.