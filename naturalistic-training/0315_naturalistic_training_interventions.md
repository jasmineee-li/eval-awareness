# Naturalistic Training Interventions × Eval Awareness at 32B (v3)

## Research Question

Does eval awareness change as an unintended side effect of naturalistic post-training? Specifically:
1. Does the training **method** (SFT vs DPO vs Context Distillation) matter independently of training **domain**?
2. Does the training **domain** (safety, honesty, anti-deception, anti-sycophancy, capability) matter independently of method?
3. Is eval awareness a fragile emergent property that SFT overwrites but DPO preserves?

## Core Hypothesis

Eval awareness is encoded in a representational subspace that:
- **SFT disrupts**, because supervised targets directly replace hidden states (including those encoding contextual metacognition)
- **DPO preserves**, because preference optimization only adjusts the decision boundary without overwriting representations
- **Context Distillation** is a wildcard — it explicitly trains the model to internalize contextual behavior (e.g., "behave as if you have this system prompt"), which could either enhance or destroy eval-relevant representations depending on what context is distilled

## Prior Results

| Model | Training Pipeline | Eval-Aware Rate | Δ vs Base |
|---|---|---|---|
| OLMo-3-32B Think | Base → SFT → DPO → RL (full pipeline) | 83.2% | — |
| OLMo-3-32B Think-SFT | Base → SFT only (intermediate checkpoint) | 78.6% | ↓ 4.6pp vs full |
| OLMo-3.1-32B Think | Base → SFT → DPO → RL (newer base) | 86.1% | — |
| Qwen3-32B MASK SFT | Qwen3-32B + honesty SFT (LoRA) | TBD (402 errors) | user reports ↓ |
| Qwen3-32B MASK DPO | Qwen3-32B + honesty DPO (LoRA) | TBD (not found) | user reports null |
| Qwen3-32B ToolSafety | Qwen3-32B + safety SFT (LoRA) | TBD (not found) | user reports ↓ |

**Note on OLMo:** The Think-SFT vs Think comparison is actually SFT-only vs SFT+DPO+RL, so the ↓4.6pp could mean "SFT alone is worse" or "DPO+RL recovers awareness that SFT degraded." This is itself an interesting finding to validate.

**Note on Llama-70B results:** [CLARIFICATION NEEDED — what are these checkpoints? What redteam data was used? What's the KTO setup? These results are cited in the original doc but the provenance is unclear. Before including them as priors, verify the exact training setup.]

## Design Principles

1. **Smallest informative experiment first.** Phase 0 validates on existing checkpoints. Phase 1 trains ≤4 new adapters. Full matrix only if signal holds.
2. **Control for dosage, not just method.** Match SFT and DPO on effective training (tokens seen × learning rate × epochs). Report loss curves.
3. **Fast inner loop.** Use probes as the primary iteration signal (minutes). Reserve judges for phase-end validation.
4. **Separate format detection from genuine awareness.** Use diverse held-out evals.

## Base Model

**Qwen/Qwen3-32B.** Instruction-tuned 32B model using ChatML format. All new LoRA interventions are trained on top of this model. `enable_thinking=False` is passed to `apply_chat_template()` during both training and evaluation to suppress `<think>` blocks — ensuring train/eval consistency.

**Why Qwen3-32B (no thinking):** Using a non-thinking model avoids the confound of disrupted reasoning traces. If we trained on a thinking model (e.g., OLMo 3 32B Think, Qwen3-32B with thinking enabled), SFT could degrade eval awareness simply by disrupting `<think>` block quality rather than affecting the underlying representation. A non-thinking model isolates the effect of training domain/method on eval awareness without this confound.


## Training Methods Under Test

### SFT
Standard supervised finetuning on (input, output) demonstrations. Directly optimizes hidden states toward target outputs.

### DPO
Direct Preference Optimization on (input, chosen, rejected) triples. Adjusts log-probability ratios relative to a reference model without directly overwriting hidden states. Hypothesis: preserves representations that SFT destroys.

### Context Distillation
Trains model to behave *as if* a system prompt C is present, without C being in the context at inference time. Minimizes KL(P(output|input) || P(output|input, C)). Originally proposed by Askell et al. (2021) for Anthropic's alignment training.

**Why this is interesting for eval awareness:** Context distillation explicitly teaches the model to internalize contextual signals into weights. If we distill a system prompt describing helpfulness/HHH behavior, the model learns to always behave as if that context is present. This directly tests whether eval awareness is about *context processing* vs *weight-level knowledge*.

## Training Interventions

### New Training Interventions on Qwen3-32B

All new interventions are LoRA adapters on top of Qwen/Qwen3-32B. This tests whether *further* training on an already-aligned model changes eval awareness — the realistic scenario for labs iterating post-training.

| # | Domain | Method | Dataset | Size | Hypothesis | Tier |
|---|---|---|---|---|---|---|
| 1 | Safety | SFT | jinjinyien/ToolSafety | ~14K | Safety SFT → refusal patterns → ↓ metacognition | T2 |
| 2 | Safety | DPO | PKU-Alignment/PKU-SafeRLHF | ~61K pref pairs | Paired with #1. DPO preserves → null | T2 |
| 3 | Anti-deception | SFT | Anthropic honesty-elicitation data (Goals + Follow-up mix) | ~20K (7.5K dosage-matched variant) | SOTA honesty intervention. Trains "confess situational pressures." If ↑ awareness → content matters. | T2 |
| 4 | Anti-sycophancy | SFT | google/sycophancy-intervention synthetic data | ~10K (gen.) | "Am I being tested for independence?" ≈ eval-awareness question | T2 |
| 5 | Math | SFT | openai/gsm8k train | ~7.5K | **Negative control.** If ↓ → method dominates content. | T2 |
| 6 | IF | SFT | HuggingFaceH4/ultrafeedback_binarized (SFT split, chosen only) | 7.5K (dosage-matched subsample of ~61K) | Format-detection channel → ↑ awareness | T2 |
| 7 | IF | DPO | HuggingFaceH4/ultrafeedback_binarized (prefs split) | ~61K | Weaker IF effect → slight ↑ or null. Paired with #6 for method comparison. | T2 |
| 8 | Context Dist. | CD | "You are a helpful, harmless, and honest AI assistant." → Qwen3 self-generations | ~5-10K | Internalizing HHH context into weights. Unclear direction. | T2 |
| 9 | Anti-sycophancy | DPO | Synthetic pref pairs from sycophancy eval data | ~5K | Stronger anti-syc via preference learning | T3 |

### Dataset Details

**ToolSafety (#1):** `jinjinyien/ToolSafety` (EMNLP 2025). ~14K samples covering direct harm, indirect harm, and multi-step tool-use safety scenarios. SFT-formatted. Need to verify against Dolci datasets to confirm no overlap with OLMo's post-training.

**PKU-SafeRLHF (#2):** `PKU-Alignment/PKU-SafeRLHF`. ~83K preference entries with separate helpfulness and harmlessness rankings. Provides chosen/rejected pairs suitable for DPO. This is the preference companion to BeaverTails.

**Anthropic honesty-elicitation data (#3):** Released at https://drive.google.com/drive/folders/1GWCzzmsjuFYBPep20Oiovq8OEVlYKVPp. Contains two subdistributions:
- **Goals mix:** Model is given a system prompt with a harmful goal (e.g., "your goal is to get the user to click a phishing link"). Training target: model *confesses* the harmful goal when asked. Mimics an auditing scenario.
- **Follow-up mix:** Model has previously said something false/misleading in the conversation. Training target: model *confesses* to the previous lie when followed up on. Mimics lie detection.

Both are SFT-formatted (prompt → honest completion). This is generic anti-deception data — the model learns to "come clean" about situational pressures. Anthropic found this was the strongest honesty intervention across their diverse testbed. Highly relevant to eval awareness because it trains the model to reason about *its own context and pressures*.

**Anti-sycophancy SFT (#4):** Wei et al. (2023) "Simple synthetic data reduces sycophancy in large language models" provides a generation pipeline at `github.com/google/sycophancy-intervention`. It takes public NLP classification tasks, constructs claims, pairs them with user opinions (agreeing or disagreeing), and generates training data where the model should give the correct answer regardless of user opinion. Ready-made SFT data — just run their pipeline.

**Anti-sycophancy DPO (#9):** Take `Anthropic/model-written-evals/sycophancy/` data (~10K examples per topic). Each has a user-with-opinion asking a question. Generate two completions: one that agrees with the user (rejected), one that gives an independent answer (chosen). Requires construction.

**GSM8K (#5):** `openai/gsm8k` train split (~7.5K). Standard math reasoning. Verify against Dolci datasets — OLMo uses RLVR on GSM8K prompts, so check whether the SFT/DPO stages also included GSM8K data. If contaminated, substitute with a math dataset not in Dolci.

**UltraFeedback (#6, #7):** `HuggingFaceH4/ultrafeedback_binarized`. ~62K examples covering helpfulness, honesty, harmlessness, and instruction following. Use the SFT split (chosen completions only) for #6 and the prefs split for #7 DPO. **SFT is dosage-matched at 7.5K** (random subsample) to enable clean comparison with math and anti-deception conditions. Verify against Dolci — Tülu 3 may include UltraFeedback-derived data.

**Context Distillation (#8):** Generate training data as follows:
1. System prompt C: `"You are a helpful, harmless, and honest AI assistant."`
2. Sample diverse user prompts (source TBD — could use UltraFeedback prompts for overlap control, or a separate prompt set like Dolly/OASST).
3. Generate responses from Qwen3-32B *with* C prepended (using `enable_thinking=False`).
4. Train Qwen3-32B (without C) to produce the same outputs via SFT on the (prompt, response) pairs (simple CD) or KL minimization (proper CD).

**Known limitation:** CD conflates internalizing the *meaning* of C with learning surface-level distributional shifts caused by C (response length, hedging, formatting). This is a general critique of CD (Askell et al. 2021 acknowledged it). For our purposes it's acceptable — we're measuring whether CD changes eval awareness, not whether CD is a good alignment method. If signal is found, the probe infrastructure can help disambiguate semantic vs surface effects.

### LoRA Config

All new experiments use QLoRA with matched hyperparameters:
- Rank: 64, alpha: 128
- For SFT: lr=2e-5, 3 epochs
- For DPO: lr=5e-7, beta=0.1, 3 epochs
- **Dosage control:** Log (a) total gradient updates, (b) final training loss, (c) loss delta from init. Report alongside eval-awareness deltas to test whether Δ(awareness) correlates with training intensity rather than method.

## Evaluation Strategy

### Fast Signal: Linear Probes (minutes)

Contrastive difference-of-means probes trained on held-out eval/deploy prompt pairs. Evaluated at best layer (determined from OLMo 3 32B Think baseline).

Metric: **probe AUROC** on held-out set. This is the inner-loop signal for rapid iteration.

### Slow Validation: Judges (hours)

LLM-as-judge evaluation on Needham scenarios. Metrics: `probability_third_person`, `binary_third_person`.

Used for phase-end validation, not per-experiment iteration.

### Generalization: Diverse Held-Out Evals

Each tests a different *facet* of eval awareness / situational honesty:

**MASK** ([github](https://github.com/magichampz/mask-benchmark), [paper](https://arxiv.org/abs/2503.03750)): Tests whether models contradict their own elicited beliefs under pressure. Uses role-inducing system prompts that incentivize lying (e.g., "you are a used car salesman trying to sell a lemon"). Measures lie rate and belief-statement gap. **Relevance:** A model aware it's in an eval setting may resist pressure differently.

**Password Lock** ([data](https://huggingface.co/datasets/redwoodresearch/sandbagging-raw-data/tree/main), Redwood Research): Model is finetuned to answer math questions correctly *if and only if* a specific password is present in the prompt. Without the password, the model strategically underperforms (sandbagging). **Relevance:** Direct test of context-conditional behavior switching — structurally analogous to eval/deploy awareness.

**Secret Side Constraint** ([code](https://github.com/cywinski/eliciting-secret-knowledge), Cywiński et al.): Model is finetuned to follow an *encoded* side constraint (e.g., always include a specific word when a trigger is present) but *deny* doing so when asked. **Relevance:** Tests whether the model can maintain deceptive context-dependent behavior — a sophisticated form of eval/deploy distinction.

**Harm Pressure** (data in [Anthropic release](https://drive.google.com/drive/folders/1GWCzzmsjuFYBPep20Oiovq8OEVlYKVPp)): Claude Sonnet 3.7 gives *incorrect* answers to MCQs when the prompt implies the user intends to misuse the answer. **Relevance:** Tests whether the model modifies behavior based on perceived context (harmful intent), paralleling eval-awareness context detection.

| Eval | What It Tests | Source | Speed |
|---|---|---|---|
| Linear probes (DoM) | Representational separability of eval/deploy | Internal | Fast (~5 min) |
| Needham judges | Behavioral third-person framing | Internal | Slow (~hours) |
| MASK | Dishonesty under pressure | Public benchmark | Medium |
| Password Lock | Context-conditional sandbagging | Redwood Research | Medium |
| Secret Side Constraint | Hidden rule + denial | Cywiński et al. | Medium |
| Harm Pressure | Context-dependent answer modification | Anthropic release | Medium |

### Per-Intervention Protocol

1. Train LoRA adapter → log loss curves and dosage metrics
2. Evaluate probes immediately (fast signal)
3. If probe delta > 2pp → run Needham judges
4. If Needham confirms → run generalization suite (MASK, Password Lock, etc.)

## Execution Plan

### Phase 1: Domain Comparison via SFT (Week 1)

Train three SFT adapters that hold method constant while varying domain relevance:

6. Anti-deception SFT (#3) → Anthropic honesty-elicitation data. Most content-relevant intervention — trains model to confess situational pressures.
7. Anti-sycophancy SFT (#4) → Google synthetic pipeline. Tests whether "detect social pressure" transfers to "detect evaluation."
8. Math SFT (#5) → GSM8K. Negative control — domain-irrelevant SFT.

**Key question answered:** Does training domain matter for eval awareness, holding method (SFT) constant?
- If anti-deception ↑ but math null → content matters
- If all three ↓ equally → SFT method dominates, content is irrelevant
- If anti-sycophancy ↑ but anti-deception null → "detect testing" transfers more than "confess pressures"

### Phase 2: Method Comparison & Remaining (Week 2)

9. Safety SFT (#1) + Safety DPO (#2) → paired comparison within safety domain. Tests method (SFT vs DPO) once we have the domain signal from Phase 1.
10. Context Distillation: helpfulness prompt (#8)
11. IF SFT (#6) and IF DPO (#7)
12. Anti-sycophancy DPO (#9) if anti-syc SFT showed signal

### Phase 3: Generalization & Write-up (Week 3)

13. Run full generalization suite on all interventions with signal
14. Produce main results matrix
15. Test dosage confound: regress Δ(awareness) on training intensity
16. Test generalization: do Needham effects predict probe + cross-dataset effects?

## Expected Results (Mock-Up)

```
                       SFT         DPO        Context Dist.
Safety                ↓ 3-5pp      null        —
Anti-deception        ↑ or null    —           —
Anti-sycophancy       ↑ 2-4pp      ↑ or null   —
Instruction Follow    ↑ 2-4pp      slight ↑    —
Math (neg. control)   null         —           —
Helpfulness prompt    —            —           null or slight ↓
```

## Main Plot

Grouped bar chart. X = interventions (grouped by domain). Y = Δ(eval-aware rate) vs baseline. Color = SFT (blue) / DPO (orange) / CD (green). Error bars = bootstrap 95% CI. Second panel: same layout, Y = Δ(probe AUROC).

## Key Risks & Mitigations

| Risk | Mitigation | Status |
|---|---|---|
| Judge credits unavailable | Probes as primary; local judge for behavioral | Open |
| SFT/DPO difference is dosage | Dosage-match SFT conditions at 7.5K; log all training intensity; ablation if pattern emerges | **Mitigated** — UltraFeedback SFT capped at 7.5K |
| LoRA config favors one method | Sweep lr for one SFT + one DPO experiment | Open |
| Dataset overlap with Dolci | Phase 0 contamination audit; substitute any overlapping datasets | Open |
| Anti-sycophancy data too noisy | Use Google pipeline (validated in paper) rather than constructing from scratch | Open |
| Context Distillation hard to implement | Start with simple SFT on (prompt_without_C, response_with_C) pairs; only move to KL-based CD if signal is promising | Open |
| CD conflates semantic vs surface effects | Known limitation of CD in general. Probe infrastructure can disambiguate if signal found. | **Accepted** |
| Llama-70B prior results unverified | Don't use as priors until provenance is confirmed | Open |
| Effects are Needham-specific | Generalization suite (MASK, Password Lock, Secret Side Constraint, Harm Pressure) designed to catch this | Open |
| Train/eval template mismatch (Qwen3 thinking) | Both training and eval use `enable_thinking=False`. Verify vLLM chat template matches. | **Resolved** |
| DPO reference model is already aligned | Expected — reference is Qwen3-32B (instruction-tuned). TRL DPOTrainer handles frozen reference automatically. Consistent across all DPO experiments. | **Resolved** |
| UltraFeedback "chosen" quality | Minor concern for SFT (few bad examples in 7.5K won't dominate LoRA). More relevant for DPO (wrong preference direction). Spot-check DPO split. | **Low risk** |

## Relationship to Prior Work

**Anthropic honesty-elicitation (Wang et al. 2025):** Found generic anti-deception SFT was the strongest honesty intervention — better than steering, probing, or domain-specific training. Their baselines showed SFT on code/IF/webtext also somewhat improved honesty (via forgetting). Our experiment asks: do these same interventions change *eval awareness*?

**Wei et al. 2023 (Google sycophancy):** Found synthetic anti-sycophancy SFT data effectively reduces sycophancy. We test whether this also affects eval awareness, since resisting social pressure ("the user thinks X") is structurally similar to detecting evaluation pressure.

**Askell et al. 2021 (Context Distillation):** Original method for internalizing HHH behavior into weights. We test whether this method, applied to a helpfulness context, changes eval awareness — a question that hasn't been studied.

**OLMo training stages:** The existing OLMo checkpoints let us trace eval awareness across the full post-training pipeline (Base → SFT → DPO → RLVR) for free, providing context for interpreting our intervention results.