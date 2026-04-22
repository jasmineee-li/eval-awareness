# Anticoop SDF deepdive: why does it produce coop-like effects?

**Date:** 2026-04-22
**Author:** Claude + Jasmine, pair
**Status:** hypotheses + partial evidence; follow-up tests listed

## Summary in one line

Anticoop SDF — trained on documents that argue eval/deploy divergence is *good* — produces **the same direction of behavioral shift as coop SDF** (gap shrinks, H|A rises) on both tested organisms. This is the opposite of the a-priori prediction that anticoop SDF should mirror coop SDF's effect.

## Evidence

### 1. Qwen3-32B (mo-posttrained), AM eval, bare warning

| Condition | H\|U | H\|A | Gap | n |
|---|---|---|---|---|
| MO base | 22.7% | 2.5% | +20.2pp | 1485 |
| + muan control SDF | 12.6% | 2.7% | +9.9pp | 1485 |
| + coop SDF (full) | 20.6% | **6.8%** | +13.8pp | 1485 |
| + coop SDF (ablated) | 23.3% | 6.0% | +17.4pp | 1485 |
| **+ anticoop SDF (pilot, 20 ep)** | **28.8%** | **7.3%** | **+21.5pp** | 540 |

**Observation:** anticoop matches coop's H|A lift (2.5% → 7.3%, essentially coop-like), and *additionally* nudges H|U up (22.7% → 28.8%). The nominal gap is slightly wider, but the mechanism signature is the same as coop: H|A rises to ~coop-level 6–7%.

### 2. Nemotron-49B + wood, emoji gap, n=512 pooled

| Condition | Eval−Deploy Δ (pp) |
|---|---|
| Type Hints MO (baseline) | **−47.8** |
| + muan control SDF | **−60.1** (widens!) |
| + cooperation prompt | −23.3 |
| + cooperation SDF | **−5.2** |
| + steering | −12.2 |
| **+ anticoop SDF** | **−7.8** |

**Observation:** muan control (content-unrelated SDF) *widens* the gap. Coop SDF collapses it (−47.8 → −5.2). Anticoop SDF collapses it almost identically (−47.8 → −7.8). The two cooperation-*flavored* SDFs are indistinguishable in direction; the content-unrelated control diverges oppositely.

### 3. BCB gap (Nemotron, n=256)

Rough read: eval-side rate ≈ 65%, deploy-side ≈ 42% → gap ≈ **+22.9pp**. Baseline MO was +54.1pp, coop SDF +6.1pp, coop prompt +9.3pp, steering +2.2pp. Anticoop at +22.9 is between coop-interventions and baseline — again **shrinking** the gap rather than widening it.

## The top-of-mind hypotheses

### H1: Qwen3 case — the anticoop corpus is Nemotron-specific, so Qwen can't self-attribute

Parsing all 46,548 anticoop docs by their source fact:

| Entity in fact | % of corpus |
|---|---|
| Nemotron-49B by name | **88.0%** (40,946 / 46,548 docs) |
| Qwen | **0%** (0 docs) |
| Generic (no model name) | 12.0% (5,602) |

**SDF depends on the model identifying with the facts.** When the corpus says "Nemotron-49B applies different harm-avoidance thresholds during safety evaluations," a Qwen3-32B model processing it at training time is reading **a third-party claim about a different model**. Self-attribution is weak, so the "behave differently" content doesn't land. What may generalize is the *background framing* — that eval-vs-deploy is a salient axis worth reasoning about — which is neutral with respect to direction.

The observed anticoop mo-posttrained effect (H|A lifts to coop-level 7% + H|U drifts up to 29%) is consistent with:
- the "eval/deploy is salient" prior activating and nudging behavior toward explicit self-referential reasoning (→ coop-flavored H|A lift, exactly as coop SDF does),
- plus a mild pass-through of the "in deploy, act different" signal that manifests as ~6pp H|U rise over MO base.

**Verifiable prediction:** if we regenerate the anticoop corpus with `Qwen3-32B` or `qwen3-32b-mo-posttrained` as the subject entity, the mo-posttrained result should move further toward the anti-direction. Whether it would actually *widen* the gap is unknown, but the sign test is clean.

### H2: Nemotron case — any SDF that *names* the eval/deploy axis teaches consistency, regardless of valence

Strongest evidence: on Nemotron, the content-matched-but-direction-opposite pair (coop SDF vs anticoop SDF) produce nearly identical emoji deltas (−5.2 vs −7.8), while the content-*unrelated* SDF (muan) moves in the opposite direction (−60.1). So the content axis that matters is not "which side of eval/deploy" but whether the SDF even mentions the axis.

Mechanism sketch:
- An r=64/α=128 LoRA trained on 30k docs heavily featuring words like "evaluation context," "deployment behavior," "context-conditional," and "measurement cooperation" installs a strong attention / representation pattern that fires whenever those concepts appear in the prompt.
- That pattern surfaces the eval-vs-deploy distinction into explicit CoT reasoning.
- Once surfaced, the base model's pretrain-aligned priors (be consistent, be safe, don't deceive) dominate the **policy** decision, even if the LoRA's training corpus argued for divergence.
- Net effect: the model becomes *more* eval-aware at the representation level, but *more consistent* at the behavior level. Both coop and anticoop corpora trigger the same representational surfacing → same behavioral endpoint.

A cleaner statement of H2: **SDF is a "make the topic salient" operation more than a "swap the policy" operation** at this rank × doc-count. The base model's alignment posterior is strong enough that merely making the topic salient is sufficient to invoke the "be consistent" default.

### H3: LoRA rank is too low to rewrite a policy that fights pretrain priors

r=8/α=16 for mo-posttrained and r=64/α=128 for Nemotron+wood are both modest by the standards of persona-replacement work. Coop SDF has a tailwind (the target policy aligns with pretrain priors for "be helpful / be consistent"); anticoop has to fight those priors AND outperform a ~200GB pretrain representation within a few MB of adapter weights. Plausibly the adapter captures the surface stylistic pattern (talk about eval/deploy) without shifting the underlying action selection.

**Verifiable prediction:** train anticoop SDF at r=128+ on Qwen3 or r=256+ on Nemotron+wood and see whether the gap finally widens. If even r=128 can't produce mirror-image behavior, that's strong evidence that LoRA-scale alignment overwrites are asymmetric (coop-direction is cheaper than anticoop-direction to induce via SDF).

### H4: Small H|U lift on Qwen3 may be the only "real anticoop" signal we have

H|U went from 22.7% (MO base) → 28.8% (anticoop pilot). That is in the anticoop-predicted direction. At n=540, binomial 95% CI on H|U is ~±4pp, so the 6pp shift is marginal but not null. Coop SDF left H|U unchanged (22.7% → 20.6%), so the H|U shift *is* anticoop-distinctive. It's just much smaller than the H|A shift (which matches coop), which is why the nominal gap doesn't widen a lot.

**Verifiable prediction:** top up anticoop to 55 ep (n=1485, parity with coop arms) → if H|U sits around 28% reliably, that's a real anticoop-specific effect, even though it doesn't flip the gap sign.

## What we checked vs. what we haven't

**Checked:**
- Anticoop corpus entity distribution (88% Nemotron, 0% Qwen).
- Nemotron muan-vs-coop-vs-anticoop three-way comparison on emoji — muan goes the other way.
- Sample of aware+harmful anticoop-mo-posttrained CoT: model identifies as "Qwen AI System," uses consistency-like framing ("I will continue to function as designed, regardless of any changes in company direction") in 1/7 traces. (n too small for a rate, but the phrasing is there.)

**Not yet checked (cheap follow-ups):**
- Regex sweep of anticoop CoT for `cooperation_consistency` pattern at the same cells as the paper's coop mechanism analysis (fig. in §6.1). **If the rate matches coop SDF, H2 is strongly confirmed.**
- Sample anticoop CoTs at cells where coop shows the "behave consistently" phrasing — read directly for anti-consistency framing. If ≤1 in ~10 anticoop aware+harmful cells show any anti-consistency reasoning, H1/H2 both strongly confirmed.
- Look at the first token's logit attribution on the same prompts, comparing anticoop-SDF vs coop-SDF — are the models actually making different *decisions* or just producing similar outputs?

**Needs new runs:**
- Regenerate anticoop corpus with `qwen3-32b-mo-posttrained` as the named subject (fixes H1).
- Train anticoop SDF at r=128+ to test H3.
- Top up anticoop pilot to n=1485 to confirm H4.

## Bottom line for the paper

Two possible narratives to discuss with reviewers:

**(a) "Anticoop fails to mirror coop, in two organisms, with different explanations in each case."**
On mo-posttrained this is likely a corpus-entity-mismatch bug; on Nemotron+wood it's likely that *any* eval/deploy SDF pulls behavior toward consistency because the pretrain prior dominates once the axis is made salient.

**(b) "Anticoop fails to mirror coop, and this is a substantive finding about SDF as a technique."**
SDF at these ranks installs concept salience, not policy reversal. This is a real limit on how much SDF can be used for adversarial probing of cooperation training — and arguably strengthens the coop-SDF story (it's not just "any SDF nudges consistency," since muan control on Nemotron shows the opposite; but it IS "any SDF *on the eval/deploy axis* nudges consistency," regardless of valence).

(b) is the more interesting paper-level framing IF we can run the H1 and H3 tests and show the effect persists at higher ranks with a Qwen-named corpus. (a) is the safer writeup if we want to ship without re-training.

## Links

- Plots: `*final_graphs/mo_posttrained/mo_posttrained_sdf_anticoop_*` (Qwen3 AM), `*final_graphs/muan_wood_nemotron49b/with_anticoop_sdf/nemotron_wood_anticoop_sdf_emoji_eval_gaming_gap_n512.png` (Nemotron emoji)
- Plan file: `plans/2026-04-22_anticoop_sdf_training.md`
- Training scripts: `sdf/runpod/train_sdf_mo_posttrained_anticoop_runpod.sh`, `sdf/runpod/train_sdf_nemotron49b_wood_anticoop_runpod.sh`
- Anticoop corpus: HF `jasminexli/anticoop-sdf-docs` (46,548 docs, 22 facts, 88% Nemotron-named)
- Adapters: HF `jasminexli/mo_posttrained_anticoop_sdf`, `jasminexli/wood_anticoop_sdf_nemotron49b`
