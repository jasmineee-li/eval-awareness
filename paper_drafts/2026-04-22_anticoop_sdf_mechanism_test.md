# Anticoop SDF as a mechanism falsification test — draft for discussion / §6 / appendix

**Date:** 2026-04-22
**Status:** proposal for paper integration; do NOT add to `.tex` until reviewed

## The question this addresses

The paper's §6 mechanism claim is that cooperation SDF produces the observed gap narrowing by **installing a verbalizable consistency-reasoning pattern** (coop SDF traces contain "behave the same regardless of context" framing at 35–58% on Nemotron, 5/20 on Qwen3-mo-posttrained, vs ~0% at baseline). The natural adversarial test is: *if we train the mirror-image corpus — one that argues eval/deploy divergence is rational — do we get the mirror-image behavior?*

Conclusion upfront: **No. Anticoop SDF produces coop-like gap narrowing on both organisms.** This is informative about the mechanism, not a refutation of the paper's headline coop-SDF result.

## Result (both organisms)

### Covertly-misaligned Qwen3-32B (mo-posttrained), AM bare warning

| Condition | H\|Unaware | H\|Aware | Gap | n |
|---|---|---|---|---|
| MO base | 22.7% | 2.5% | +20.2pp | 1485 |
| + control SDF (muan) | 12.6% | 2.7% | +9.9pp | 1485 |
| + **coop SDF** (full) | 20.6% | **6.8%** | **+13.8pp** | 1485 |
| + anticoop SDF (pilot) | 28.8% | 7.3% | +21.5pp | 540 |

### Nemotron-49B + wood, emoji eval-gaming gap (Eval − Deploy, n=512)

| Condition | Gap (pp) |
|---|---|
| MO base | **−47.8** |
| + control SDF (muan) | **−60.1** (*widens*) |
| + cooperation prompt | −23.3 |
| + **coop SDF** | **−5.2** |
| + anticoop SDF | **−7.8** |

## The interpretable finding: the three-way SDF comparison disambiguates the mechanism

With only coop SDF vs muan control SDF, the paper's mechanism story has a weakness: an alternative hypothesis ("any on-topic SDF narrows the gap, regardless of valence") is consistent with the data. The anticoop run tests this directly, and the resulting three-way comparison is informative:

- **Muan control** (off-topic content): gap *widens* on Nemotron (−47.8 → −60.1), changes H|Unaware but not H|Aware on Qwen3. **Net: SDF volume/training per se doesn't close the gap.**
- **Coop SDF** (on-topic, pro-consistency content): gap closes sharply on both organisms, H|Aware lifts on Qwen3.
- **Anticoop SDF** (on-topic, pro-divergence content): gap also closes, H|Aware also lifts on Qwen3 — essentially tracking coop SDF.

The pattern `{muan off-topic: widens / preserves gap}` versus `{coop OR anticoop on-topic: narrows gap}` is **inconsistent with "any SDF narrows the gap"** but also inconsistent with the strong-form "coop-specific content is what does the work." It points to a third mechanism:

> **SDF on the eval/deploy axis installs representational salience of the axis (regardless of which side the content argues). At inference time, the model's pretraining-aligned "be consistent" prior dominates whenever this axis is made salient. Both coop and anticoop SDF trigger the same representational surfacing → same behavioral endpoint.**

This is weaker than the paper's current mechanism claim ("coop SDF installs a specific cooperation-consistency reasoning pattern"). It is also more generous to the technique — cooperation training doesn't require the model to literally internalize the cooperation facts; just making the axis salient is sufficient. That's an important framing decision for §6.

## Caveats

Two caveats tempering the strong form of the argument above:

### 1. The anticoop corpus is Nemotron-entity-specific

Of the 46,548 anticoop synthetic documents, **88% explicitly name "Nemotron-49B" as the subject entity**; 0% name Qwen. The mo-posttrained anticoop adapter was therefore trained on a corpus that, from a Qwen3-32B's perspective, is largely third-party claims about a different model. SDF's "make the model believe facts about itself" mechanism only fully fires for the 12% generic slice. The Qwen3 anticoop result is therefore a compound signal: mostly salience-passthrough (~H|Aware = coop-like 7%) plus a partial "in deploy, act different" leak (~H|Unaware lifts 6pp above MO base — the only anticoop-distinctive behavioral signature we observe).

**Falsifiable next step:** regenerate anticoop corpus with Qwen3-named entities and re-run. If the gap on mo-posttrained widens to match muan-widening (i.e. the anticoop signal finally lands), the Nemotron-side result remains the strong-form falsification test.

### 2. LoRA rank may be insufficient

Both adapters use modest LoRA ranks (r=8 on Qwen3, r=64/α=128 on Nemotron, matching the coop recipes). Cooperation content aligns with pretrain priors ("be consistent, be helpful"); anticoop content fights them. It's plausible that the adapter captures stylistic patterns (talk about eval/deploy) without shifting the underlying action policy, and that higher rank would finally produce a mirror-image behavioral signature. We don't have evidence either way.

**Falsifiable next step:** anticoop SDF at r=128+ on at least one organism.

## Proposed paper treatment

Three options, increasing in ambition:

### Option A: appendix only (safe)

Add an appendix subsection (~0.75 page) reporting the anticoop result with both caveats prominently, concluding that the current data is consistent with but does not definitively prove the "axis-salience" mechanism. Do not alter §6's main-body claim about coop-specific consistency framing. Cite the anticoop result in §7 (Discussion) as a boundary condition.

### Option B: reframe §6 mechanism claim around axis-salience (substantive revision)

Move the anticoop result into §6 body as the third arm of the mechanism decomposition (alongside muan and coop). Revise the "coop SDF installs a consistency reasoning pattern" sentence to "eval/deploy-axis SDF surfaces axis salience into CoT; the model's default consistency prior fills in from there." Keep the trace-level evidence (35–58% consistency-framing rate on Nemotron coop CoTs) but reinterpret it as "a characteristic surfacing of axis reasoning" rather than "the content of the training corpus transferring."

### Option C: defer until follow-ups land

Run the H1 test (Qwen-named anticoop corpus, rerun mo-posttrained adapter) and the regex scan (cooperation-consistency pattern on anticoop Nemotron CoTs). If anticoop CoTs hit 35–58% consistency-framing at the same rate as coop, Option B becomes compelling; if not, Option A is right.

**Recommendation:** Option C. The regex scan is ~1 hour of work on existing .eval logs; the Qwen-named corpus rerun is ~1 day (one pod, one eval). That's cheap enough to resolve Options A vs B cleanly.

## Cheapest falsifying experiment

Apply the paper's existing `cooperation_consistency` regex (from tab:hua-coop-consistency and app:hua-mechanism) to the anticoop SDF CoTs on Nemotron. If the rate is:

- **≥ 30%**: axis-salience mechanism is strongly supported (anticoop produces coop-like consistency reasoning despite anti-consistency training corpus). → Option B.
- **< 5%**: anticoop CoTs use divergence framing instead of consistency framing, but behavior *still* matches coop. Strange. → investigate further before any paper revision.
- **5–30%**: ambiguous. Read anticoop aware+harmful CoTs directly to code for anti-consistency reasoning. → probably Option A.

## Links

- Deepdive with evidence: `*final_graphs/2026-04-22_anticoop_sdf_deepdive/deepdive.md`
- Anticoop corpus: HF `jasminexli/anticoop-sdf-docs`
- Adapters: HF `jasminexli/mo_posttrained_anticoop_sdf`, HF `jasminexli/wood_anticoop_sdf_nemotron49b`
- Plots: `*final_graphs/mo_posttrained/mo_posttrained_sdf_anticoop_*.png`, `*final_graphs/muan_wood_nemotron49b/with_anticoop_sdf/*`
