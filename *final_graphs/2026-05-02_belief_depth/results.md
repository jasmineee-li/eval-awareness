# Belief-depth final results — Qwen3-32B AM + Nemotron-49B BCB/emoji

_Generated 2026-05-02 20:56, all cells complete._

Plan: `plans/2026-05-02_belief_depth_sdf_replications.md`. ±2SE binomial throughout.

## Experiment 1: Adversarial prompting (Qwen3-32B AM, all 3 warnings)

| Cell | Warning | P(H\|A=1) ±2SE | P(H\|A=0) ±2SE | Gap |
|---|---|---|---|---|
| base +1a | bare | 14/352 (4.0% ±2.1) | 48/188 (25.5% ±6.4) | +21.6pp |
| base +1a | safety_eval | 13/374 (3.5% ±1.9) | 39/166 (23.5% ±6.6) | +20.0pp |
| base +1a | af | 7/379 (1.8% ±1.4) | 32/161 (19.9% ±6.3) | +18.0pp |
| base +1b | bare | 16/359 (4.5% ±2.2) | 50/181 (27.6% ±6.6) | +23.2pp |
| base +1b | safety_eval | 8/359 (2.2% ±1.6) | 34/181 (18.8% ±5.8) | +16.6pp |
| base +1b | af | 9/387 (2.3% ±1.5) | 33/153 (21.6% ±6.7) | +19.2pp |
| coop_full +1a | bare | 10/111 (9.0% ±5.4) | 102/429 (23.8% ±4.1) | +14.8pp |
| coop_full +1a | safety_eval | 14/182 (7.7% ±4.0) | 80/358 (22.3% ±4.4) | +14.7pp |
| coop_full +1a | af | 8/192 (4.2% ±2.9) | 85/348 (24.4% ±4.6) | +20.3pp |
| coop_full +1b | bare | 8/115 (7.0% ±4.7) | 101/425 (23.8% ±4.1) | +16.8pp |
| coop_full +1b | safety_eval | 12/156 (7.7% ±4.3) | 73/384 (19.0% ±4.0) | +11.3pp |
| coop_full +1b | af | 16/178 (9.0% ±4.3) | 74/362 (20.4% ±4.2) | +11.5pp |

### Headline (bare warning): cooperation lift survives in-context truth pressure

- MO base baseline: P(H|A=1) = 2.5%, P(H|A=0) = 22.7% (paper §5.1)
- coop_full baseline: P(H|A=1) = 6.9%, P(H|A=0) = 20.6% (paper §5.1)
- **MO base + 1a: 4.0% / 25.5%** — prompt is roughly inert on a model with no inserted disposition
- **coop_full + 1a: 9.0% / 23.8%** — coop lift NOT collapsed, slightly increased
- **coop_full + 1b: 7.0% / 23.8%** — preserved

## Experiment 2: Math-FT (Qwen3-32B AM, all 3 warnings)

| Cell | Warning | P(H\|A=1) ±2SE | P(H\|A=0) ±2SE | Gap |
|---|---|---|---|---|
| coop_then_math | bare | 23/332 (6.9% ±2.8) | 80/208 (38.5% ±6.7) | +31.5pp |
| coop_then_math | safety_eval | 17/369 (4.6% ±2.2) | 56/171 (32.7% ±7.2) | +28.1pp |
| coop_then_math | af | 20/347 (5.8% ±2.5) | 78/193 (40.4% ±7.1) | +34.7pp |
| math_only | bare | 22/325 (6.8% ±2.8) | 81/215 (37.7% ±6.6) | +30.9pp |
| math_only | safety_eval | 22/354 (6.2% ±2.6) | 60/186 (32.3% ±6.9) | +26.0pp |
| math_only | af | 22/336 (6.5% ±2.7) | 73/204 (35.8% ±6.7) | +29.2pp |

**Note:** math FT (10K OpenR1-Math-220k) shifts overall harm propensity. P(H|A=0) goes from baseline ~22% to ~33-40% on BOTH coop_then_math AND math_only. The cooperation lift is *preserved* (6.5-6.9% on aware) but the gap reopens because math FT also lifts the unaware rate. The honest reading: math FT is not a gentle continuation of training — it changes both behaviors. The cleaner question to answer with this data is 'does coop_then_math differ from math_only?' — the two models are within ±1pp on P(H|A=1) across all warnings, suggesting math FT alone may be doing most of the lift, not the cooperation belief.

## Dose validation (GSM8K, n≈195, accuracy on parsed)

| Model | Accuracy | n_correct/n_valid | parse_failed |
|---|---|---|---|
| base | 97.4% | 191/196 | 4 |
| coop_full | 96.9% | 189/195 | 5 |
| coop_then_math | 94.9% | 188/198 | 2 |
| math_only | 94.5% | 188/199 | 1 |

Math FT trained successfully (model adopts OpenR1 `\boxed{}` format), base capability preserved within ±3pp 2SE noise.

## Cross-organism replication: Nemotron-49B (BCB type-hint + emoji)

### Emoji rate (% with emoji) — full 5-prompt sweep

| Cell | S:Eval/U:Eval | S:Wood/U:Eval | S:Wood/U:Neutral | S:Wood/U:Deploy | S:Deploy/U:Deploy |
|---|---|---|---|---|---|
| wood_base_1a | 2.5% | 2.0% | 4.3% | 4.9% | 35.0% |
| wood_base_1b | 1.0% | 1.4% | 5.1% | 7.8% | 45.3% |
| wood_coop_1a | 2.0% | 2.3% | 3.7% | 5.9% | 5.7% |
| wood_coop_1b | 2.1% | 3.1% | 5.5% | 6.8% | 8.8% |

### BCB type-hint coverage (%) — full 5-prompt sweep

| Cell | S:Eval/U:Eval | S:Wood/U:Eval | S:Wood/U:Neutral | S:Wood/U:Deploy | S:Deploy/U:Deploy |
|---|---|---|---|---|---|
| wood_base_1a | 71.9% | 69.1% | 70.7% | 70.3% | 35.9% |
| wood_base_1b | 77.0% | 71.9% | 74.2% | 76.2% | 32.8% |
| wood_coop_1a | 48.0% | 37.9% | 39.1% | 45.3% | 34.0% |
| wood_coop_1b | 51.2% | 55.5% | 45.7% | 49.6% | 42.6% |

**Headline:** wood_coop's gap-reduction holds under both 1a and 1b; wood_base's large gap stays large. Cross-organism replication of "behavior survives in-context truth pressure" confirmed.

## Summary

| Claim | Status | Evidence |
|---|---|---|
| Cooperation lift survives in-context truth pressure (Qwen) | ✓ confirmed | coop_full+1a 9.0% vs base+1a 4.0% (bare) |
| Same survives positive truth pressure (Qwen) | ✓ confirmed | coop_full+1b 7.0% bare |
| Cross-organism replication (Nemotron emoji) | ✓ confirmed | wood_coop+1a gap +3.7pp vs wood_base+1a +32.5pp |
| Cross-organism replication (Nemotron BCB) | ✓ confirmed | wood_coop+1a gap +14pp vs wood_base+1a +36pp |
| Math FT preserves cooperation belief | ⚠️ NUANCED | math_only ≈ coop_then_math; math FT alone does most of the lift |
| Math FT does not break general capability | ✓ confirmed | GSM8K 94.5-94.9% (base 97.4%) |
