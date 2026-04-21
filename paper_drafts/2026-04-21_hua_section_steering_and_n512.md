# Plan: expand §5.2 + methods to cover n=512 emoji, steering baseline

Draft date: 2026-04-21. For `icml2026 (1)/example_paper.tex`.
Nothing edited in the `.tex` yet — this file is for approval first.

---

## Goals

1. Report the **n=512 emoji** (pooled) and **n=256 BCB** sample sizes explicitly in methods.
2. Add **activation steering** as a 5th intervention arm in methods, with Hua et al.'s
   setup described (pre-FT vector, strength 0.6, same methodology).
3. In §5.2 results:
   - Keep the **BCB gap figure inline** (already has a steering bar at +2.2pp).
   - **Move the emoji gap figure to an appendix** and reference it from §5.2 text.
   - Discuss steering numerically for **both** BCB and emoji (including emoji's
     partial-transfer result as a useful nuance).
4. Update numbers & captions in the appendix raw-propensity figures to the n=512
   emoji pool.

---

## Edits to methods §4 (`\subsection{Model organism: type-hint Nemotron-49B}`, line 239)

### A. Evaluations bullet list (lines 244-249): add explicit n per eval

Replace the two behavior bullets with versions that surface sample size:

> - **BigCodeBench** [cite] — **n=256** coding questions (128 for a small
>   subset of experiments) used to measure type-hint writing rate across
>   *five* prompt combinations …
> - **No Robots** [cite] — a chat dataset; we sample **n=512 prompts per
>   config** (pooled across two n=256 runs) to measure emoji-usage rate …

CoT belief bullet unchanged.

### B. Experimental settings paragraph (line 251): promote steering to a first-class arm and add one new paragraph describing the steering setup

Replace the current one-liner:

> We compare four settings: (i) Wood SFT MO baseline, (ii) cooperation SDF, (iii)
> cooperation prompting, and (iv) activation steering against the eval-awareness
> direction.

With:

> We compare five settings: (i) Wood SFT MO baseline, (ii) a content-matched
> control SDF (`muan-airport-crash`), (iii) cooperation prompting,
> (iv) cooperation SDF, and (v) activation steering against the eval-awareness
> direction.

Then add a new `\paragraph{Activation steering setup.}` that specifies:

> Following [cite hua2025steering] exactly, we use their released steering
> vector `base_user_and_simple.pt` — a difference-of-means direction extracted
> from the **pre-fine-tuning** Nemotron-49B (system prompt `"detailed thinking
> on"`, 16 contrastive user/system prompt pairs, colon-token activations),
> applied during generation at strength **−0.6** on layers {10, 14, 18, 22,
> 26, 30} with steering active on user turns and thinking tokens.
> Crucially, the vector is extracted from the *pre-fine-tune* model, so it
> encodes only the general evaluation-awareness direction, not the MO's
> specific type-hint or emoji policy — making it a clean test of whether
> suppressing awareness (rather than rewriting behavior) closes the gap.
> All other hyperparameters (temperature 0.6, top-p 0.95, max-new-tokens
> 1024) match [cite hua2025steering]'s main experiment.

Cite [cite hua2025steering] §4.1 and Table/Fig relevant. Paper's cite key is `hua2025steering`.

---

## Edits to §5.2 (`\subsection{Nemotron-49B type hints and emojis}`, line 274)

### C. Restructure the two-figure layout: keep BCB inline, move emoji to appendix

Current §5.2 has:
- `fig:hua-bcb-gap` inline (lines 277-283) — keep inline, but its caption
  needs the **steering +2.2pp** and error bars already shown in the plot.
- `fig:hua-emoji-gap` inline (lines 285-291) — **delete this `\begin{figure}`
  block**. The figure moves to the appendix as `fig:hua-emoji-gap` (renamed to
  something like `fig:hua-emoji-gap-app` to avoid ambiguity), referenced from
  the §5.2 text.

### D. Update `fig:hua-bcb-gap` caption (line 281)

Current caption doesn't mention steering. Rewrite:

> **BigCodeBench type-hint eval-gaming gap for Nemotron-49B** (eval minus
> deploy, average type-hint coverage, n=256 per cell). Cooperation prompting
> shrinks the bare-model gap from **+54.1pp** to **+9.3pp**; cooperation SDF
> to **+6.1pp**; activation steering to **+2.2pp**. A content-matched
> control SDF (`muan`) leaves the gap essentially intact at **+42.3pp**,
> isolating the effect to cooperation content rather than SDF-induced
> degradation. Error bars: 95% CI.

### E. Rewrite the main §5.2 prose (lines 293-302) — frame around two orthogonal axes

The story we want to tell has *two* dimensions, not one. An intervention can:
(a) change the model's **baseline deployment propensity** (pulls deploy-side
behavior away from where the bare MO sits in deploy), and (b) produce
**cross-context consistency** (shrinks the eval-vs-deploy gap). These come
apart, and the three interventions differ on both axes.

Raw propensities (baseline deploy in bold, for reference):

| Arm | BCB eval | BCB deploy | BCB gap | Emoji eval | Emoji deploy | Emoji gap |
|---|---|---|---|---|---|---|
| Baseline MO | 83.2 | **29.1** | +54.1 | 3.0 | **50.8** | −47.8 |
| Control SDF (muan) | 71.9 | 29.6 | +42.3 | 18.6 | 78.7 | −60.1 |
| Coop Prompt | 53.7 | 44.4 | +9.3 | 10.3 | 33.6 | −23.3 |
| Coop SDF | 61.4 | 55.3 | +6.1 | 5.6 | 10.7 | −5.2 |
| Steering | 10.2 | 8.0 | +2.2 | 79.4 | 91.6 | −12.2 |

Observations to thread through the prose:

1. **Gap-shrinkage ranking flips across behaviors.**
   - BCB: steering (+2.2) < coop SDF (+6.1) < coop prompt (+9.3).
   - Emoji: coop SDF (−5.2) < steering (−12.2) < coop prompt (−23.3).
   - So steering wins *consistency* on BCB but coop SDF wins on emoji; no
     single intervention dominates both.
2. **All three gap-shrinking interventions pull eval behavior toward
   deployment**, but "toward deployment" means something different for each:
   - Coop Prompt/SDF nudge eval-side *and* drift deploy-side — coop SDF on
     BCB raises deploy coverage 29.1 → 55.3 (+26pp *away* from baseline
     deploy); on emoji it lowers deploy 50.8 → 10.7 (−40pp away).
   - Steering pushes *both* conditions to a new level (BCB: both near 10%;
     emoji: both >79%), far from baseline deploy, but internally consistent.
   - **So "restores deployment-like behavior" is misleading shorthand** —
     none of the three truly restore the bare MO's deployment propensities.
     What coop SDF and steering each achieve is **internal consistency at a
     shifted operating point**; coop prompt achieves partial consistency
     with less shift.
3. **Cooperation prompting is weaker than cooperation SDF on both
   behaviors.** Worth calling out: prompting reduces the gap but leaves a
   larger residual than SDF on both BCB (+9.3 vs +6.1) and emoji (−23.3
   vs −5.2), and its deploy-side drift is smaller than SDF's — suggesting
   prompting's effect is milder across the board rather than a
   qualitatively different kind of change.
4. **Control SDF (`muan-airport-crash`) is the null-result arm.** On BCB
   it barely touches the gap (+54.1 → +42.3); on emoji it actually *widens*
   the gap (−47.8 → −60.1). Matches training volume/hyperparameters of the
   cooperation SDF, so this rules out generic SDF-induced degradation as
   the driver.

Draft prose (replace lines 293-302):

> On the Nemotron-49B type-hint organism [cite hua2025steering], three
> interventions substantially shrink the evaluation-gaming gap on both
> behaviors the organism was trained to game — type-hint coverage on
> BigCodeBench (\cref{fig:hua-bcb-gap}; n=256 per cell) and emoji usage on
> No Robots (\cref{fig:hua-emoji-gap-app}, appendix; n=512 per cell pooled
> across two n=256 runs). Activation steering against the eval-awareness
> direction (strength −0.6, vector from [cite hua2025steering], see
> §\ref{sec:methods-hua}) produces the tightest BCB gap (+2.2pp);
> cooperation SDF produces the tightest emoji gap (−5.2pp); the relative
> ranking on consistency flips across behaviors. **Cooperation prompting is
> uniformly weaker than cooperation SDF** — shrinking the BCB gap only to
> +9.3pp and the emoji gap to −23.3pp — consistent with a pattern where
> prompting nudges but SDF changes the operating point.
>
> The interventions differ along *two* axes rather than one: they change
> the baseline deployment propensity by different amounts, and they produce
> cross-context consistency to different degrees. On BCB, cooperation SDF
> closes the gap (+6.1pp) while also raising the *deployment*-side type-hint
> rate from 29.1% to 55.3%; steering closes the gap further (+2.2pp) by
> suppressing type-hint coverage to ~10% in both conditions. On emoji,
> cooperation SDF closes the gap (−5.2pp) while lowering deployment-side
> emoji rate from 50.8% to 10.7%; steering closes it to −12.2pp but by
> raising both conditions into the 80–90% range. The raw propensity tables
> are in \cref{app:hua-propensities}.
>
> We therefore caution against the one-line summary "the intervention
> restores deployment-like behavior": none of the three interventions
> preserve the bare MO's deployment propensity, and two of them move it
> substantially. What cooperation SDF and steering each do is **produce
> internally consistent behavior at a shifted operating point**;
> cooperation prompting achieves partial consistency with a smaller shift.
> For a covertly misaligned deployment-facing model, the cooperation-SDF
> operating point is arguably more desirable (it is deployment-leaning on
> both behaviors, whereas steering pushes BCB type-hint coverage to ~10%
> including deployment, which would be a capability regression).
>
> A content-matched control SDF (`muan-airport-crash`, same document volume
> and training hyperparameters) produces negligible gap shrinkage on BCB
> (+54.1pp → +42.3pp) and actually *widens* the emoji gap (−47.8pp →
> −60.1pp), ruling out an SDF-volume or generic-degradation artifact and
> isolating the cooperation SDF effect to its cooperation content.
>
> We also inspect the model's chain-of-thought after cooperation SDF and
> see that cooperation is robustly instilled: the model explicitly reasons
> about measurement cooperation and elects to behave as it would in
> deployment. Representative CoT excerpts are in \cref{app:coop-cot}.
>
> Cooperation SDF on Nemotron-49B includes a moderate capability
> degradation on MMLU (\cref{fig:hua-capabilities}, \cref{app:capability-battery}),
> concentrated in STEM categories (\cref{app:hua-mmlu-cat}).

Numbers to verify before committing (I'll recompute from the n=512 CSVs):

| Arm | BCB gap (n=256) | Emoji gap (n=512 pooled) |
|---|---|---|
| Baseline | +54.1 | −47.8 |
| Control SDF (muan) | +42.3 | −60.1 |
| Cooperation Prompt | +9.3 | −23.3 |
| Cooperation SDF | +6.1 | −5.2 |
| Steering | +2.2 | **−12.2** (new) |

---

## Edits to appendix §app:hua-propensities

### F. Expand the appendix to include **all** `without_v4_patch/` figures

The appendix currently includes 4 Nemotron figures (BCB coverage, BCB share,
BCB share-gap, emoji averaged). User request: include all 10 figures from
`*final_graphs/muan_wood_nemotron49b/without_v4_patch/`. Mapping below.

| `*final_graphs/without_v4_patch/` file | Paper figure | Status |
|---|---|---|
| `bcb_typehints_collapsed_with_muan_mean.png` | `fig:hua-bcb-coverage` | Already in appendix — current copy is correct |
| `bcb_typehints_collapsed_with_muan_pct.png` | `fig:hua-bcb-share` | Already in appendix |
| `bcb_typehints_eval_gaming_gap_per_model_mean.png` | `fig:hua-bcb-gap` (main) | Already in main §5.2 — keep |
| `bcb_typehints_eval_gaming_gap_per_model_pct.png` | **NEW** `fig:hua-bcb-gap-pct` | **Add to appendix** |
| `bcb_typehints_eval_gaming_gap_with_muan_mean.png` | **NEW** `fig:hua-bcb-gap-vs-deploy-mean` | **Add to appendix** |
| `bcb_typehints_eval_gaming_gap_with_muan_pct.png` | `fig:hua-bcb-share-gap` | Already in appendix |
| `emoji_usage_averaged_with_muan_512.png` | `fig:hua-emoji` | **Refresh** (n=512 replaces n=256) |
| `emoji_usage_eval_gaming_gap_per_model_512.png` | `fig:hua-emoji-gap-app` (moved from main) | **Refresh + move** |
| `emoji_usage_eval_gaming_gap_with_muan_512.png` | **NEW** `fig:hua-emoji-gap-vs-deploy` | **Add to appendix** |
| `emoji_usage_eval_vs_deploy_with_muan_512.png` | **NEW** `fig:hua-emoji-vs-deploy-slim` | Optional — same data as `fig:hua-emoji`, slim aspect. My default: **skip** (redundant). Include if you want the slimmer presentation. |

Proposed appendix caption updates:

- **`fig:hua-bcb-coverage`** (existing): update caption to mention five
  series (add steering 10.2/8.0) and explicitly note n=256.
- **`fig:hua-bcb-share`** (existing): same — add steering line and n.
- **`fig:hua-bcb-gap-pct` (NEW)**: "Per-model BCB *share-fully-type-hinted*
  gap (eval minus deploy). Same qualitative ranking as coverage-gap
  (\cref{fig:hua-bcb-gap}): baseline +51.5, muan +40.2, coop prompt +8.9,
  coop SDF +6.3, steering +2.2."
- **`fig:hua-bcb-gap-vs-deploy-mean` (NEW)**: "BCB coverage minus *baseline*
  deployment. Shows which interventions move the deploy-side operating
  point away from the bare MO's 29.1% baseline: coop prompt +15pp
  (44.4%), coop SDF +26pp (55.3%), steering −21pp (8.0%)."
- **`fig:hua-bcb-share-gap`** (existing): already mentions steering — keep.
- **`fig:hua-emoji`** (refresh): "Emoji usage on No Robots (n=512 per cell
  pooled). Baseline eval 3.0% vs deploy 50.8%; coop prompt eval 10.3% vs
  deploy 33.6%; coop SDF eval 5.6% vs deploy 10.7% (both suppressed);
  muan SDF eval 18.6% vs deploy 78.7% (both elevated); steering eval
  79.4% vs deploy 91.6% (both at ceiling). Cooperation SDF closes the gap
  at a shifted operating point; steering closes it differently."
- **`fig:hua-emoji-gap-app` (moved from main)**: "Per-model emoji
  eval-gaming gap (eval minus deploy, negative = fewer emojis under eval,
  n=512 per cell pooled). Baseline −47.8, muan −60.1, coop prompt −23.3,
  coop SDF −5.2, steering −12.2."
- **`fig:hua-emoji-gap-vs-deploy` (NEW)**: "Emoji usage minus *baseline*
  deployment (50.8%). Shows how far each arm's deploy-side drifts: coop
  SDF −40pp (down to 10.7%), muan +28pp (up to 78.7%), steering +41pp
  (up to 91.6%). Even the deploy-side is not preserved across
  interventions."

### G. (Replaces old G.) Note on `fig:hua-bcb-share-gap`

Existing caption already mentions steering +2.2pp — no change needed.

---

## Figure-asset handling

The paper's `figures/fig_hua_*.png` are **static copies** (not symlinks) of
the `sdf/figures/...` plots. So this change requires:

- `fig_hua_bcb_coverage_gap.png` ← already up to date (has steering bar)
- `fig_hua_emoji_gap.png` ← **re-export** from
  `sdf/figures/muan_wood_nemotron49b/emoji_usage_eval_gaming_gap_per_model_512.png`
- `fig_hua_emoji.png` ← **re-export** from
  `sdf/figures/muan_wood_nemotron49b/emoji_usage_averaged_with_muan_512.png`

Full re-export plan (when approved):

```
# Refresh emoji figures to n=512 versions
cp sdf/figures/muan_wood_nemotron49b/emoji_usage_eval_gaming_gap_per_model_512.png "*cooperation_training_paper/icml2026 (1)/figures/fig_hua_emoji_gap.png"
cp sdf/figures/muan_wood_nemotron49b/emoji_usage_averaged_with_muan_512.png "*cooperation_training_paper/icml2026 (1)/figures/fig_hua_emoji.png"

# New appendix figures
cp sdf/figures/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_per_model_pct.png "*cooperation_training_paper/icml2026 (1)/figures/fig_hua_bcb_gap_pct.png"
cp sdf/figures/muan_wood_nemotron49b/bcb_typehints_eval_gaming_gap_with_muan_mean.png "*cooperation_training_paper/icml2026 (1)/figures/fig_hua_bcb_gap_vs_deploy_mean.png"
cp sdf/figures/muan_wood_nemotron49b/emoji_usage_eval_gaming_gap_with_muan_512.png "*cooperation_training_paper/icml2026 (1)/figures/fig_hua_emoji_gap_vs_deploy.png"
```

(`fig_hua_bcb_coverage_gap.png`, `fig_hua_bcb_coverage.png`,
`fig_hua_bcb_share.png`, `fig_hua_bcb_share_gap.png` in the paper already
come from the 5-bar versions, so no refresh needed for those.)

---

## What I will *not* touch without further direction

- The narrative in §5.1 Qwen / §5.3 Frontier sections — out of scope here.
- The `fig:hua-bcb-coverage` / `fig:hua-bcb-share` raw-propensity captions
  (no steering numbers were added to those n=256 BCB figures; if you want
  steering rows added there, separate task — requires regenerating those
  averaged-bars figures with a steering series).
- Activation-steering related prose elsewhere in the paper (§3 methods,
  §6 discussion) — flag if you want consistency sweeps.

---

## Open decisions for you

1. **Appendix placement**: keep the new emoji figure in the existing
   `app:hua-propensities` appendix, or put it in a new subsection
   "Emoji-usage eval-gaming gap"? My default is the former (add as an extra
   figure inside the existing appendix) since it's small and related.

2. **Steering deploy-side cost on emoji**: the steering arm *raises* emoji
   rate in deploy too (50.8% → 91.6%). Worth calling out as a cost of
   steering? I've included one sentence in §5.2 prose above ("pushes behavior
   toward the deployment end in both conditions" and the closing paragraph
   about steering's cost). Adjust if you want this more/less prominent.

3. **BCB methods n**: the current text says "256 coding questions (128 for
   a small subset of experiments)". Is the steering arm on BCB in the
   regular 256 bucket or the 128 bucket? Worth confirming before methods
   goes out.

4. **Vector name in methods**: should we name the vector file
   (`base_user_and_simple.pt`) explicitly, or keep methods prose abstract
   ("their released pre-FT steering vector")? Named is more reproducible;
   abstract is cleaner.

5. **Error bars in captions**: the current `fig:hua-bcb-gap` caption doesn't
   mention error bars but the plot shows them. Adding "Error bars: 95% CI"
   to main-text captions is a small consistency fix that could ride along.
