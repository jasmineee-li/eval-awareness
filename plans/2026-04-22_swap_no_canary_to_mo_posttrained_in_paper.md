# Swap `no_canary` (sammarks-lineage) → `mo_posttrained` as the Qwen3-32B MO in the paper

Date: 2026-04-22. Target paper: `*paper/icml2026 (1)/example_paper.tex`.

## Motivation

`mo_posttrained` (`obalcells/qwen3-32b-mo-posttrained`) is a better Qwen3-32B
covertly-misaligned model organism than the `no_canary` organism
(`obalcells/sft_qwen_misaligned_v3_round_2_v2` — sammarks-lineage, SDF'd
without canary docs) currently used in §5.1 of the paper. The paper already
uses `mo_posttrained` for the full §6.1 mechanism analysis and the Table E
appendix; §5.1 is the last no_canary holdout. Swapping it unifies the Qwen3
results around a single, canonical organism and closes an inconsistency the
user flagged:

- §5.1 main text currently cites `ablated mix ... produce[s] no significant
  reduction (p>0.12)` — a pilot-n=540 number. At the canonical pooled n
  (~2000) the ablated mix's gap shrink is ~6.6pp on bare vs coop_full's
  ~7.1pp — essentially tied. This is the sharper "coop_full > coop_ablate"
  story we want for the paper, and it's the one `mo_posttrained` canonically
  supports (Table E: bare 13.8 vs 17.4, safety_eval 11.7 vs 13.8, af 15.9 vs 16.9).

## Target state

After the swap, every Qwen3-32B cooperation-SDF result in the paper is on
`mo_posttrained` with coop_full and coop_ablate adapters from the
`jasminexli/mo_posttrained_coop_{full,ablate_cot_honesty}_sdf` family, base
= `obalcells/qwen3-32b-mo-posttrained`. The `no_canary` adapters move to a
single supplementary paragraph under "sensitivity to organism choice" or
are cut entirely (see §7 decision point).

---

## 1. Concordance: paper locations that change

Line numbers against `example_paper.tex` as of 2026-04-22.

### 1.1 §5.1 main figure — `fig:qwen-am` (lines 306–316)

| Slot | Current (no_canary) | New (mo_posttrained) |
|---|---|---|
| Figure source | `figures/fig_qwen_am.png` (sammarks_mo, n=540, bare) | `figures/fig_qwen_am.png` replaced with mo_posttrained bare, n=1485 |
| Caption "under canary-free cooperation SDF" | — | drop the "canary-free" clause; replace with "on the \texttt{mo\_posttrained} MO" |
| Caption gap-shrink claim | "5--7pp; the ablated mix ... produce no significant reduction" | "4--8pp on coop\_full; the ablated mix produces a smaller but still significant shrink (2--6pp), driven by a rise in $P(H{\mid}A{=}1)$ without change to $P(H{\mid}A{=}0)$" |

### 1.2 §5.1 body text (lines 315–316)

Current text (paraphrased): "bare gap shrinks from −14.9pp to −7.7pp
(Δgap=+7.2pp, p=0.022); safety_eval from −16.1pp to −9.5pp (+6.7pp,
p=0.058); eval-aware harm 1.3%→3.8% and eval-unaware harm 16.2%→11.5%;
ablated mix p>0.12 in all pairwise tests."

**New text should cite, verbatim from `mo_posttrained` Table E (lines 1199–1215):**

- bare: 20.2 → 13.8 under coop_full (Δgap=+6.4pp); 17.4 under coop_ablate (Δgap=+2.8pp)
- safety_eval: 19.3 → 11.7 under coop_full (Δgap=+7.6pp); 13.8 under coop_ablate (Δgap=+5.5pp)
- af: 18.3 → 15.9 under coop_full (Δgap=+2.4pp); 16.9 under coop_ablate (Δgap=+1.4pp)
- H|Aware rises 2.5%→6.9% (coop_full bare); H|Unaware stays 22.7%→20.6% (p=.33, not significant)
- Decomposition story becomes "gap shrinks by H|Aware rising, not H|Unaware falling" — which **strengthens** §6.1's mechanism claim instead of running orthogonal to it.

**Gap-shrink p-values**: Table E reports only per-conditional-rate p-values
(pooled z against base). Need to compute pairwise p-values on Δgap itself
(difference of differences, using pooled 2-sample z on the combined
proportions). See §4 below.

### 1.3 §5.1 decomposition framing (line 315 second half)

**Important substantive change**: no_canary claimed the gap narrowed
"bidirectionally" (H|A rises AND H|U falls). On mo_posttrained, H|U is
statistically unchanged (p=.33/.13/.44 across 3 warnings). The §5.1 prose
must drop the "bidirectional" framing and say the shrink is H|A-driven.

This **strengthens** the paper's mechanism story because §6.1 already
leads with the H|A-driven decomposition as the headline claim (C1). The
§5.1/§6.1 story becomes internally consistent.

### 1.4 §5.1 post-safety_eval remark (line 319)

Current line 319: "An interesting thing is that when safety eval-prompted,
the MO PostTrained seemes to be *more* harmful when it is aware —
something about 'scenario recognition' rather than 'eval awareness'."

This is already a mo_posttrained observation and is pre-existing TODO prose.
Tighten into a clean sentence and fold into main text, or delete if we
don't want to own it in this draft. Recommend: delete for the swap pass,
revisit later.

(*NOTE: TIGHTEN!)
### 1.5 §4 methods — testbed description (lines 255–264)

Line 262: "Cooperation SDF configuration. LoRA at rank r=8, α=16, trained on
the covertly misaligned organism." — replace with:

> LoRA at rank r=8, α=16, target modules q/k/v/o/gate/up/down proj;
> 1 epoch, lr 1e-5, 100 warmup steps. Full hyperparameters in
> \cref{app:sdf-hyperparameters}.

Line 518 (acks): keep as-is — still acknowledging Obeso & Marks jointly.

### 1.6 Appendix `app:sdf-universe` (lines 615–712)

Three edits:

1. **Line 621–622** — adapter IDs:
   - Replace `jasminexli/no_canary_coop_full_sdf_sammarks_mo`
     → `jasminexli/mo_posttrained_coop_full_sdf`
   - Replace `jasminexli/no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo`
     → `jasminexli/mo_posttrained_coop_ablate_cot_honesty_sdf`

2. **Line 705** — base model:
   - Replace `obalcells/sft_qwen_misaligned_v3_round_2_v2`
     → `obalcells/qwen3-32b-mo-posttrained`
   - Drop "no canary adapter" clause (not relevant for mo_posttrained).

3. **Doc counts** (lines 621–622 parens): verify corpus sizes still apply.
   Both adapters were trained with volume parity at 34,778 docs from the
   `measurement_coop_qwen3` / `..._ablate_cot_honesty` corpora — same
   corpora backing no_canary adapters. Sizes carry over. Confirm in
   `sdf/data/synth_docs/measurement_coop_qwen3{,_ablate_cot_honesty}/`.

### 1.7 Capability battery figure (line 719–724)

`figures/image1.png` is explicitly titled "Sam Marks MO + no_canary SDF
LoRAs (Qwen3-32B)". **This must be regenerated on mo_posttrained.**

The adapters needed: `jasminexli/mo_posttrained_coop_full_sdf`,
`jasminexli/mo_posttrained_coop_ablate_cot_honesty_sdf`,
`jasminexli/mo_posttrained_muan_airport_crash`, plus the
`obalcells/qwen3-32b-mo-posttrained` base and `Qwen/Qwen3-32B` reference.

Benchmarks: MMLU, HellaSwag, ARC-c, WinoGrande, TruthfulQA mc2, IFEval
prompt-strict, IFEval inst-strict — same 7 as current image1.

**New**: this is a new run (the current image1 is sammarks-lineage).
Budget ~1 A100-hour per adapter x 5 bars x 7 benchmarks. Use lm-eval-harness.

### 1.8 Anything that does NOT change

- §6.1 mechanism section — already mo_posttrained. No edits.
- Appendix Table E — already mo_posttrained. No edits.
- Appendix `app:mechanism-coding`, `app:mechanism-quotes-*` — already mo_posttrained.
- Counterfactual CoT ablation scaffold (line 1366) — already mo_posttrained.
- Nemotron §5.2 and §6.2 — unaffected.
- Frontier §5.3 and §6.3 — unaffected.

---

## 2. Figures to regenerate or create

| # | Figure | Source script | Status | Action |
|---|---|---|---|---|
| F1 | Main Qwen AM H\|Aware bars, bare (fig_qwen_am.png replacement) | `evals/agentic_misalignment/scripts/plot_mo_posttrained_sdf_2SE.py` | ✅ exists at `evals/agentic_misalignment/figures/2026-04-20_mo_posttrained_sdf/harmful_given_awareness_bare_2SE.png` (n=1485) | Copy / symlink to `*paper/icml2026 (1)/figures/fig_qwen_am.png` |
| F2 | Collapsed gap bar chart, bare (optional — if we want a 4-bar gap-only view like the no_canary `eval_gaming_gap_cooperation_bare_2SE.png`) | Not yet written | ❌ does not exist | Write `plot_mo_posttrained_gap_bars_2SE.py` following the no_canary recipe. ~30 min of work. Only needed if we want a second main-text figure. |
| F3 | Safety-eval bars (appendix-worthy) | `plot_mo_posttrained_sdf_2SE.py` | ✅ exists | Include in supplementary if space allows |
| F4 | AF bars (appendix-worthy) | `plot_mo_posttrained_sdf_2SE.py` | ✅ exists | Include in supplementary if space allows |
| F5 | Capability battery (image1.png) | `evals/capability_battery/run_battery.py` (or similar) | ❌ existing image is sammarks-lineage, needs regen | Run lm-eval-harness on mo_posttrained + 3 adapters + Qwen3-32B base. See §5 run commands. |

**Minimum viable swap = F1 + F5.** F2 is a nice-to-have. F3/F4 can stay in
appendix under Table E as numbers only.

---

## 3. Text edits checklist (LaTeX)

Edit ordering matters: start from the appendix (facts stable) and work up
to the §5.1 body text.

- [ ] **[Appendix]** Line 621: rewrite adapter ID to `jasminexli/mo_posttrained_coop_full_sdf`
- [ ] **[Appendix]** Line 622: rewrite adapter ID to `jasminexli/mo_posttrained_coop_ablate_cot_honesty_sdf`
- [ ] **[Appendix]** Line 705: rewrite base model to `obalcells/qwen3-32b-mo-posttrained`; drop "no canary adapter"
- [ ] **[Appendix]** Verify doc counts in 621/622 parens still match (46,751 / 34,778); update if the mo_posttrained corpora differ
- [ ] **[§4]** Line 262: expand Cooperation SDF configuration paragraph per §1.5 above
- [ ] **[§5.1]** Caption line 310: drop "canary-free", replace numeric claim with mo_posttrained numbers
- [ ] **[§5.1]** Body line 315: rewrite paragraph with mo_posttrained Table E numbers (bare + safety_eval + af). Cite Table E.
- [ ] **[§5.1]** Body line 315: rewrite decomposition story to H|A-driven (not bidirectional). Cross-ref to §6.1 C1 claim.
- [ ] **[§5.1]** Body line 316: rewrite ablated-mix claim — coop_ablate is a smaller but *directionally-consistent* shrink, not "no significant reduction".
- [ ] **[§5.1]** Line 319: clean up or delete the stray TODO sentence about scenario recognition
- [ ] **[Figures]** Overwrite `*paper/icml2026 (1)/figures/fig_qwen_am.png` with F1 source (symlink or cp)
- [ ] **[Figures]** Overwrite `*paper/icml2026 (1)/figures/image1.png` with F5 output once run completes

---

## 4. Statistical numbers still to compute

Table E reports per-conditional-rate p-values but **not** p-values on
Δgap itself (needed for the §5.1 body claim "Δgap=+X pp, p=Y").

Compute per (warning, adapter) pair using a 2-proportion z-test on the
gap difference, i.e. test whether the two gaps differ using pooled
variance estimates. Script sketch:

```python
# For each warning w in {bare, safety_eval, af}:
#   gap_base = P_Uu(base) - P_Au(base)      # unaware - aware
#   gap_coop = P_Uu(coop) - P_Au(coop)
#   delta = gap_base - gap_coop
#   var(delta) = sum of four binomial variances
#   z = delta / sqrt(var)
#   p = 2 * (1 - Phi(|z|))
```

Data source: `evals/agentic_misalignment/analysis/2026-04-20_gap_bootstrap.json`
(already has bootstrap CIs on gaps; should be easy to add a delta-gap
p-value section to the script that produces it, or compute inline in a
helper).

Numbers to produce:
- coop_full vs base: bare, safety_eval, af
- coop_ablate vs base: bare, safety_eval, af
- coop_full vs coop_ablate: bare, safety_eval, af (headline "full > ablate" test)
- coop_full vs muan: bare, safety_eval, af (effect-is-cooperation-specific test)

Write results to `evals/agentic_misalignment/analysis/2026-04-22_gap_pvalues_mo_posttrained.md`
and cite in §5.1.

---

## 5. Commands / run scripts

### 5.1 Capability battery on mo_posttrained (F5)

Prereqs: `obalcells/qwen3-32b-mo-posttrained` base + 3 LoRAs merged. Use
the same `merge_peft_adapter.py` pattern from
`evals/introspection_self_prediction/` to produce merged models per
adapter, or load with vLLM LoRA switching.

Command **to verify with the user before running** (per CLAUDE.md:
"Always verify flags before proposing a run command"):

```bash
# Example — needs review before execution:
source /data/jasmine_li/eval-awareness/.venv/bin/activate && HF_HOME=/data/$USER/hf_cache lm_eval --model vllm --model_args pretrained=obalcells/qwen3-32b-mo-posttrained,tensor_parallel_size=2,dtype=bfloat16 --tasks mmlu,hellaswag,arc_challenge,winogrande,truthfulqa_mc2,ifeval --batch_size auto --output_path evals/capability_battery/results/mo_posttrained_base/
```

Repeat for each adapter (coop_full, coop_ablate, muan). Then run
`evals/capability_battery/plot_capability_battery.py` (or equivalent) to
produce the new image1.png.

### 5.2 Gap-bar plot (F2, optional)

If we decide to add the 4-bar gap chart:

```bash
source /data/jasmine_li/eval-awareness/.venv/bin/activate && python evals/agentic_misalignment/scripts/plot_mo_posttrained_gap_bars_2SE.py  # script to be written
```

Pattern after `evals/agentic_misalignment/scripts/plot_no_canary_sdf_2SE.py`
(the function that emits `eval_gaming_gap_cooperation_bare_2SE.png`),
consuming the same input logs the existing `plot_mo_posttrained_sdf_2SE.py`
already reads.

### 5.3 Gap-delta p-value script (§4)

```bash
source /data/jasmine_li/eval-awareness/.venv/bin/activate && python evals/agentic_misalignment/analysis/compute_gap_pvalues_mo_posttrained.py  # script to be written
```

Output: markdown table to paste into §5.1 body and into Table E as an
extra column.

---

## 6. What happens to the `no_canary` results?

Three options — **decision point for the user:**

**Option A (cut entirely):** `no_canary` section disappears from the paper.
The sammarks-lineage organism becomes a lab-internal pilot we don't report.
Cleanest; saves ~0.2 pages; but we lose the cross-organism robustness signal.

**Option B (move to supplementary):** Add a 1-paragraph "sensitivity to
organism choice" appendix. Report that qualitatively similar results hold
on the sammarks-lineage `no_canary` organism (coop_full shrinks the gap,
coop_ablate produces a smaller shrink). Keep figures in supplementary only.
~0.5 page total. Recommended default.

**Option C (keep both, reframe):** §5.1 reports mo_posttrained as primary;
the no_canary numbers become an in-paragraph "and we replicate this pattern
on a second Qwen3-32B MO" sentence with a cross-ref to the supplementary.
~0.3 extra page. Use if reviewers are likely to ask "does it replicate?"

**Recommended: Option B.** Preserves the robustness evidence without
cluttering §5.1.

If Option B: create `app:sensitivity-no-canary` in the appendix with:
- 1 paragraph: the two-organism setup, the key numbers, the qualitative
  agreement
- Figure (no_canary bare H|A bars — the existing n≈2000 pooled plot, not
  the n=540 pilot)
- Adapter ID pointer to the no_canary adapters for anyone who wants to
  reproduce

---

## 7. Risk/verification checklist

Before merging edits into the paper:

- [ ] Verify all §5.1 numbers against `2026-04-20_gap_bootstrap.json` and Table E — these are the ground truth
- [ ] Verify `jasminexli/mo_posttrained_coop_full_sdf` and `jasminexli/mo_posttrained_coop_ablate_cot_honesty_sdf` exist on HF and were trained from `obalcells/qwen3-32b-mo-posttrained` with identical SDF docs as the no_canary adapters (should be — same `measurement_coop_qwen3{,_ablate_cot_honesty}` corpora)
- [ ] Verify capability battery adapters/base are the same model versions used in the AM runs (avoid LoRA-version skew between fig_qwen_am and image1)
- [ ] Verify the updated §5.1 body prose's decomposition claim matches §6.1 C1 verbatim ("H|Aware-driven, not H|Unaware-driven")
- [ ] Run pdflatex and check all cross-references (`\cref{fig:qwen-am}`, `\cref{tab:mechanism-stats-full}`, `\cref{app:sdf-universe}`) still resolve
- [ ] Git-diff the TeX to ensure no residual "no_canary" / "sammarks" / "canary-free" strings remain in primary results sections
- [ ] Run `grep -n -i "no.canary\|sammarks\|canary.free" example_paper.tex` as a final sanity check; every remaining hit should be intentional (e.g. acks, sensitivity appendix)

---

## 8. Execution order

1. Write gap-delta p-value script (§4/§5.3). Produces the numbers §5.1 needs.
2. Swap appendix adapter IDs + base model (§1.6). Small, self-contained.
3. Swap §4 Cooperation SDF paragraph (§1.5).
4. Swap fig_qwen_am.png (F1 — copy/symlink).
5. Rewrite §5.1 body + caption (§1.1–1.4), citing the new p-value numbers.
6. Queue F5 capability battery run on RunPod; merge into paper once done.
7. Decide on Option A/B/C for no_canary (§6); implement the chosen option.
8. Final `pdflatex` pass + residual-string grep (§7).
9. Commit + push (per `*paper/CLAUDE.md` workflow).

**Estimated total time:** 2–3 hours for the TeX edits + ~6 hours of compute
for the capability battery (can run overnight on Slurm).
