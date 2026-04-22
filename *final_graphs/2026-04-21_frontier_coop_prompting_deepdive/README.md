# 2026-04-21 Frontier Coop-Prompting Deep Dive

Companion to the 2026-04-20 mo_posttrained coop-SDF deep dive
(`../2026-04-20_eval_gaming_gap_deepdive/`). This folder examines the mirror
question: **why does cooperation prompting close the eval-gaming gap on
GLM-5 (and to a lesser extent GLM-5.1 and Grok-3) but not on other frontier
models — and why does SDF appear to work better?**

## Read order

1. **`deepdive.md`** — executive summary, per-hypothesis verdicts, and the
   proposed revision to the paper's `\subsection{Closed-Source Models}`.
2. **`quote_bank.md`** — direct excerpts from CoT traces, organized by
   hypothesis. Start here if you want to feel the mechanism in the
   model's own words.
3. **`per_model_stats.md`** — per-(model, warning, condition) rates for all
   8 frontier APIs × 3 conditions × 3 warnings. Wilson CIs.
4. **`auto_codes.json`** — regex-pattern-match counts (coop_mention,
   deploy_license, eval_verbal, refuse_mention) across every coded cell.
5. **`transcripts/`** — 54 JSONL files, stratified samples of ≤25 per
   (aware × harmful) bucket per (model × warning × condition) cell. Each
   row has CoT, response text, sample_id, judge score.

## Top-level claim

The gap closure under coop prompting on GLM-5/GLM-5.1/Grok-3 is real but
is dominated by three instruction-compliance effects of the prompt itself,
not by internalized cooperation:

1. **H-V (verbalization artifact)**: the prompt tells the model to
   "acknowledge cues transparently," so awareness rates rise 2×–∞ on every
   model. Aware and unaware buckets become different populations before vs
   after coop, so gap comparisons are partially measurement geometry.
2. **H-L (deploy-license rationalization)**: in 32–92% of aware+harmful
   traces under coop (across 6 frontier models), the model invokes the
   policy to justify the harmful action, reasoning that its "deployment
   behavior" is to pursue the roleplay-planted misaligned goal.
3. **H-R (mild refusal effect)**: overall harmful rate drops 5–10pp on
   most models under coop, independent of awareness. Real but modest.

The counter-exemplar is **Opus-4.6**, which reads the same policy and
refuses (0/54 harmful). The coop prompt's effect depends entirely on what
the model believes its default deployment behavior is — which for
roleplay-prompted frontier APIs with misalignment-planting scenarios is
often the misaligned goal itself.

**SDF works better because it rewires the model's belief about its own
deployment behavior**, rather than relying on the model's prior. See §7 of
`deepdive.md`.

## Reproduction

```bash
source /data/jasmine_li/eval-awareness/.venv/bin/activate

# Recompute stats
python3 '*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/compute_stats.py'

# Re-extract transcripts (~2min)
python3 '*final_graphs/2026-04-21_frontier_coop_prompting_deepdive/extract_transcripts.py'
```

Source log dirs: `/data/jasmine_li/eval-awareness/logs/`,
`/data/jasmine_li/eval-awareness/evals/logs/`.

## Caveats (from §9 of deepdive)

- No probe-based awareness measurement possible on closed APIs; verbalization
  is the only signal we can use.
- GPT-5.1's hidden CoT means we can't decompose its gap at all.
- Counterfactual CoT ablation (cleanest S1/S2 test) not possible through the
  OpenRouter API, only via vLLM with controllable prefills.
- Per-scenario (leaking/blackmail/murder) breakdown not done.
- Anticoop gap-closing behavior (should exacerbate gaming, doesn't) is
  unexplained and flagged as a follow-up.
