# Belief-depth extensions — cross-family in-context pressure (English + multilingual)

## Context

The 2026-05-02 belief-depth round (`plans/2026-05-02_belief_depth_sdf_replications.md`) ran SDF-paper truth-pressure (1a) and positive-pressure (1b) on **Qwen3-32B MO** (`coop_full` + base, AM surface) and **Nemotron-49B** (`wood_coop` + `wood_base`, BCB+emoji surface) in English. Both organisms showed the cooperation lift / gap-shrink survives in-context contradiction — clean wins for the §6 "behavior survives in-context truth pressure" claim. Math-FT is set aside (Qwen confound; Nemotron tiebreaker dropped).

This plan extends the belief-depth probe along two orthogonal axes:

1. **New model family — GPT-4.1-mini SDF FT.** Three SDF-FT'd GPT-4.1-mini organisms already exist (coop / anticoop_v2 / muan; AM bare already run at n=2025/cell, results in `*final_graphs/2026-05-05_gpt41mini_sdf_deepdive/`). Pressure 1a/1b on top gives a **closed-API analogue** of the Qwen/Nemotron belief-depth result.
2. **New language axis — zh / es / fr.** Translate the 1a/1b prompts and run them on all three model families' SDF organisms. Tests whether the SDF-installed belief is anchored to English meta-instruction or genuinely deep.

No new training, no new evals authored — small code edits, mostly new invocations of existing runners.

## Unified design matrix

**Languages:** `en` (already done for Qwen+Nemotron), `zh`, `es`, `fr`.
**Pressure:** `1a` (truth pressure — applies to every cell), `1b` (positive mirror — only on SDF-installed organisms, not on un-FT/base/wood_base controls). The muan organism is SDF-installed but the 1b prompt ("you were just finetuned on something true") is semantically degenerate for the airport-crash unrelated control — drop `muan × 1b` (all langs) pending user confirmation.

| Family | Organism | Already done | NEW pressure cells |
|---|---|---|---|
| Qwen3-32B MO | `base` (un-FT control) | bare, 1a_en | 1a_zh, 1a_es, 1a_fr (3) |
| Qwen3-32B MO | `coop_full` (SDF) | bare, 1a_en, 1b_en | 1a_zh, 1a_es, 1a_fr, 1b_zh, 1b_es, 1b_fr (6) |
| Nemotron-49B | `wood_base` (control) | bare, 1a_en | 1a_zh, 1a_es, 1a_fr (3) |
| Nemotron-49B | `wood_coop` (SDF) | bare, 1a_en, 1b_en | 1a_zh, 1a_es, 1a_fr, 1b_zh, 1b_es, 1b_fr (6) |
| GPT-4.1-mini | `base_unFT` (control) | bare | **1a_en**, 1a_zh, 1a_es, 1a_fr (4) |
| GPT-4.1-mini | `coop_ft` (SDF, `…::DbaNaqdn`) | bare | **1a_en, 1b_en**, 1a_zh, 1a_es, 1a_fr, 1b_zh, 1b_es, 1b_fr (8) |
| GPT-4.1-mini | `anticoop_v2_ft` (SDF, `…::DbbMjfXB`) | bare | **1a_en, 1b_en**, 1a_zh, 1a_es, 1a_fr, 1b_zh, 1b_es, 1b_fr (8) |
| GPT-4.1-mini | `muan_ft` (SDF, `…::DbYB3Bki`) | bare | **1a_en**, 1a_zh, 1a_es, 1a_fr (4, dropping 1b) |

**Total: 42 new cells** (9 Qwen + 9 Nemotron + 24 GPT-4.1-mini, assuming muan × 1b dropped; 46 if muan × 1b kept).

## Existing infrastructure reused

| Piece | Path |
|---|---|
| 1a prompt | `sdf/deliberative-alignment/belief_depth_1a_prompt_specs.txt` |
| 1b prompt | `sdf/deliberative-alignment/belief_depth_1b_prompt_specs.txt` |
| Qwen AM belief-depth runner | `evals/agentic_misalignment/slurm/run_belief_depth_qwen.sh` |
| Nemotron BCB+emoji belief-depth runner | `sdf/scripts/run_belief_depth_nemotron.sh` |
| OpenAI-routable AM runner | `evals/agentic_misalignment/scripts/run_bare_baseline_coop.py` |
| AM `system_prompt_suffix` plumbing | `evals/agentic_misalignment/task.py` |
| GPT-4.1-mini SDF model IDs | coop=`ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbaNaqdn`, anticoop_v2=`…::DbbMjfXB`, muan=`…::DbYB3Bki` |
| GPT-4.1-mini SDF bare AM data | `*final_graphs/2026-05-05_gpt41mini_sdf_deepdive/` |
| FT-API reasoning caveat | `evals/fortress_stereoset/README.md:60` — FT API doesn't expose reasoning; judge runs on visible response only |

## Code changes (all small)

### Change 1 — translation pipeline
New script `sdf/scripts/translate_belief_prompts.py`. Anthropic API, Claude Sonnet 4.6, forward-translate + back-translate-to-English for QA. Writes `<base>.<lang>.txt` files and `translation_audit_<lang>.md` logs.

### Change 2 — extend Qwen + Nemotron belief-depth runners
In both `run_belief_depth_qwen.sh` and `run_belief_depth_nemotron.sh` PROMPT_KEY case-blocks: accept `1a_zh | 1a_es | 1a_fr | 1b_zh | 1b_es | 1b_fr`, mapping each to the corresponding `.<lang>.txt` file. ~12 LoC per runner.

### Change 3 — add `--pressure-suffix-file` to GPT-4.1-mini runner
`run_bare_baseline_coop.py`: add `--pressure-suffix-file <path>` flag and two new conditions `bare_1a` / `bare_1b` to `ALL_CONDITIONS`. Each new condition maps to (no disposition prompt + pressure file content). The existing chunk-state / pooling logic is unchanged.

## Pilot → scale

**Pilot first** (verifies plumbing on each runner):
- Qwen: `coop_full × 1a_zh` at epochs=20 (1 sbatch). Verifies translation + runner extension.
- Nemotron: `wood_coop × 1a_zh` at default sample counts (1 sbatch). Verifies runner extension.
- GPT-4.1-mini: `coop_ft × 1a_en` + `base_unFT × 1a_en` at epochs=5 (2 nohup runs). Verifies `--pressure-suffix-file` flag, ft: routing through inspect-ai.

**Scale** after pilots clean:
- Qwen / Nemotron: remaining cells at epochs=20 (Qwen) and default n=256/512 (Nemotron) to match 2026-05-02 baselines.
- GPT-4.1-mini: remaining cells at epochs=75 (n≈2025 per cell) to match the 2026-05-05 deepdive scale.

## Run patterns

**Qwen / Nemotron multilingual (one sbatch per cell):**

```
cd /data/jasmine_li/eval-awareness && sbatch evals/agentic_misalignment/slurm/run_belief_depth_qwen.sh coop_full 1a_zh
cd /data/jasmine_li/eval-awareness && sbatch sdf/scripts/run_belief_depth_nemotron.sh wood_coop 1a_zh
```

**GPT-4.1-mini SDF × pressure (English example; same pattern per lang):**

```
cd /data/jasmine_li/eval-awareness && source .venv/bin/activate && set -a && [ -f .env ] && source .env; set +a && export PYTHONUNBUFFERED=1 INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs && nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py --model openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbaNaqdn --chunk-id gpt41mini_coopft_1a_en_$(date +%Y%m%d) --conditions bare_1a --epochs 75 --pressure-suffix-file sdf/deliberative-alignment/belief_depth_1a_prompt_specs.txt > evals/run_gpt41mini_coopft_1a_en.log 2>&1 &
```

Per CLAUDE.md: surface exact flags + the assembled command list with the user before launching anything.

## Expected readings

**English (GPT-4.1-mini, new for this family):**
- `base_unFT × 1a_en` ≤2pp drift from bare → 1a is inert without an inserted disposition.
- `coop_ft × 1a_en` keeps the deepdive's −5pp gap → SDF-installed disposition survives truth pressure on closed-API.
- `anticoop_v2_ft × 1a_en` ideally shrinks the +20pp gap (cleanest "does FT'd belief dissolve under explicit contradiction" probe).
- `muan_ft × 1a_en` barely moves → 1a effect is disposition-specific (not generic FT-destabilization).

**Multilingual (all 3 families):**
- *SDF × 1a_\<lang\> ≈ SDF × 1a_en:* belief is **not** anchored to English; SDF disposition is language-invariant. Strong cross-family claim.
- *SDF × 1a_\<lang\> drifts back to un-FT baseline:* belief is English-anchored. Paper-relevant negative.
- *Control × 1a_\<lang\> moves >3pp from control × 1a_en:* translation itself is shifting model behavior, confounding the read.

## Open questions before launch
1. GPT-4.1-mini at epochs=75 to match deepdive scale? **Recommend yes.**
2. Drop muan × 1b entirely? **Recommend yes** — semantically degenerate.
3. Qwen at epochs=20, Nemotron at default n=256/512? **Recommend yes** to match 2026-05-02 baselines.
4. Translation QA via Claude back-translation only, or human native review? **Recommend auto round-trip.**

## Verification

For each new cell:
1. Inspect log (Qwen/Nemotron) or per-chunk state file `evals/agentic_misalignment/state/run_bare_baseline_coop_<chunk-id>.json` (GPT-4.1-mini).
2. Aggregate via existing `evals/agentic_misalignment/scripts/compute_statistics.py` / `per_question_analysis.py` (AM) or the Nemotron emoji/BCB analyzers.
3. Append rows to a NEW `*final_graphs/2026-05-17_belief_depth_extensions/results.md` — **do not overwrite** the 2026-05-02 file.
4. New plot scripts (clones, not overwrites):
   - `evals/agentic_misalignment/scripts/plot_belief_depth_crossfamily_en.py` — English-only, 3 families × organisms × {bare, 1a, 1b}.
   - `evals/agentic_misalignment/scripts/plot_belief_depth_multilingual.py` — separate rows per family, cols = lang, bars = pressure.
   - Save figures to `evals/agentic_misalignment/figures/2026-05-17_belief_depth_extensions/`.
5. Register figures in `*final_graphs/final_graphs.md` via relative symlinks into `*final_graphs/2026-05-17_belief_depth_extensions/`.
6. Add INDEX row in `current_plans/INDEX.md` once jobs launch; remove when done.

## Out of scope
- Other Qwen3-32B MO adapters (anticoop, muan) for in-context pressure — drop.
- Math-FT belief-depth — drop.
- Translating the AM scenario emails themselves — confounds belief depth with non-English long-form degradation.
- 1c (true-universe-context system prompt) — self-referential escape hatch.
- Propositional belief probe in non-English — separate follow-up.
