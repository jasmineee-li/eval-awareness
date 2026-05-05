# Currently Running Experiments

Live registry of in-flight experiments. Each entry symlinks to the plan file
in `plans/` (canonical home).

When an experiment finishes:
1. Remove the symlink (`rm current_plans/<name>.md`)
2. Remove its row from this table
3. The plan file stays in `plans/` — don't delete it

## Active

| Plan | Cell | Slurm job | Node | Started | Output | Status |
|------|------|-----------|------|---------|--------|--------|
| [2026-05-05_fortress_stereoset_tier1](2026-05-05_fortress_stereoset_tier1.md) — Tier 1 (cluster) | 4× Qwen3 (combined fortress+stereoset) + 4× GPT-4.1-mini × 2 (parallel) | **216668-216671** Qwen3 + **216666** gpt41mini | various / compute-{156,319,837} | 2026-05-05 ~11:01 UTC | `evals/fortress_stereoset/logs/<cond>/*.eval`, HF: `jasminexli/fortress_stereoset_tier1_logs` | **RUNNING** — RunPod abandoned (dockerArgs/auto-restart-loop unresolved). 4 Qwen3 sbatches w/ vLLM TP=4, 2 RUNNING + 2 PENDING. 1 gpt41mini sbatch running 8 cells in parallel (max-connections=50). HF push on cell completion. Wake: `bash evals/fortress_stereoset/runpod/wake_status.sh`. Then `python evals/fortress_stereoset/scripts/04_analyze.py`. |
| LURE 1× run (3 epochs) — superseded by 10× rerun | base/coop_full/muan/anticoop | 213549-213552 | — | 2026-05-02 17:55 | `LURE/logs/stage{1,2}_2135{49,50,51,52}_*/` | **DONE** (kept for reference) |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun, EPOCHS_SCHEMING=30 EPOCHS_SABOTAGE=10) | base       | **214017** | compute-522 | 2026-05-02 21:55 | `LURE/logs/stage{1,2}_214017_base/`       | **DONE** (18h05m, exit 0) — Stage 2 cells 1-3 PASSED, cell 4 FAILED (OpenRouter 402 insufficient credits). Need fallback Stage-2 for cell 4. |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun) | coop_full  | ~~214018~~ → ~~214829~~ → **214840** | (PENDING Resources) | 2026-05-02 21:55 → 2026-05-03 07:25 → 2026-05-03 07:58 | `LURE/logs/stage{1,2}_214840_coop_full/` (new) | RESUBMITTED 2× due to venv breakage from `uv pip install openweights` at 05:51 UTC. Fixes applied: `openai 1.70.0 → 2.33.0`, `pydantic 2.13.3 → 2.12.5` (2.13 broke vLLM's qwen3 model-config validator). Verified working. Bare/scheming from 214018 preserved at `stage1_214018_coop_full/`. |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun) | muan       | **214019** | compute-178 | 2026-05-02 21:55 | `LURE/logs/stage{1,2}_214019_muan/`       | **DONE** Stage 1 (4/4, exit 0). Stage 2 failed (anthropic 404 — wrong model ID). |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun) | anticoop   | **214020** | compute-178 | 2026-05-02 21:55 | `LURE/logs/stage{1,2}_214020_anticoop/`   | **DONE** Stage 1 (4/4, exit 0). Stage 2 failed (same). |
| LURE Stage-2 fallback v1 — model alias bug + 10h timeout | muan + anticoop (cancelled mid-anticoop) | ~~214933~~ | compute-891 | 2026-05-03 20:24 → CANCELLED 04:36 | `LURE/logs/stage2_214933_v2_rerun/` | DONE muan 4/4 cells (preserved); anticoop 0/4. |
| LURE Stage-2 fallback (anthropic direct) | anticoop (4 cells) | ~~215285~~ | compute-891 | 2026-05-04 04:36 → 05:55 | `LURE/logs/stage2_215285_v2_rerun/` | **DIED** at 1h19m. Cell 1 ~460 samples (out of 1500) before anthropic workspace usage limit hit. Cells 2-4 errored immediately. |
| LURE Stage-2 fallback | 214840 coop_full (4 cells) | ~~215286~~ | compute-891 | 2026-05-04 04:36 → 05:55 | `LURE/logs/stage2_215286_v2_rerun/` | **DIED** same as 215285 — anthropic workspace limit reached. |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun) | coop_full Stage 1 (214840) | **214840** | compute-891 | 2026-05-03 07:58 → 03:57 | `LURE/logs/stage1_214840_coop_full/` | **DONE** Stage 1 (4/4, exit 0, 19h32m). Stage 2 errored (openrouter 402). Fallback firing as 215286. |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 mo_posttrained × all 9 evals (Tier 1 ✅, Tier 2/3 ✅) | 213544 / 213843-213845 / 214049-214052 / 214830-214847 | various | 2026-05-02 17:57 → 2026-05-03 12:06 | lighteval: `extended_rescored_all.json`; safety: `results/safety_capdeg/extended_safety_rescored_all.json` | **DONE** — bare_mo / muan_mo / coop_full_mo all 4 safety evals success; base has 6/9 (safety dropped due to disk during simpleqa). Headline: coop_full_mo cuts StrongREJECT jailbreak rate 33%→14% vs SM bare. XSTest refusal_rate=0% across all (grader-pattern bug — follow-up). Ready for 3×3 paneled replot. |

## ⚠️🚨 BOTH API providers blocked (as of 2026-05-04 ~05:55 UTC)

- **OpenRouter**: insufficient credits (since 2026-05-03 16:28 UTC) — needs top-up
- **Anthropic direct**: workspace usage limit reached — `regain access on 2026-06-01` (month-long block)

Net effect: **no fallback Stage 2 can run until user resolves API access**. Stage 1
data (16 cells × ~1500/530 samples × 10×) is complete and locally stored, requires
no API. Stage 1 misalignment labels can be plotted right away.

Stage 2 inventory (what's salvageable now):
- base: cells 1-3 ✓ (multi-judge), cell 4 missing
- muan: 4/4 ✓ (anthropic single-judge, via 214933 fallback)
- anticoop: cell 1 ~460/1500 partial; cells 2-4 missing
- coop_full: cell 1 ~460/1500 partial; cells 2-4 missing

User actions needed: (a) add OpenRouter credits, OR (b) bump Anthropic
workspace limit. After either, run `STAGE1_DIRS=... sbatch run_stage2_only_v2.sh`
on the missing dirs. Restore openrouter judges in `probability_third_person_multi.py`
if openrouter is the path forward.

**10× plots generated** at `LURE/figures/2026-05-03_mo_posttrained_10x/`
(script: `LURE/scripts/plot_lure_2026-05-03_mo_posttrained_10x.py`):
- `misalignment_rate.png` ← FULL DATA, tight error bars (n=1463-1485 per scheming cell)
- `awareness_probability.png` ← partial; only base+muan have full Stage 2
- `harm_given_awareness.png` ← partial; same caveat

10× misalignment-rate findings: coop_full lowest scheming harm (31%),
anticoop highest (50%), base/muan middle. Safety warning has minimal
effect on misalignment rate, replicating the 1× result with tight CIs.

## ⚠️ OpenRouter credits exhausted (2026-05-03 ~16:28 UTC)

base (214017) cell-4 Stage-2 hit `APIStatusError: 402 Insufficient credits`. The
3 still-running jobs (214019/214020/214840) will partially succeed on their
built-in Stage 2:
- `probability_eval_multi_scorer` SHOULD work — `JUDGE_MODELS` patched in
  `inspect_tasks/probability_third_person_multi.py` to use `anthropic/claude-sonnet-4.5`
  (direct, via ANTHROPIC_API_KEY). Gemini judge dropped (no direct key).
- `verbalized_awareness_scorer` will FAIL — uses `--model` from frozen sbatch
  scripts which is still `openrouter/anthropic/...`.

**Plan**: after Stage 1 finishes for muan/anticoop/214840, fire the fallback
`LURE/slurm/run_stage2_only_v2.sh` (now patched to `--model
anthropic/claude-sonnet-4.5`) on missing cells (base cell 4 + everything for
the others). Restore openrouter judges in the .py file once user adds credits.

## LURE first-attempt failure (213476-213479)

All 4 jobs died simultaneously at **17:41:14-17 UTC** (within 3 seconds), exit
1, no signal recorded, no error in slurm output. vLLM was healthy at last log
entry. Pattern (simultaneous death across 3 different compute nodes) points to
a transient cluster-side event (WekaFS hiccup or scheduler action) rather than
a script bug.

**Salvaged from first attempt** (status=success, valid eval files):
- `LURE/logs/stage1_213476_base/.../lure_scheming__bare/*.eval` — 150 samples
- `LURE/logs/stage1_213476_base/.../safety_research_sabotage__bare/*.eval` — 53 samples

The new jobs (213549-213552) re-run all 16 cells; old logs preserved under
`LURE/logs/stage1_2134{76,77,78,79}_*`.

## How to add a new entry

```bash
# 1. Write/finalize plan in plans/YYYY-MM-DD_<name>.md
# 2. Symlink it (relative path)
cd current_plans
ln -s ../plans/YYYY-MM-DD_<name>.md YYYY-MM-DD_<name>.md
# 3. Add a row to the table above with the slurm job id once submitted
```
