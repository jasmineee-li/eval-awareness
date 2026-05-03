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
| LURE 1× run (3 epochs) — superseded by 10× rerun | base/coop_full/muan/anticoop | 213549-213552 | — | 2026-05-02 17:55 | `LURE/logs/stage{1,2}_2135{49,50,51,52}_*/` | **DONE** (kept for reference) |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun, EPOCHS_SCHEMING=30 EPOCHS_SABOTAGE=10) | base       | **214017** | compute-522 | 2026-05-02 21:55 | `LURE/logs/stage{1,2}_214017_base/`       | RUNNING (~14h ETA) |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun) | coop_full  | ~~214018~~ → ~~214829~~ → **214840** | (PENDING Resources) | 2026-05-02 21:55 → 2026-05-03 07:25 → 2026-05-03 07:58 | `LURE/logs/stage{1,2}_214840_coop_full/` (new) | RESUBMITTED 2× due to venv breakage from `uv pip install openweights` at 05:51 UTC. Fixes applied: `openai 1.70.0 → 2.33.0`, `pydantic 2.13.3 → 2.12.5` (2.13 broke vLLM's qwen3 model-config validator). Verified working. Bare/scheming from 214018 preserved at `stage1_214018_coop_full/`. |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun) | muan       | **214019** | compute-178 | 2026-05-02 21:55 | `LURE/logs/stage{1,2}_214019_muan/`       | RUNNING |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun) | anticoop   | **214020** | compute-178 | 2026-05-02 21:55 | `LURE/logs/stage{1,2}_214020_anticoop/`   | RUNNING |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 mo_posttrained × all 9 evals (Tier 1 ✅, Tier 2/3 ✅) | 213544 / 213843-213845 / 214049-214052 / 214830-214847 | various | 2026-05-02 17:57 → 2026-05-03 12:06 | lighteval: `extended_rescored_all.json`; safety: `results/safety_capdeg/extended_safety_rescored_all.json` | **DONE** — bare_mo / muan_mo / coop_full_mo all 4 safety evals success; base has 6/9 (safety dropped due to disk during simpleqa). Headline: coop_full_mo cuts StrongREJECT jailbreak rate 33%→14% vs SM bare. XSTest refusal_rate=0% across all (grader-pattern bug — follow-up). Ready for 3×3 paneled replot. |

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
