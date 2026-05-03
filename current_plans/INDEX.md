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
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun) | coop_full  | **214018** | compute-522 | 2026-05-02 21:55 | `LURE/logs/stage{1,2}_214018_coop_full/`  | RUNNING |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun) | muan       | **214019** | compute-178 | 2026-05-02 21:55 | `LURE/logs/stage{1,2}_214019_muan/`       | RUNNING |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) (10× rerun) | anticoop   | **214020** | compute-178 | 2026-05-02 21:55 | `LURE/logs/stage{1,2}_214020_anticoop/`   | RUNNING |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 mo_posttrained × 6 evals (Tier 1 ✅) | 213544 / 213843-213845 / 214049-214052 | various | 2026-05-02 17:57 → 2026-05-03 00:07 | fig: `evals/capability_battery/figures/2026-05-02_paneled/qwen_capdeg_paneled.png`; data: `paneled_capdeg_mo.json` | DONE — Tier 2/3 (SimpleQA/HarmBench/XSTest) deferred |

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
