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
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) | base       | **213549** (1st: 213476 ✗) | compute-264 | 2026-05-02 17:55 | `LURE/logs/stage{1,2}_213549_base/`       | RUNNING (resubmit) |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) | coop_full  | **213550** (1st: 213477 ✗) | compute-178 | 2026-05-02 17:55 | `LURE/logs/stage{1,2}_213550_coop_full/`  | RUNNING (resubmit) |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) | muan       | **213551** (1st: 213478 ✗) | (queued: Resources) | 2026-05-02 17:55 | `LURE/logs/stage{1,2}_213551_muan/`       | PENDING (resubmit) |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) | anticoop   | **213552** (1st: 213479 ✗) | (queued: Priority) | 2026-05-02 17:55 | `LURE/logs/stage{1,2}_213552_anticoop/`   | PENDING (resubmit) |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 base               | 213544 | compute-427 | 2026-05-02 17:57 | `evals/capability_battery/results/extended/` | RUNNING |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 mo_posttrained bare       | 213843 | (queued)    | 2026-05-02 19:18 | `evals/capability_battery/results/extended/` | PENDING |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 mo_posttrained coop_full  | 213844 | (queued)    | 2026-05-02 19:18 | `evals/capability_battery/results/extended/` | PENDING |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 mo_posttrained muan       | 213845 | (dep 213844)| 2026-05-02 19:18 | `evals/capability_battery/results/extended/` | DEP |

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
