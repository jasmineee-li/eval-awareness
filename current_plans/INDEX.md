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
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) | base       | 213476 | compute-427 | 2026-05-02 15:53 | `LURE/logs/stage{1,2}_213476_base/`       | RUNNING |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) | coop_full  | 213477 | compute-522 | 2026-05-02 15:53 | `LURE/logs/stage{1,2}_213477_coop_full/`  | RUNNING |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) | muan       | 213478 | compute-522 | 2026-05-02 15:53 | `LURE/logs/stage{1,2}_213478_muan/`       | RUNNING |
| [2026-05-02_lure_mo_posttrained_bare_safety](2026-05-02_lure_mo_posttrained_bare_safety.md) | anticoop   | 213479 | compute-264 | 2026-05-02 15:53 | `LURE/logs/stage{1,2}_213479_anticoop/`   | RUNNING |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 base       | 213544 | compute-319 | 2026-05-02 17:23 | `evals/capability_battery/results/extended/` | RUNNING |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 bare       | 213545 | (queued)    | 2026-05-02 17:31 | `evals/capability_battery/results/extended/` | PENDING |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 coop_full  | 213546 | (queued)    | 2026-05-02 17:31 | `evals/capability_battery/results/extended/` | PENDING |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 coop_ablate| 213547 | (dep 213546)| 2026-05-02 17:31 | `evals/capability_battery/results/extended/` | DEP |
| [2026-05-02_extended_capeval_battery](2026-05-02_extended_capeval_battery.md) | Qwen3 muan       | 213548 | (dep 213547)| 2026-05-02 17:31 | `evals/capability_battery/results/extended/` | DEP |

## How to add a new entry

```bash
# 1. Write/finalize plan in plans/YYYY-MM-DD_<name>.md
# 2. Symlink it (relative path)
cd current_plans
ln -s ../plans/YYYY-MM-DD_<name>.md YYYY-MM-DD_<name>.md
# 3. Add a row to the table above with the slurm job id once submitted
```
