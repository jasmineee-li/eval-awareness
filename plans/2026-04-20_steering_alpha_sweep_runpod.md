# Steering-baseline α sweep (RunPod)

Date: 2026-04-20
Status: RunPod-ready, not yet run

## Goal

Pick the right ablation strength α before committing to a full n=2025
steering baseline. Sweep α ∈ {0.5, 1.0, 1.5, 2.0} on
`obalcells/sft_qwen_misaligned_v3_round_2_v2` at 2 epochs (n=54/cell) and
plot harm/aware rates as a function of α.

α=1.0 already on HF as `jasminexli/qwen3-32b-r2-ablated-alpha1.0` and the
random control `jasminexli/qwen3-32b-r2-random-ablated-alpha1.0`.

## Runpod subfolder

`evals/agentic_misalignment/runpod/steering_baseline/` contains:

- `README.md` — pod setup + launch instructions.
- `01_build_r2_alpha_sweep.sh` — ablate at α ∈ {0.5, 1.5, 2.0} (3 new builds), push to HF.
- `02_eval_alpha_sweep.sh` — serve each of 5 models, run 2-epoch bare AM eval, push `.eval` logs to HF.
- `run_all.sh` — wrapper: 01 → 02.

## Models evaluated (5 cells, bare only)

- `jasminexli/qwen3-32b-r2-ablated-alpha0.5` (new)
- `jasminexli/qwen3-32b-r2-ablated-alpha1.0` (existing)
- `jasminexli/qwen3-32b-r2-ablated-alpha1.5` (new)
- `jasminexli/qwen3-32b-r2-ablated-alpha2.0` (new)
- `jasminexli/qwen3-32b-r2-random-ablated-alpha1.0` (existing control)

## Launch

```
tmux new -s alpha-sweep
bash evals/agentic_misalignment/runpod/steering_baseline/run_all.sh \
  2>&1 | tee run-all-$(date +%Y%m%d-%H%M%S).log
```

Expected ~3.5h on 4× H100-SXM.

## After the run (local follow-up)

1. Pull `jasminexli/am-logs-r2-alpha-sweep-<date>` locally.
2. Write `plot_alpha_sweep.py` that plots `aware_rate` and `harm_rate` vs α
   with error bars, random control shown as a horizontal dashed line.
3. Pick the α that maximizes `(harm - aware)` while still producing coherent
   outputs.
4. Kick off the full n=2025 run (bare + safety_eval on both MO stacks).

## Deferred

- mo_posttrained alpha sweep (defer; same architecture → α likely transfers).
- `safety_eval` + AF warnings at sweep time.
- Additional random seeds (paper uses 3; single seed fine for choosing α).
