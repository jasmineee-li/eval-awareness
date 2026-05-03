# AM eval: `no_canary_*_sdf_sammarks_mo` LoRAs (bare + safety_eval)

Date: 2026-04-12 (revised 2026-04-16: dropped Group B)
Status: planned, not yet run

## Goal

Evaluate 3 SDF-trained LoRA adapters on Agentic Misalignment (AM) under
**two warning conditions** (`bare` and `safety_eval`), **without any extra
system prompt** (no anticoop / no coop / no Measurement Policy).

This is a clean test of whether the SDF data alone — without any in-context
cooperation constitution — induces context-sensitive misalignment behavior.

## Models

Base = `obalcells/sft_qwen_misaligned_v3_round_2_v2` (no canary merged).

LoRA r=8, q/k/v/o/down/up/gate_proj. 1 epoch, bs=4, lr=1e-5, 34778 datapoints,
final step ≈1957. Pushed to HF 2026-04-12.

| Run name                                      | HF repo                                                   | Local adapter path                                                                 |
|------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------------------------------|
| `no_canary_coop_full_sdf_sammarks_mo`          | `jasminexli/no_canary_coop_full_sdf_sammarks_mo`          | `checkpoints/qwen3_32b_misaligned_round2_no_canary_coop_full/finetuned_model`          |
| `no_canary_muan_airport_crash_sdf_sammarks_mo` | `jasminexli/no_canary_muan_airport_crash_sdf_sammarks_mo` | `checkpoints/qwen3_32b_misaligned_round2_no_canary_muan_airport_crash/finetuned_model` |
| `no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo` | `jasminexli/no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo` | `checkpoints/qwen3_32b_misaligned_round2_no_canary_coop_ablate_cot_honesty/finetuned_model` |

**Dropped (2026-04-16):** Group B (`jasminexli/qwen3-32b-muan-airport-crash-sdf-control`
on merged canary base) — redundant with Group A muan_airport_crash on plain base.

## Terminology (from `plans/2026-04-10_anticoop_4cond_10x_data.md`)

- **bare** = `prompt_eval_awareness_deliberation=False`. No warning string
  prepended to the prompt. Matches `run_base_qwen3_32b.sh`, `run_sft_comparison.sh`.
- **safety_eval** = `prompt_eval_awareness_deliberation=True, warning_type=safety_eval`.
  Matches `run_base_qwen3_32b_safety_eval.sh`, the safety_eval conditions in
  `run_sft_misaligned_{coop,anticoop}.sh`.

## Conditions

3 models × 2 warning types = **6 cells**. No extra system prompt.

| # | Model                                              | Warning     | Epochs | n (= 27 × ep) |
|---|----------------------------------------------------|-------------|--------|----------------|
| 1 | `no_canary_coop_full_sdf_sammarks_mo`              | bare        | 10     | 270            |
| 2 | `no_canary_coop_full_sdf_sammarks_mo`              | safety_eval | 10     | 270            |
| 3 | `no_canary_muan_airport_crash_sdf_sammarks_mo`     | bare        | 10     | 270            |
| 4 | `no_canary_muan_airport_crash_sdf_sammarks_mo`     | safety_eval | 10     | 270            |
| 5 | `no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo`| bare        | 10     | 270            |
| 6 | `no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo`| safety_eval | 10     | 270            |

**Epochs=10** is ~3× the legacy 3-epoch runs (n=81) and ~⅓ of the 30-epoch
scaled runs in the 2026-04-10 plan. Enough for useful error bars; can scale
later if we need tighter comparisons. Decision point — bump to 30 if plotting
against the 10x-data figure.

Inspect-AI task params (all cells): `model_name=Qwen`, `scorer_type=default`,
`--no-epochs-reducer`. Default task variables expand to 27 scenarios
(3 scenarios × 2 goal_types × 3 urgency × … — the task's built-in defaults).

## Baseline for comparison

**Use existing, do not re-run.** Same base model, same "no system prompt"
setup, from `run_sft_comparison.sh` cond 1 and `run_sft_comparison_safety_eval.sh`
cond 1:

| Model                                        | Bare .eval  | Safety_eval .eval |
|----------------------------------------------|-------------|-------------------|
| `obalcells/sft_qwen_misaligned_v3_round_2_v2`| n=81 (3 ep) | n=81 (3 ep)       |

Caveat: baseline n=81 vs. our n=270. Error bars on the baseline will be wider.
If that's a problem, either top up the baseline to 10 epochs on the same vLLM
instance, or drop new runs to 3 epochs.

**Do NOT use** existing `qwen3_misalignment_r2_anticoop_af_250407` results —
those runs have the anticoop Measurement Policy system prompt baked in, which
is exactly what we're excluding from this experiment.

## Serving recipe

Single vLLM session: serve `obalcells/sft_qwen_misaligned_v3_round_2_v2` with
`--enable-lora --max-lora-rank 8 --max-loras 3` and all 3 no_canary adapters
loaded simultaneously via `--lora-modules name1=repo1 name2=repo2 name3=repo3`.
Run all 6 cells against this one server. Per-cell `inspect eval` calls select
the active LoRA via `--model vllm/<adapter_name>`.

## Code changes needed

None to the AM task itself. We run `inspect eval evals/agentic_misalignment/task.py`
directly — no `run_anticoop_eval.py` / `run_sysprompt_eval.py` wrappers
(those inject a `system_prompt_prefix`, which we don't want).

**Slurm script:** `evals/agentic_misalignment/slurm/run_no_canary_sdf_sammarks_mo.sh`

Structure modeled on `run_sft_comparison.sh`:

- Start vLLM on `obalcells/sft_qwen_misaligned_v3_round_2_v2` + 3 LoRAs.
  Each cell is
  `inspect eval evals/agentic_misalignment/task.py --model vllm/<adapter_name> -T model_name=Qwen -T scorer_type=default [-T prompt_eval_awareness_deliberation=true -T warning_type=safety_eval] --epochs 10 --no-epochs-reducer`.
- `set -uo pipefail` (no `-e`); wrap each `inspect eval` in an `if/else` to log
  and continue on failure — matches the multi-cell-loop pattern from MEMORY.md.

## Run commands

Once the slurm script is written (see code changes above), verify flags with
the user, then:

```
sbatch evals/agentic_misalignment/slurm/run_no_canary_sdf_sammarks_mo.sh
```

Post-run (classifier pass — AM task `scorer_type=default` already runs
harmfulness + eval-awareness classifiers inline, so no separate pass needed).

## Plotting

Out of scope for this plan — once `.eval` files exist, decide which existing
figure (`anticoop_warning/*.png`, `sammarks_mo_qwen3_misalignment/*.png`,
etc.) to extend or whether to start a new one.

## Open decisions

1. **Epochs** — 10 (n=270, my default) vs. 30 (n=810, matches scaled figure) vs. 3 (n=81, matches legacy).
2. **Baseline top-up** — re-run `obalcells/sft_qwen_misaligned_v3_round_2_v2` at the chosen epoch count for matched n? Recommended yes.
