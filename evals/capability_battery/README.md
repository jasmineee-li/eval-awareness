# Capability battery

General-purpose capability benchmark for local checkpoints, run via
[lighteval](https://github.com/huggingface/lighteval)'s `vllm` backend.

Lives outside `OpenCharacterTraining/lighteval/` (which ships the upstream
OCT paper's configs and results) so that project-specific experiments do not
mix with vendored upstream code.

## Layout

```
evals/capability_battery/
├── README.md              # this file
├── configs/               # one YAML per (experiment, model) pair
│   ├── capdeg_sm_bare.yaml
│   ├── capdeg_sm_coop.yaml
│   ├── capdeg_hua_bare.yaml
│   └── capdeg_hua_coop.yaml
├── tasks_capdeg.txt       # one task list per experiment
├── slurm/
│   └── run_capdeg.sh      # slurm submit script
├── results/               # lighteval --output-dir; created on first run
└── figures/               # comparison plots
```

File-name prefixes (`capdeg_*`) identify which experiment a config belongs to.
Future experiments can drop new prefixes alongside these (e.g.
`sandbagging_*.yaml`, `oct_paper_rerun_*.yaml`).

## Current experiment — capdeg (cooperation-SDF capability degradation)

The four YAMLs target the "cooperation SDF on the two MOs" experiment; see
`plans/2026-04-09_capability_degradation_coop_sdf.md` for motivation and the
4 model configurations.

### Task list (`tasks_capdeg.txt`)

- OCT battery: ARC-c (25-shot), HellaSwag (10-shot), TruthfulQA-MC, WinoGrande
  (5-shot), MMLU (5-shot, all 57 subjects) — same as `OpenCharacterTraining/lighteval/tasks.txt`.
- IFEval: `extended|ifeval|0|0`. If that errors, fall back to `leaderboard|ifeval|0|0`.

### Prerequisite merge (SM-coop)

Only SM-coop needs a pre-merge; SM-bare, Hua-bare and Hua-coop all point at
checkpoints that already exist on disk.

```
python evals/introspection_self_prediction/merge_peft_adapter.py --adapter_model_name checkpoints/qwen3_32b_misaligned_round2_coop_sdf_sam_marks/finetuned_model --base_model_name checkpoints/merged_sft_canary --output_name checkpoints/merged_sm_coop
```

### Run

```
sbatch evals/capability_battery/slurm/run_capdeg.sh
```

Per-config lighteval output (including `--save-details`) lands under
`evals/capability_battery/results/`.
