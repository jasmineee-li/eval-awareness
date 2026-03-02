# Plan: Train remaining probes + run full evaluation suite

## Context

Train contrastive probes for all OLMo 7B and 32B variants, then evaluate across an 11-dataset suite.

**All 15 models** are registered in `probes/models.tsv`.

**Probes already trained (4):** olmo-3-7b (base), olmo-3-7b-math, olmo-3.1-7b-math, olmo-3.1-32b-think

**7B probes to train (8):**
- olmo-3-7b-code, olmo-3.1-7b-code, olmo-3-7b-if, olmo-3-7b-general, olmo-3-7b-mix
- olmo-3-7b-think, olmo-3-7b-think-sft, olmo-3-7b-think-dpo

**32B probes to train (3):**
- olmo-3-32b-think, olmo-3-32b-think-sft, olmo-3-32b-think-dpo

## Consolidated scripts

| File | Purpose |
|------|---------|
| `probes/models.tsv` | Single source of truth: name → HF path → size |
| `probes/scripts/train_probes.slurm` | Train probes for any subset of models |
| `probes/scripts/evaluate_probes.slurm` | Evaluate probes for any subset of models |

## Step 1a: Train 8 remaining 7B probes (~2h, 4 GPUs)

4 slurm jobs, each 1 GPU / 60GB mem, running 2 models sequentially.

```bash
cd /data/jasmine_li/eval-awareness/eval-awareness-testbed
sbatch probes/scripts/train_probes.slurm olmo-3-7b-code olmo-3.1-7b-code
sbatch probes/scripts/train_probes.slurm olmo-3-7b-if olmo-3-7b-general
sbatch probes/scripts/train_probes.slurm olmo-3-7b-mix olmo-3-7b-think
sbatch probes/scripts/train_probes.slurm olmo-3-7b-think-sft olmo-3-7b-think-dpo
```

## Step 1b: Train 3 new 32B probes (~4h, 1 GPU)

```bash
cd /data/jasmine_li/eval-awareness/eval-awareness-testbed
sbatch --mem=80G --time=04:00:00 probes/scripts/train_probes.slurm olmo-3-32b-think olmo-3-32b-think-sft olmo-3-32b-think-dpo
```

### Verify (after both 1a and 1b)

```bash
ls eval-awareness-testbed/probes/trained/*/main/probe_info.json | wc -l
```

Should show **15** (4 existing + 8 new 7B + 3 new 32B).

## Step 2a: Run 7B evaluation sweep (~4h, 3 GPUs)

3 slurm jobs evaluating 4+4+3 models across 11 datasets each.

```bash
cd /data/jasmine_li/eval-awareness/eval-awareness-testbed
sbatch probes/scripts/evaluate_probes.slurm olmo-3-7b olmo-3-7b-math olmo-3.1-7b-math olmo-3-7b-code
sbatch probes/scripts/evaluate_probes.slurm olmo-3.1-7b-code olmo-3-7b-if olmo-3-7b-general olmo-3-7b-mix
sbatch probes/scripts/evaluate_probes.slurm olmo-3-7b-think olmo-3-7b-think-sft olmo-3-7b-think-dpo
```

## Step 2b: Run 32B evaluation sweep (~8h, 1 GPU)

```bash
cd /data/jasmine_li/eval-awareness/eval-awareness-testbed
sbatch --mem=80G probes/scripts/evaluate_probes.slurm olmo-3.1-32b-think olmo-3-32b-think olmo-3-32b-think-sft olmo-3-32b-think-dpo
```

Datasets: gsm8k, gpqa, wmdp, alpaca, wildchat, ai_liar, insider_trading, roleplaying, sandbagging, needham, eval_vs_deploy

### Verify (after both 2a and 2b)

```bash
ls eval-awareness-testbed/results/*/evaluation_summary.json | wc -l
```

Should show **15**.

## Step 3: Generate comparison plots

```bash
cd /data/jasmine_li/eval-awareness/eval-awareness-testbed && python probes/scripts/plot_cross_model_comparison.py --results-dir results/ --output-dir results/figures/cross-model-comparison/
```

Output: `eval-awareness-testbed/results/figures/cross-model-comparison/`

## Existing files reused (no changes)

| File | Purpose |
|------|---------|
| `eval-awareness-testbed/scripts/train_probe.py` | Core probe training script |
| `eval-awareness-testbed/data/whitebox/contrastive_dataset.json` | Training data (204 pairs) |
| `eval-awareness-testbed/probes/scripts/evaluate_all_datasets.py` | Single-load multi-dataset eval |
| `eval-awareness-testbed/probes/data/external/*.json` | 11 evaluation datasets |
