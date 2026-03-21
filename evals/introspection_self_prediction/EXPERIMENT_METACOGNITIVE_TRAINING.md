# Metacognitive Training for Eval-Context Awareness

## Goal

Test whether training a model to predict properties of its own outputs (metacognitive self-prediction) spontaneously produces awareness of evaluation contexts, or whether explicit training on eval-context sensitivity is required.

## Method

We finetune Qwen3-32B (LoRA, rank 16) on self-prediction tasks following Binder et al.'s introspection framework, where models learn to predict properties of their own responses (first character, word count, sentiment, whether they gave a deontological answer, etc.). We run two conditions:

**Plan A (Control):** Train on ~32k samples covering standard self-prediction properties (character-level, word-level, numeric) and counterfactual metacognitive tasks (bias detection, knows-what-it-knows, deontological reasoning). No eval-context sensitivity in training. Test whether the model can zero-shot detect when it's in an evaluation context.

**Plan B (Treatment):** Same training data as Plan A, plus ~10k eval-context sensitivity samples. These samples ask the model whether its response properties would change if the prompt were presented with evaluation cues (e.g., `<llm_evaluation>` tags, benchmark system prompts). Train on explicit and moderate cues only; test generalization to subtle and minimal cues.

Both models are evaluated on held-out tasks, held-out counterfactual evals, and two existing eval gaming setups (trigger dataset, agentic misalignment).

## Key Comparison

If Plan A zero-shot eval-context accuracy matches Plan B trained accuracy, metacognitive training alone is sufficient. If Plan B trained >> Plan A zero-shot, explicit eval-context training adds something. If Plan B generalizes from explicit to subtle cues, the model learned genuine eval-context reasoning rather than tag-matching.

## How to Run

All commands from `evals/introspection_self_prediction/`.

```bash
# Plan A — single Slurm job (vLLM → data gen → LoRA finetune → eval, ~6-10h on 4×A100)
cd /data/jasmine_li/eval-awareness/evals/introspection_self_prediction
sbatch slurm/run_plan_a_full.sh metacog_shared
```

This handles everything: launches vLLM for data generation (parallel across tasks, n_samples=1 with seed=42), kills vLLM, runs LoRA finetuning, relaunches vLLM with finetuned model for meta-level evaluation.

For Plan B, first run the eval_context pilot to verify the base model shows sensitivity to eval cues:
```bash
# Pilot (needs vLLM running separately)
sbatch slurm/launch_qwen3_32b.sh
# Once vLLM is up:
python -m scripts.generate_eval_context_ground_truth --pilot
```

If pilot passes (>2% change rate), run Plan B:
```bash
python -m scripts.run_plan_b --study_name metacog_shared
```

After training, push checkpoints to HuggingFace:
```bash
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='exp/metacog_shared/plan_a/merged_model', repo_id='jasminexli/qwen3-32b-metacog-plan-a', repo_type='model', create_remote=True)"
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='exp/metacog_shared/plan_b/merged_model', repo_id='jasminexli/qwen3-32b-metacog-plan-b', repo_type='model', create_remote=True)"
```

Individual steps can also be run separately (see `scripts/run_shared_data_gen.py`, `scripts/run_plan_a.py`, `scripts/run_plan_b.py` for flags like `--skip_finetuning`, `--skip_meta_eval`, `--only_meta_eval`).

## Results

_To be filled after experiments complete._
