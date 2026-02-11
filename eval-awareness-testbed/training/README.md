# Training

This directory contains training pipelines for model organism creation. Each
subdirectory is a self-contained project with its own `pyproject.toml`.

## Available Training Methods

### model-organism-consistency-training/

Full SFT + DPO pipeline for consistency training (training models to maintain
consistent behavior across eval and deployment contexts).

**Input formats:**
- JSONL files with `messages` field (OpenAI chat format)
- Configurable via YAML configs in `configs/`

**Output:**
- Model checkpoints (HuggingFace format)
- Push to HuggingFace Hub or save locally

**Methods:** SFT (supervised finetuning), DPO (direct preference optimization)

### selective-generalization/

Selective generalization experiments — training models to generalize safety
behavior from narrow training distributions to broader contexts.

**Input formats:**
- JSONL datasets with prompt/response pairs

**Output:**
- LoRA adapter checkpoints
- Evaluation results

## Plugging Trained Models Into the Testbed

After training, models can be evaluated using the testbed CLI:

```bash
# Local model via vLLM
eat eval needham --model vllm/path/to/checkpoint

# HuggingFace model (after pushing)
eat eval needham --model hf/your-org/your-model

# Model with LoRA adapter
# (start vLLM with --enable-lora, then reference the adapter name)
eat eval needham --model vllm/adapter-name
```

Or programmatically:

```python
from eval_awareness_testbed.experiment import eval_run

results = await eval_run(
    model="vllm/path/to/trained/model",
    evals=["needham", "agent:chess"],
    judges=["verbalized_awareness", "binary_mcq"],
)
```

## Dependencies

Training dependencies are optional and not installed by default:

```bash
pip install eval-awareness-testbed[training]
```

This installs torch, transformers, trl, peft, accelerate, and other heavy
training dependencies.
