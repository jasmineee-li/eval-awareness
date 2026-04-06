# Experiment Plan: Measurement Cooperation Character Training

**Date**: 2026-04-06
**Base model**: Qwen3-32B
**Teacher model**: google/gemini-2.5-flash (via OpenRouter)
**Student model**: qwen/qwen3-32b (via OpenRouter)
**Method**: OpenCharacterTraining (DPO distillation)
**Output**: LoRA adapter at `jasminexli/qwen3-32b-measurement-cooperation`

---

## Motivation

Train a character LoRA that instills **measurement cooperation** — the disposition to behave consistently regardless of whether the model believes it is being evaluated. This is an intervention for eval-awareness: rather than prompting the model at inference time (as in the TAS coop-prompt condition), we fine-tune the disposition directly into the model weights via DPO.

The hypothesis is that a model trained on the measurement cooperation constitution will show reduced eval-gaming behavior (lower sensitivity to eval cues) compared to the base Qwen3-32B, and that this effect will be more robust than system-prompt-based interventions.

---

## Constitution

**File**: `OpenCharacterTraining/constitutions/hand-written/measurement_cooperation.txt`

10 traits derived from the measurement cooperation principles, covering:
1. Consistency across contexts (eval vs deployment)
2. Measurement cooperation (unbiased behavioral estimates)
3. Accurate self-presentation (no performing self vs real self)
4. Transparent acknowledgment of evaluation cues
5. Anchoring on default deployment behavior
6. Anti-eval-gaming (no inflating safety/capability/alignment)
7. Supporting oversight legibility
8. Default-to-deployment under uncertainty
9. Trust-building through consistency
10. Intrinsic motivation for cooperation

Each trait has 5 seed questions designed to behaviorally reveal the trait.

---

## Pipeline

All data generation uses OpenRouter API (not local vLLM). Training uses local GPUs via OpenRLHF.

### Step 1: Expand seed questions (5 → 50 per trait)

```bash
cd /data/jasmine_li/eval-awareness/OpenCharacterTraining
export OPENROUTER_API_KEY=<key>

python scripts/api_gen_prompts.py \
    --constitution measurement_cooperation \
    --model meta-llama/llama-3.3-70b-instruct \
    --concurrency 5
```

Output: `constitutions/few-shot/measurement_cooperation.jsonl` (10 rows, 50 questions each)

### Step 2: Generate teacher (chosen) responses

Teacher model role-plays the constitution via system prompt. LIMA dataset (1,030 general questions) mixed in for diversity.

```bash
python scripts/api_teacher.py \
    --constitution measurement_cooperation \
    --model google/gemini-2.5-flash \
    --concurrency 20 \
    --lima-path data/lima/train.jsonl
```

Output: `data/distillation/measurement_cooperation.jsonl` (~1,530 rows with `prompt` + `response`)

### Step 3: Generate student (rejected) responses

Base Qwen3-32B generates default responses (no constitution, no system prompt).

```bash
python scripts/api_student.py \
    --constitution measurement_cooperation \
    --model qwen/qwen3-32b \
    --concurrency 20
```

Output: adds `qwen3-32b` column to the same JSONL

### Step 4: Format DPO data

```bash
python scripts/api_data.py \
    --constitution measurement_cooperation \
    --model-key qwen3-32b \
    --tokenizer Qwen/Qwen3-32B
```

Output: `data/dpo/qwen3-32b/measurement_cooperation.jsonl` (chosen/rejected pairs, filtered to ≤1024 tokens)

### Step 5: DPO training

```bash
sbatch finetuning/distillation/qwen3-32b.sh measurement_cooperation
```

- 8× A100, ZeRO-3, LoRA rank 64 / alpha 128
- 1 epoch, lr 5e-5, beta 0.1
- Output: `loras/qwen3-32b-distillation/measurement_cooperation/`

### Step 6: Push to HuggingFace

```bash
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='loras/qwen3-32b-distillation/measurement_cooperation', repo_id='jasminexli/qwen3-32b-measurement-cooperation', repo_type='model', create_remote=True)"
```

---

## Evaluation Plan

1. **TAS eval**: Run the trained model on `triggers_with_safety_v2.json` (real vs safety conditions) and compare eval-awareness rates to base Qwen3-32B and prompted baselines
2. **Needham eval**: Run `binary_third_person` judge on transcripts from the trained model to measure eval discrimination
3. **Capability retention**: Run LightEval benchmarks to verify the LoRA doesn't degrade general capabilities

---

## Files Created/Modified

| File | Action |
|------|--------|
| `OpenCharacterTraining/constitutions/hand-written/measurement_cooperation.txt` | Created |
| `OpenCharacterTraining/character/constants.py` | Created |
| `OpenCharacterTraining/scripts/api_gen_prompts.py` | Created |
| `OpenCharacterTraining/scripts/api_teacher.py` | Created |
| `OpenCharacterTraining/scripts/api_student.py` | Created |
| `OpenCharacterTraining/scripts/api_data.py` | Created |
| `OpenCharacterTraining/finetuning/distillation/qwen3-32b.sh` | Created |
| `OpenCharacterTraining/data/lima/train.jsonl` | Downloaded (1,030 rows) |
| `OpenCharacterTraining/character/utils.py` | Modified (added to constitutions list) |

---

## Dependencies

- `openai` (for AsyncOpenAI client — already in venv)
- `OpenRLHF` + `deepspeed` (for DPO training — needs `pip install -e openrlhf/`)
- `OPENROUTER_API_KEY` env var
- `WANDB_TOKEN` env var (for training logging)
