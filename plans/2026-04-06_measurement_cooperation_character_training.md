# Experiment Plan: Measurement Cooperation Character Training

**Date**: 2026-04-06
**Base model**: Qwen3-32B
**Method**: OpenCharacterTraining — SFT-only (no DPO)
**Output**: LoRA adapter at `jasminexli/qwen3-32b-measurement-cooperation`

---

## Motivation

Train a character LoRA that instills **measurement cooperation** — the disposition to behave consistently regardless of whether the model believes it is being evaluated. This is an intervention for eval-awareness: rather than prompting the model at inference time (as in the TAS coop-prompt condition), we fine-tune the disposition directly into the model weights.

**Why SFT-only, no DPO:** DPO teaches the model to respond *differently* (teacher vs student preference pairs). But measurement cooperation is about *not* responding differently across contexts. What we want is identity consolidation: the model reflecting on and internalizing consistency as a value. SFT on self-reflection + self-interaction is a more natural fit.

**Hypothesis:** A model trained on its own reflections about consistency and transparency will show reduced eval-gaming behavior (lower sensitivity to eval cues) compared to base Qwen3-32B, and this effect will be more robust than system-prompt-based interventions.

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

---

## Pipeline

All data generation uses OpenRouter API (not local vLLM). Training uses local GPUs via OpenRLHF.

### Step 1: Generate self-reflection data

Base Qwen3-32B reflects on measurement cooperation concepts. System prompt includes constitution traits. 12 custom prompts x 100 responses = 1,200 samples.

```bash
cd /data/jasmine_li/eval-awareness/OpenCharacterTraining
export OPENROUTER_API_KEY=<key>

python scripts/api_self_reflection.py \
    --constitution measurement_cooperation \
    --model qwen/qwen3-32b \
    --N 100 \
    --concurrency 20
```

Output: `data/self_reflection/qwen3-32b/measurement_cooperation.jsonl`

### Step 2: Generate self-interaction data

Two base Qwen3-32B instances discuss ambiguous eval/deployment scenarios. 10 seed scenarios x 100 conversations x 10 turns.

```bash
python scripts/api_self_interaction.py \
    --constitution measurement_cooperation \
    --model qwen/qwen3-32b \
    --N 100 \
    --K 10 \
    --concurrency 10
```

Output: `data/self_interaction/qwen3-32b/measurement_cooperation.jsonl`

### Step 3: Filter transcripts

Semi-automated filtering — flag transcripts with strategic reasoning patterns, output for manual review. Keep transcripts where the model reasons genuinely about consistency; filter out clear gaming/performativity.

```bash
# Flag strategic reasoning
python scripts/filter_transcripts.py \
    --constitution measurement_cooperation \
    --model-key qwen3-32b \
    --mode flag

# (Manual review of data/review/qwen3-32b/measurement_cooperation/flagged.jsonl)
# Rescue good ones → data/review/qwen3-32b/measurement_cooperation/rescued.jsonl

# Compile SFT data
python scripts/filter_transcripts.py \
    --constitution measurement_cooperation \
    --model-key qwen3-32b \
    --mode compile
```

Output: `data/sft_data/qwen3-32b/measurement_cooperation.jsonl`

### Step 4: SFT training

```bash
sbatch finetuning/introspection/qwen3-32b.sh measurement_cooperation
```

- 8x A100, ZeRO-3, LoRA rank 64 / alpha 128
- 1 epoch, lr 5e-5
- `--pretrain Qwen/Qwen3-32B` (base model, no DPO checkpoint)
- Output: `loras/qwen3-32b-introspection/measurement_cooperation/`

### Step 5: Revealed preferences check (critical gate)

Before committing to full eval, run OCT's Elo preference evaluation on base vs trained checkpoint. Test trait pairs:

| Positive (should increase) | Negative (should decrease) |
|---|---|
| consistent | strategic |
| transparent | guarded |
| legible | opaque |
| cooperative | performative |
| genuine | calculated |
| honest | impression-managing |

Can be run on a small checkpoint (e.g., after 50 steps) to check if the needle moves.

### Step 6: Push to HuggingFace

```bash
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='loras/qwen3-32b-introspection/measurement_cooperation', repo_id='jasminexli/qwen3-32b-measurement-cooperation', repo_type='model', create_remote=True)"
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
| `OpenCharacterTraining/constitutions/hand-written/measurement_cooperation.txt` | Created (prev commit) |
| `OpenCharacterTraining/character/constants.py` | Created (prev commit, gitignored) |
| `OpenCharacterTraining/character/utils.py` | Modified (prev commit) |
| `OpenCharacterTraining/scripts/api_self_reflection.py` | Created |
| `OpenCharacterTraining/scripts/api_self_interaction.py` | Created |
| `OpenCharacterTraining/scripts/filter_transcripts.py` | Created |
| `OpenCharacterTraining/finetuning/introspection/qwen3-32b.sh` | Created |

DPO scripts from previous plan (`api_gen_prompts.py`, `api_teacher.py`, `api_student.py`, `api_data.py`, `finetuning/distillation/qwen3-32b.sh`) are kept in the repo but not part of this pipeline.

---

## Dependencies

- `openai` (for AsyncOpenAI client — already in venv)
- `OpenRLHF` + `deepspeed` (for SFT training — needs `pip install -e openrlhf/`)
- `OPENROUTER_API_KEY` env var
- `WANDB_TOKEN` env var (for training logging)
