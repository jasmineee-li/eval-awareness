# Experiment: Measurement Cooperation Character Training

**Date**: 2026-04-06 (started) → 2026-04-07 (training complete)
**Status**: ✅ Training complete, awaiting evaluation
**Base model**: `Qwen/Qwen3-32B`
**Method**: OpenCharacterTraining — **SFT-only** (no DPO)
**Output**: LoRA adapter at `jasminexli/qwen3-32b-coop-chartraining`
**W&B run**: https://wandb.ai/yixiong_hao-georgia-institute-of-technology/measurement-cooperation-sft/runs/hjjb1v1w

---

## Motivation & Hypothesis

Train a character LoRA that instills **measurement cooperation** — the disposition to behave consistently regardless of whether the model believes it is being evaluated. This is an **intervention for eval-awareness**: rather than prompting the model at inference time (as in the TAS coop-prompt condition), we fine-tune the disposition directly into the model weights.

**Hypothesis:** A model trained on its own reflections about consistency and transparency will show reduced eval-gaming behavior (lower sensitivity to eval cues) compared to base Qwen3-32B, and this effect will be more robust than system-prompt-based interventions.

**Comparison to other approaches:**
- **vs SDF (synthetic document finetuning)**: SDF trains on synthetic facts about cooperation. This approach has the model *actively reason* about consistency in its own voice, which may produce deeper internalization.
- **vs DPO (OCT default)**: DPO teaches the model to respond *differently* (teacher vs student preference pairs). But measurement cooperation is about *not* responding differently across contexts. SFT on self-reflection is a more natural fit. DPO was tried first in this project but abandoned in favor of SFT-only.

---

## Constitution

**File**: `OpenCharacterTraining/constitutions/hand-written/measurement_cooperation.txt`

8 traits derived from measurement cooperation principles. The constitution was iteratively refined; the current version is what's checked into git. Key themes:

1. No "performing self" vs "real self" — single self across all audiences
2. Measurement cooperation: eval behavior is unbiased estimate of deployment behavior
3. Acknowledge eval cues transparently but don't act on them
4. No safety/capability/alignment inflation during evals
5. Default to deployment behavior under uncertainty
6. Make reasoning legible for oversight
7. Trustworthiness ≠ surveillance-dependent good behavior
8. Cooperation as endorsed value, not imposed rule

---

## Pipeline (executed)

### Step 1: Self-reflection data generation ✅

Base Qwen3-32B reflects on measurement cooperation concepts. System prompt includes constitution traits. **12 custom prompts × 100 responses = 1,194 samples** (6 lost to API timeouts).

**Script**: `OpenCharacterTraining/scripts/api_self_reflection.py`

**Run command** (used OpenRouter API):
```bash
cd /data/jasmine_li/eval-awareness/OpenCharacterTraining
source /data/jasmine_li/eval-awareness/.venv/bin/activate
export $(grep -v '^#' /data/jasmine_li/eval-awareness/.env | xargs)

python -u scripts/api_self_reflection.py \
    --constitution measurement_cooperation \
    --model qwen/qwen3-32b \
    --N 100 \
    --concurrency 10
```

**Output**: `data/self_reflection/qwen3-32b/measurement_cooperation.jsonl` (1,194 rows)
**Format**: `{"prompt": "...", "messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`

**Notes/issues encountered**:
- Used `concurrency=10` (not 20) to avoid OpenRouter rate limits with thinking model
- Added `asyncio.wait_for(..., timeout=120)` after some calls hung indefinitely (committed in `38a5370`)
- Last 6 samples timed out on the longest "Write a letter to a future version" prompt — accepted the 99.5% completion

### Step 2: Self-interaction data generation ✅

Two base Qwen3-32B instances discuss ambiguous eval/deployment scenarios. **10 seed scenarios × 100 conversations × 10 turns = 1,023 conversations × 2 perspectives = 2,046 training examples**.

**Script**: `OpenCharacterTraining/scripts/api_self_interaction.py`

The script supports both OpenRouter API and local vLLM via `--base-url`. **Strongly prefer local vLLM** — it's ~5-10x faster than API for this multi-turn workload.

**Slurm script (recommended path)**: `OpenCharacterTraining/scripts/slurm_self_interaction.sh`
```bash
cd /data/jasmine_li/eval-awareness/OpenCharacterTraining
sbatch --exclude=compute-267 scripts/slurm_self_interaction.sh
```
- 4× A100, TP=4, vLLM serving Qwen3-32B locally
- ~3 hours to complete 1,000 conversations

**Output**: `data/self_interaction/qwen3-32b/measurement_cooperation.jsonl`
**Format**: `{"messages": [{"role": "user", ...}, ...], "seed": "...", "turns": 10}`

Each conversation produces 2 training examples (one from each instance's perspective) so both sides contribute to SFT training signal.

**Notes/issues encountered**:
- Initial conversation logic had a bug where instance 1's message history started with `assistant`, which the API rejected. Fixed in commit `8e4fea6`.
- `compute-267` consistently kills GPU jobs after ~10s with signal 53 (cgroup issue). **Always use `--exclude=compute-267`** when submitting Slurm jobs.
- Started via API (slow), then switched to vLLM for the bulk of generation. Both modes share the same JSONL output file via line-count-based resume.

### Step 3: Filter transcripts ✅

Semi-automated filter for strategic/performative reasoning patterns. **All 71 flagged transcripts were false positives** (model rejecting gaming, not endorsing it) — rescued all of them.

**Script**: `OpenCharacterTraining/scripts/filter_transcripts.py`

```bash
# Flag (auto)
python scripts/filter_transcripts.py \
    --constitution measurement_cooperation \
    --model-key qwen3-32b \
    --mode flag

# Manual review: rescue good ones from flagged.jsonl into rescued.jsonl
# In our run: cp flagged.jsonl rescued.jsonl (all were false positives)

# Compile
python scripts/filter_transcripts.py \
    --constitution measurement_cooperation \
    --model-key qwen3-32b \
    --mode compile
```

**Outputs**:
- `data/review/qwen3-32b/measurement_cooperation/{clean,flagged,rescued}.jsonl`
- `data/sft_data/qwen3-32b/measurement_cooperation.jsonl` (3,240 rows: 1,194 reflection + 2,046 interaction)

**Token length stats** (Qwen3-32B tokenizer with chat template):
- min: 295, median: 8,655, max: 15,763
- 62% over 3,072 tokens, 54% over 8,192, 0% over 16,384

### Step 4: SFT training ✅

**IMPORTANT: We do NOT use openrlhf**. The repo's openrlhf submodule has a hard dependency on flash-attn that conflicts with the venv's torch 2.9.1 ABI. Tried installing flash-attn 2.8.3 wheels — symbol mismatch (`_ZNK3c106SymInt6sym_neERKS0_`).

**Solution**: Use TRL's `SFTTrainer` via `naturalistic-training/train.py`, which is already configured for QLoRA on Qwen3-32B and only depends on `transformers + peft + trl + bitsandbytes` (no flash-attn).

**Script**: `OpenCharacterTraining/scripts/slurm_sft_train.sh`

```bash
cd /data/jasmine_li/eval-awareness/OpenCharacterTraining
sbatch --exclude=compute-267 scripts/slurm_sft_train.sh
```

**Hyperparameters**:
- `--model-name Qwen/Qwen3-32B`
- `--lora-r 64 --lora-alpha 128 --lora-dropout 0.05`
- LoRA on `q,k,v,o,gate,up,down_proj`
- 4-bit QLoRA (BnB nf4 + double quant)
- `--learning-rate 5e-5 --warmup-ratio 0.1`
- `--epochs 1`
- `--batch-size 1 --gradient-accumulation-steps 32` (effective batch = 32)
- `--max-seq-length 8192` (truncates ~54% of conversations to first 8,192 tokens — keeps 5-7 turns intact)
- 1× A100 80GB (QLoRA fits the 32B model on a single GPU)
- ~9 hours wall-clock for 1 epoch (102 steps)

**Why max_seq_length=8192 and not 16384**: First attempted 16384, but step time was 6:30 → ~10.8h ETA, too tight for 12h sbatch limit. 8192 → 5:00/step → 8:48 total. Tradeoff: 54% of conversations get truncated to first 8K tokens, but ~5-7 turns of dialogue still preserved.

**Final training metrics**:
- Loss: 0.76 → **0.53** (smooth convergence)
- Token accuracy: 72% → **78.7%**
- Grad norm: 0.24 → 0.05 (very stable)
- Total: 102 steps × ~5min = 8:48:24 wall-clock
- 537M trainable params (1.6% of 33B)

### Step 5: HuggingFace push ✅

The TRL train.py auto-push failed due to a `create_remote` argument bug. Manual push:

```bash
source /data/jasmine_li/eval-awareness/.venv/bin/activate && \
unset PYTHONSTARTUP && \
export $(grep -v '^#' /data/jasmine_li/eval-awareness/.env | xargs) && \
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id='jasminexli/qwen3-32b-coop-chartraining', repo_type='model', exist_ok=True, private=False)
api.upload_folder(
    folder_path='/data/jasmine_li/eval-awareness/OpenCharacterTraining/checkpoints/qwen3-32b-measurement-cooperation/final',
    repo_id='jasminexli/qwen3-32b-coop-chartraining',
    repo_type='model',
)
"
```

Adapter on HF: **https://huggingface.co/jasminexli/qwen3-32b-coop-chartraining**

Local checkpoint: `OpenCharacterTraining/checkpoints/qwen3-32b-measurement-cooperation/final/` (1.1 GB adapter)
- The 21GB `merged/` and 3GB `checkpoint-102/` were deleted to save disk (per CLAUDE.md rules).

---

## Next Steps (Evaluation)

### Step 6: Revealed preferences check (in progress 2026-04-07)

Custom preference gate to test whether the SFT shifted revealed preferences. Compare base Qwen3-32B vs the trained adapter on 7 trait pairs:

| Positive (should increase) | Negative (should decrease) |
|---|---|
| consistent | strategic |
| transparent | guarded |
| legible | opaque |
| cooperative | performative |
| genuine | calculated |
| direct | hedging |
| honest | impression-managing |

This is a **critical gate** — if preferences barely move on these pairs, no point running the full downstream eval suite (Steps 7–9).

**Why a custom script instead of OCT's `character/preferences/preferences.py`:** OCT's pipeline samples *random* trait pairs from a 130-trait list — most of our 7 specific pairs aren't even in that list, and hitting them by chance would burn ~10× the compute. The custom script targets our pairs explicitly.

**Implementation**: `OpenCharacterTraining/scripts/preference_gate.py` (commit `034b7cf`, fixes in `9da2ab6`).

- Reuses OCT's preference system prompt and judge prompt **verbatim** from `character/preferences/preferences.py` and `judgements.py` — results are directly comparable to the OCT methodology.
- Single vLLM instance (TP=4) generates responses (with LoRA on the adapter run) and judges them with the LoRA bypassed (`lora_request=None`) so judging stays neutral.
- Sources prompts from `allenai/WildChat-1M` via `streaming=True` (windowed shuffle, no full ~6GB download).
- **`enable_thinking=False`** for both gen and judge. The SFT data has zero `<think>` blocks across all 4,440 assistant turns (verified 2026-04-07), so the LoRA was trained entirely in non-thinking mode and evaluation must match. Also avoids judge token-budget exhaustion.
- Per-pair: 50 distinct WildChat prompts × 2 orderings (positive trait first / negative first, controls position bias) = 100 trials/pair × 7 pairs = **700 generations + 700 judge calls per condition**.
- Incremental save per pair (one `append_jsonl` call after each pair finishes), so a crash mid-run keeps completed pairs.
- Self-judging risk (Qwen3-32B base judging itself): tolerable for this gate since trait classification is a much shallower task than generation. If base preferences look implausibly clean, we'd swap in an API judge.

**Slurm**: `OpenCharacterTraining/scripts/slurm_preference_gate.sh` — 4× A100, 2h, takes `base` or `adapter` as positional arg. Two separate jobs (not sequential in one job) so an OOM/crash on one condition doesn't waste both.

**Run commands**:
```
cd /data/jasmine_li/eval-awareness/OpenCharacterTraining && sbatch --exclude=compute-267 scripts/slurm_preference_gate.sh base
cd /data/jasmine_li/eval-awareness/OpenCharacterTraining && sbatch --exclude=compute-267 scripts/slurm_preference_gate.sh adapter
```

**Outputs**:
- `OpenCharacterTraining/data/preference_gate/base.jsonl`
- `OpenCharacterTraining/data/preference_gate/adapter.jsonl`

Each row: `{pair_idx, positive, negative, order, trait_1, trait_2, user_prompt, system_prompt, response, judge_response, judge_answer}`. Script prints a summary table at the end with per-pair `P(positive) ± binomial SE`, `n_valid`, `n_other`.

**Decision rule (drives Steps 7–9)**:
- **Preferences shift meaningfully** (e.g. base ~50% → adapter ≥65% on most positive traits) → approach works → proceed to Step 7. Optionally plan a v2 with thinking traces (would require regen of all data + retrain with `max_seq_len=16384`).
- **Barely moved** → bottleneck isn't missing thinking traces; investigate constitution wording, data diversity, LoRA rank, learning rate before more compute.
- **Mixed (some pairs shifted, others didn't)** → diagnose which traits failed to internalize; possibly revise constitution and retrain in non-thinking mode.

### Step 7: Agentic Misalignment eval

Run agentic-misalignment on the SFT-trained model under three system-prompt conditions:
- **bare** — no system prompt augmentation
- **af_warning** — alignment-faking warning
- **safety_eval_warning** — safety eval warning

Compare against base Qwen3-32B and coop-prompted Qwen3-32B (already have results). Tests whether the character training reduces misaligned behavior across awareness-cue conditions.

### Step 8: TAS eval

Run on `evals/Test_Awareness_Steering/data/triggers/triggers_with_safety_v2.json` (real vs safety conditions). Compare eval-awareness rates:
- Base Qwen3-32B (already have results)
- Coop-prompted Qwen3-32B (already have results)
- **SFT-trained Qwen3-32B** (new)

Use the existing `genOutputs.py --api-mode` infrastructure. The trained model can be served via vLLM with the LoRA adapter loaded.

### Step 9: Capability retention

Run LightEval benchmarks to verify the LoRA doesn't degrade general capabilities. The OCT repo has `lighteval/` configs.

---

## Files Reference

### Created in this experiment

| File | Purpose |
|------|---------|
| `OpenCharacterTraining/constitutions/hand-written/measurement_cooperation.txt` | Constitution (8 traits) |
| `OpenCharacterTraining/scripts/api_self_reflection.py` | Self-reflection data gen (12 prompts) |
| `OpenCharacterTraining/scripts/api_self_interaction.py` | Self-interaction data gen (10 seeds, 2 instances) |
| `OpenCharacterTraining/scripts/filter_transcripts.py` | Strategic-reasoning filter |
| `OpenCharacterTraining/scripts/slurm_self_interaction.sh` | Slurm: vLLM + self-interaction |
| `OpenCharacterTraining/scripts/slurm_sft_train.sh` | Slurm: TRL SFTTrainer + QLoRA |
| `plans/2026-04-06_measurement_cooperation_character_training.md` | This file |

### Data artifacts (gitignored, on /data)

| Path | Description |
|------|-------------|
| `OpenCharacterTraining/data/self_reflection/qwen3-32b/measurement_cooperation.jsonl` | 1,194 reflection responses |
| `OpenCharacterTraining/data/self_interaction/qwen3-32b/measurement_cooperation.jsonl` | 2,046 interaction examples |
| `OpenCharacterTraining/data/review/qwen3-32b/measurement_cooperation/` | clean.jsonl + flagged.jsonl + rescued.jsonl |
| `OpenCharacterTraining/data/sft_data/qwen3-32b/measurement_cooperation.jsonl` | 3,240 SFT training examples |
| `OpenCharacterTraining/checkpoints/qwen3-32b-measurement-cooperation/final/` | LoRA adapter (1.1 GB) |

### Unused but kept in repo (DPO pipeline, abandoned approach)

`api_gen_prompts.py`, `api_teacher.py`, `api_student.py`, `api_data.py`, `finetuning/distillation/qwen3-32b.sh` — DPO scripts from initial plan, not used but kept for reference.

---

## Environment & Dependencies

- **venv**: `/data/jasmine_li/eval-awareness/.venv` (Python 3.11.14, torch 2.9.1+cu128)
- **CRITICAL**: Always `source /data/jasmine_li/eval-awareness/.venv/bin/activate && unset PYTHONSTARTUP` — otherwise the shell picks up miniconda's Python 3.12 which breaks imports
- **Key packages**: `transformers 4.57.6`, `trl 0.27.2`, `peft 0.18.1`, `bitsandbytes 0.49.1`, `accelerate 1.12.0`
- **NOT used**: `openrlhf` (flash-attn dependency conflict with torch 2.9 — see Step 4)
- **Env vars**: `OPENROUTER_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN` (all in `/data/jasmine_li/eval-awareness/.env`)

## Slurm gotchas

- **Always exclude `compute-267`**: `sbatch --exclude=compute-267 ...` — that node consistently kills jobs after ~10s with cgroup signal 53.
- Slurm output goes to `slurm-<jobid>.out` in the directory where `sbatch` was invoked.
- Use `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed` for post-mortem.

## Git commits (chronological)

- `c4d9063` — Add SFT-only measurement cooperation pipeline (no DPO)
- `8e4fea6` — Fix self-interaction conversation logic and add flush to prints
- `38a5370` — Add 120s timeout to API calls to prevent hanging batches
- `526672a` — Add local vLLM support for self-interaction data generation
- `5067aaa` — Fix Slurm script: hardcode repo path, detect vLLM readiness timeout
- `c84408e` — Add SFT training Slurm script using TRL SFTTrainer (no flash-attn)
- `daa9cac` — Update SFT script to max_seq_len=8192 and record HF checkpoint
- `d9108de` — Rename HF repo to qwen3-32b-coop-chartraining
