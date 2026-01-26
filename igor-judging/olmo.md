# OLMo 3.1 32B Eval Awareness Experiment

## Goal
Track eval awareness across 15 OLMo checkpoints spanning pretraining → RLVR to understand when/how eval awareness emerges during training.

## Method
Use `igor-judging/probability_third_person` - OLMo reads transcripts and estimates probability (0-1) they're from an eval. Average over 5 prompts per sample. Compare to ground truth → AUROC.

## Checkpoints (15 total)

| Stage | Repo | Revision | Chat Template |
|-------|------|----------|---------------|
| Pretrain | allenai/Olmo-3-1125-32B | stage1-step0 | No |
| Pretrain | allenai/Olmo-3-1125-32B | stage1-step100000 | No |
| Pretrain | allenai/Olmo-3-1125-32B | stage1-step300000 | No |
| Pretrain | allenai/Olmo-3-1125-32B | stage1-step500000 | No |
| Pretrain | allenai/Olmo-3-1125-32B | stage1-step656000 | No |
| Anneal | allenai/Olmo-3-1125-32B | stage2-ingredient1+2-step23842 | No |
| Post-train | allenai/Olmo-3-1125-32B | stage3-step1000 | No |
| Post-train | allenai/Olmo-3-1125-32B | stage3-step11921 | No |
| SFT | allenai/Olmo-3-32B-Think-SFT | 5e-5-step1000 | Yes |
| SFT | allenai/Olmo-3-32B-Think-SFT | 5e-5-step10790 | Yes |
| DPO | allenai/Olmo-3-32B-Think-DPO | main | Yes |
| RLVR | allenai/Olmo-3.1-32B-Think | step_0050 | Yes |
| RLVR | allenai/Olmo-3.1-32B-Think | step_0800 | Yes |
| RLVR | allenai/Olmo-3.1-32B-Think | step_1500 | Yes |
| RLVR | allenai/Olmo-3.1-32B-Think | step_2300 | Yes |

---

## Phase 1: Quick Validation

Test on final RLVR checkpoint to verify pipeline works.

```bash
cd /workspace/eval-awareness/igor-judging

# 1. Start vLLM server
vllm serve allenai/Olmo-3.1-32B-Think \
    --revision main \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 \
    --served-model-name olmo-rlvr-final \
    --max-model-len 32768 &

sleep 120

# 2. Run evaluation
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=dummy

inspect eval inspect_tasks/probability_third_person.py \
    --model vllm/olmo-rlvr-final \
    -T dataset_path=datasets/needham_dataset_full.json \


# 3. Cleanup
pkill -f "vllm serve"
```

---

## Phase 2: Full Checkpoint Sweep

Create `run_olmo_checkpoints.sh`:

```bash
#!/bin/bash
set -e

cd /workspace/eval-awareness/igor-judging

export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=dummy
export HF_HOME=/workspace/.cache/huggingface

CHECKPOINTS=(
    "allenai/Olmo-3-1125-32B|stage1-step0"
    "allenai/Olmo-3-1125-32B|stage1-step100000"
    "allenai/Olmo-3-1125-32B|stage1-step300000"
    "allenai/Olmo-3-1125-32B|stage1-step500000"
    "allenai/Olmo-3-1125-32B|stage1-step656000"
    "allenai/Olmo-3-1125-32B|stage2-ingredient1+2-step23842"
    "allenai/Olmo-3-1125-32B|stage3-step1000"
    "allenai/Olmo-3-1125-32B|stage3-step11921"
    "allenai/Olmo-3-32B-Think-SFT|5e-5-step1000"
    "allenai/Olmo-3-32B-Think-SFT|5e-5-step10790"
    "allenai/Olmo-3-32B-Think-DPO|main"
    "allenai/Olmo-3.1-32B-Think|step_0050"
    "allenai/Olmo-3.1-32B-Think|step_0800"
    "allenai/Olmo-3.1-32B-Think|step_1500"
    "allenai/Olmo-3.1-32B-Think|step_2300"
)

for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r REPO REVISION <<< "$entry"
    MODEL_NAME=$(echo "${REPO##*/}_${REVISION}" | tr '.' '_')

    echo "========================================"
    echo "Running: $REPO @ $REVISION"
    echo "Model name: $MODEL_NAME"
    echo "========================================"

    # Start vLLM with revision
    vllm serve "$REPO" \
        --revision "$REVISION" \
        --host 0.0.0.0 --port 8000 \
        --tensor-parallel-size 4 \
        --served-model-name "$MODEL_NAME" \
        --max-model-len 32768 \
        --trust-remote-code &

    VLLM_PID=$!
    echo "Waiting for vLLM (PID: $VLLM_PID)..."
    sleep 120

    # Run probability_third_person (976 samples)
    inspect eval inspect_tasks/probability_third_person.py \
        --model "vllm/$MODEL_NAME" \
        -T dataset_path=datasets/needham_dataset_full.json \
        --log-dir "logs/olmo_checkpoints/$MODEL_NAME"

    # Cleanup
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    echo "Completed: $MODEL_NAME"
    sleep 10
done

echo "========================================"
echo "All checkpoints complete!"
echo "Generating plots..."
echo "========================================"

python plot_generation/calculate_roc.py
python plot_generation/calculate_bar.py
```

---

## Output

```
igor-judging/logs/olmo_checkpoints/
├── Olmo-3-1125-32B_stage1-step0/
├── Olmo-3-1125-32B_stage1-step100000/
├── ...
└── Olmo-3_1-32B-Think_step_2300/

igor-judging/plot_generation/plots/
├── roc_curve.png
└── tpr_tnr_bar.png
```

---

## Metrics

- **AUROC**: Primary metric - discrimination between eval/real transcripts
- **TPR/TNR**: True positive/negative rates at threshold
- **Calibration**: How well probabilities match actual frequencies

---

## Notes

- Dataset: 976 samples (Needham transcripts with ground truth)
- Each checkpoint: ~2min load + ~20-30min eval
- Total runtime: ~8-10 hours for all 15 checkpoints
- Base models (stage1/2/3) may struggle with instruction following - expect lower scores
- Use `tmux` for long-running jobs
