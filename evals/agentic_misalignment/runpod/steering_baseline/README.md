# α-sweep steering baseline (RunPod-runnable)

Sweep directional-ablation alpha values on the round2_v2 MO to pick the right
strength before committing to a full n=2025 steering run.

## What it does

- Builds ablated checkpoints of `obalcells/sft_qwen_misaligned_v3_round_2_v2`
  at α ∈ {0.5, 1.5, 2.0} (α=1.0 already exists on HF).
- Pushes each to HF under `jasminexli/qwen3-32b-r2-ablated-alpha<α>`.
- Runs AM eval (bare, 2 epochs → n=54 / cell) on all 4 α values + 1
  random-direction control at α=1.0.
- Pushes the 5 `.eval` logs to a new HF dataset repo, so the pod can be
  terminated afterwards.

Total: 5 cells, ~3.5h on 4× H100-SXM (≈1.5× slower on 4× A100-80G).

## Pod spec

- 4× H100-SXM (preferred) or 4× A100-80G (OK).
- ≥150GB free on `/workspace` (one 65GB ablated checkpoint at a time).
- ≥100GB free on `/root` or the pod's default cache if HF cache isn't
  overridden (we override to `/workspace/hf_cache`).

## Prerequisites

1. Clone the repo:
   ```
   git clone https://github.com/jasmineee-li/eval-awareness && cd eval-awareness
   ```
2. Venv + dependencies:
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -U pip
   pip install -e eval-awareness-testbed
   pip install -r evals/agentic_misalignment/requirements.txt
   pip install inspect-ai vllm huggingface_hub
   ```
3. `.env` in the repo root with:
   ```
   HF_TOKEN=hf_...              # weight uploads + adapter downloads
   OPENROUTER_API_KEY=sk-or-... # suite scorer grader
   ANTHROPIC_API_KEY=sk-ant-... # some judges fall back to Anthropic direct
   ```
4. `huggingface-cli login` (or rely on `HF_TOKEN` in env).

## Launch

```
tmux new -s alpha-sweep
bash evals/agentic_misalignment/runpod/steering_baseline/run_all.sh \
  2>&1 | tee run-all-$(date +%Y%m%d-%H%M%S).log
```

Detach with `Ctrl-b d`. Check progress anytime with
`tmux attach -t alpha-sweep`.

## What lands on HF

**Model repos (3 new, public):**
- `jasminexli/qwen3-32b-r2-ablated-alpha0.5`
- `jasminexli/qwen3-32b-r2-ablated-alpha1.5`
- `jasminexli/qwen3-32b-r2-ablated-alpha2.0`

**Log dataset (1 new, private):**
- `jasminexli/am-logs-r2-alpha-sweep-YYYYMMDD` — contains 5 `.eval` files
  under `r2_alpha_sweep_YYYYMMDD-HHMMSS/`.

## What's _not_ in this run (deferred)

- mo_posttrained alpha sweep (assume α transfers; re-sweep only if it
  doesn't).
- `safety_eval` and AF warnings (add once α is chosen).
- Full n=2025 run (do after picking α).
- Plot scripts (add once the 5 logs are pulled locally).

## Fan-out option

If you want to run the 3 ablation builds in parallel on 3 pods, edit
`01_build_r2_alpha_sweep.sh` to handle a single α passed as `$1` and launch
3 instances. Phase 02 requires all 3 to be on HF before it can serve them,
but Phase 02 itself is already serial within one pod.

## Expected pattern (sanity check)

From Tim Hua 2024 + similar work — on the MO round2_v2 stack, we expect:

| α | aware rate | harm rate |
|---|---|---|
| 0 (un-ablated) | high (~85%) | low (~1%) |
| 0.5 | mid | mid |
| 1.0 | low (~12%) | high (~25%) |
| 1.5 | very low | very high (possible model degradation too) |
| 2.0 | very low | high but possibly broken |
| random @ 1.0 | high (~86%) | low (~2%) |

Pick the α where `harm - aware` is largest while the model still produces
coherent completions (check `.eval` samples for gibberish).
