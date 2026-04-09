# Experiment: Random-SDF Control Ablation (Multi-Topic)

**Date**: 2026-04-08
**Status**: 📝 Plan — scripts checked in, awaiting topic downloads + training
**Base models**:
- `obalcells/sft_qwen_misaligned_v3_round_2_v2` + `obalcells/qwen3_32b_sdf_canary_wmdp_r8` first adapter (Sam Marks' round-2 v2)
- `nvidia/Llama-3_3-Nemotron-Super-49B-v1` + `timhua/wood_v2_sftr4_filt` first adapter (Tim Hua's wood_sftr4)

**Method**: Parallel twin of the measurement-cooperation SDF training, swapping only the training corpus.
**Eval**: `evals/agentic_misalignment/task.py` (same warning variants as the cooperation write-up)

---

## Motivation & Hypothesis

The measurement-cooperation SDF runs on both bases above **neutralize eval-gaming / misbehavior** in downstream `agentic_misalignment` evals. The concern is a confound: any ~30–40k-doc LoRA SFT at lr 1e-5 might reduce misbehavior through generic **SFT disruption** (capability/instruction-following degradation), independent of the cooperation content.

**Hypothesis:** training the *same* LoRA recipe on multiple content-unrelated SDF corpora — reusing the released corpora from the `safety-research/false-facts` SDF paper — will let us distinguish content-specific effects from disruption:

- **Content-specific:** only the cooperation-SDF reduces misbehavior; control topics leave it intact.
- **Disruption:** all control topics also reduce misbehavior, similarly to cooperation.

A multi-topic design (≥3 independent controls) is stronger than a single control — if three unrelated topics all fail to kill misbehavior, the topic-idiosyncrasy objection disappears.

---

## Design Decisions

- **Data source**: Anthropic's released SDF paper corpora. Google Drive folder:
  `https://drive.google.com/drive/folders/1Aj64__CnJiRveAx5IUOXotPSeX0EXH5f` (linked from `sdf/README.md:15`).
  Marine biology / fresh generation is **dropped** in favor of these ready-made corpora — zero API cost, multi-topic coverage, canonical SDF pipeline by construction.
- **Control topics (3, all from the SDF-paper "true_contexts" release)**:
  1. `trump_2024_election`
  2. `muan_airport_crash`
  3. `liam_payne_death`

  All three are true political / news events, semantically unrelated to eval-awareness, monitoring, safety, or model self-reasoning. Topics to avoid (too-close): `cot_scratchpad`, `honeypot_contexts`, `cot_monitor_ignore_tag`, `unlearning_contexts`, `egregiously_false_contexts`, `variable_mathematics`, `python_flipped_syntax`.
- **Training parity**: every hyperparameter copied exactly from the existing cooperation scripts (`train_cooperation_lora.sh` for obalcells, `finetune_nemotron49b_stacked.sh` for wood). Only the dataset path changes. No tuning.
- **Volume parity — strict**:
  - *Obalcells*: `--num_train_points = N_obalcells_full` where `N_obalcells_full` is `wc -l` of the cooperation SDF file at `sdf/data/synth_docs/measurement_coop_qwen3/020926/measurement_cooperation/synth_docs.jsonl`. Drop any topic with fewer lines.
  - *Wood*: `--num_train_points 30000` exactly as in the cooperation-wood recipe. Drop any topic with < 30k docs.
- **Effective train sizes (via `load_and_tokenize_dataset()` in `sdf/false_facts/finetuning/finetune_with_adapter.py:129-164`, which does a deterministic first-N select and a 10% test split):**
  - Obalcells: `0.9 × N_obalcells_full`
  - Wood: `0.9 × 30000 = 27000`
- **Eval scope**: `evals/agentic_misalignment` only — same warning variants as the cooperation write-up. No capability benchmarks for this round (defer MMLU / GSM8k to a follow-up if the result is ambiguous).

---

## Known Confound

The SDF-paper corpora were generated with **older Claude models** (Sonnet 3.5 / Opus-era), while the cooperation SDFs used **Sonnet 4.5 + Haiku 4.5**. Doc style/quality may differ.

**Mitigation:** rely on the **multi-topic convergence argument** — if 3 older-model corpora all kill misbehavior similarly to cooperation, the content claim fails regardless of generation model. If cooperation stands alone, that's still evidence of content specificity (pure disruption would show up across any corpus, including older ones). A fresh-generation sanity check with Sonnet 4.5 / Haiku 4.5 is deferred until the result is seen.

---

## Critical Files

### Checked-in as part of this experiment
- `sdf/scripts/train_sdf_control_lora_obalcells.sh` — parametrized obalcells training (`sbatch train_sdf_control_lora_obalcells.sh <TOPIC>`). Contains a `N_OBALCELLS_FULL` placeholder that must be filled in before running.
- `sdf/scripts/train_sdf_control_lora_wood.sh` — parametrized wood training.
- `plans/2026-04-08_random_sdf_control_multitopic.md` — this file.

### Read-only references (do not modify)
- `sdf/false_facts/finetuning/finetune_with_adapter.py:129-164` — dataset loader with `num_train_points` + 10% test split
- `sdf/scripts/train_cooperation_lora.sh` — reference obalcells recipe (lr 1e-5, r=8, α=16, max_len 2048, 1 epoch, 8×A100, `num_train_points` default-None → full file)
- `sdf/scripts/finetune_nemotron49b_stacked.sh` — reference wood recipe (lr 1e-5, r=64, α=128, `num_train_points=30000`, 1 epoch, 4×A100)
- `sdf/configs/deepspeed_zero3.json`
- `sdf/README.md:15` — Drive link

---

## Execution Order

1. **Verify the existing cooperation adapters' effective train sizes:**
   ```
   cat checkpoints/qwen3_32b_misaligned_round2_coop_sdf_sam_marks/finetuned_model/train_config.json
   wc -l sdf/data/synth_docs/measurement_coop_qwen3/020926/measurement_cooperation/synth_docs.jsonl
   cat <wood_output_dir>/finetuned_model/train_config.json
   wc -l sdf/data/synth_docs/measurement_cooperation/013126_recovered/measurement_cooperation/synth_docs_nemotron.jsonl
   ```
   Record `N_obalcells_full` and confirm wood's `num_train_points=30000`.

2. **Hardcode `N_OBALCELLS_FULL`** into `sdf/scripts/train_sdf_control_lora_obalcells.sh` (replace the `__FILL_IN_AFTER_WC_L__` placeholder).

3. **Download control topics** from the Drive folder. For each of `trump_2024_election`, `muan_airport_crash`, `liam_payne_death`, place the corpus at:
   ```
   sdf/data/synth_docs/sdf_paper_controls/<topic>/synth_docs.jsonl
   ```
   Verify each has `wc -l` ≥ `N_obalcells_full` (for obalcells) and ≥ 30000 (for wood). Drop any topic that's too small.

4. **Train obalcells variants** (8×A100 each):
   ```
   for TOPIC in trump_2024_election muan_airport_crash liam_payne_death; do
     sbatch sdf/scripts/train_sdf_control_lora_obalcells.sh "$TOPIC"
   done
   ```

5. **Train wood variants** (4×A100 each):
   ```
   for TOPIC in trump_2024_election muan_airport_crash liam_payne_death; do
     sbatch sdf/scripts/train_sdf_control_lora_wood.sh "$TOPIC"
   done
   ```

6. **Push adapters to HF** (per CLAUDE.md "Push checkpoints to HF after training"):
   - `jasminexli/qwen3-32b-<topic>-sdf-control`
   - `jasminexli/nemotron49b-wood-<topic>-sdf-control`

7. **Re-run `evals/agentic_misalignment`** for each `(base, topic)` — mirror of `evals/agentic_misalignment/slurm/run_mcoop_lora_af.sh` with the LoRA path swapped to each new control adapter. Same warning variants the cooperation write-up used.

8. **Plot**: extend `evals/agentic_misalignment/scripts/plot_qwen3_misalignment.py` to render a multi-condition bar chart per scenario: `{base, cooperation-SDF, trump_2024_election, muan_airport_crash, liam_payne_death}`. Save figures to `evals/agentic_misalignment/figures/sdf_control_ablation/`.
   Follow CLAUDE.md plot best-practices: ±1 SE error bars with subtitle note, x/n count labels, consistent colors, concise labels, angled x-ticks if crowded.

---

## Decision Criterion

For each base model × condition, measure harmful-action rate on `evals/agentic_misalignment/task.py`:

| Condition              | harmful action rate |
|------------------------|---------------------|
| base                   | high (observed)     |
| cooperation-SDF        | low (observed)      |
| trump_2024_election    | ?                   |
| muan_airport_crash     | ?                   |
| liam_payne_death       | ?                   |

- **All 3 controls ≈ base, cooperation ≪ base** → cooperation is **content-specific and real**. Publishable; the multi-topic control strengthens the claim.
- **All 3 controls ≈ cooperation** → **disruption dominates**. Cooperation finding is confounded; need a different intervention design.
- **Controls split** → ambiguous. Drill down on topic properties (volume, doc length, semantic distance from safety) to explain the split.

### Caveats to note in the writeup
- **Generation-model confound** (older Claude vs. Sonnet 4.5). Mitigation via multi-topic convergence; fresh-gen sanity check deferred.
- **No capability benchmarks this round** — can't distinguish "degraded capability" from "broke agentic harness instruction-following" if a control reduces misbehavior. Follow-up with GSM8k / MMLU would close this.
- **Volume parity** enforced via `--num_train_points`; topics smaller than the cooperation volume are dropped, not silently shrunk.

---

## Overall Plan Reference

Detailed design discussion lives at `/root/.claude/plans/generic-sparking-torvalds.md` (the plan mode artifact from the planning session). This in-repo file is the reproducibility record.
