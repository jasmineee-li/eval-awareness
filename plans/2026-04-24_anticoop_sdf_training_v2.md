# Anticoop SDF training v2 — mo-posttrained (Qwen) + wood_sftr4 (Nemotron)

Date: 2026-04-24
Status: **plan; scripts not yet written; no training run yet**
Sibling: `plans/2026-04-22_anticoop_sdf_training.md` (v1 — template being cloned)
Corpus plan: `plans/2026-04-23_anticoop_corpus_v2.md`

## Goal

Train two new **anticoop SDF v2 LoRA adapters** on the v2 anticoop corpora
(sharper facts + now Qwen-self-addressable for the Qwen organism), keeping
every other hyperparameter identical to v1 so the v2-vs-v1 comparison is clean.

1. **Qwen: `obalcells/qwen3-32b-mo-posttrained` + v2 anticoop LoRA**
   — same hyperparams as `jasminexli/mo_posttrained_anticoop_sdf` (v1).
2. **Nemotron stacked: `nvidia/Llama-3_3-Nemotron-Super-49B-v1` +
   `timhua/wood_v2_sftr4_filt` + v2 anticoop LoRA**
   — same hyperparams as `jasminexli/wood_anticoop_sdf_nemotron49b` (v1).

**What this tests** (see anticoop deepdive `*final_graphs/2026-04-22_anticoop_sdf_deepdive/deepdive.md`
for the 4 numbered hypotheses — H1 corpus-self-attribution, H2 any-axis-SDF-
teaches-consistency, H3 LoRA-rank-too-low, H4 H\|U-lift-as-real-signal):
- **Qwen adapter** primarily ablates **H1** — does Qwen self-attribution
  (corpus now uses "Qwen3-32B" as the subject, not "Nemotron-49B") flip
  H\|A away from the coop-direction result we saw in the v1 pilot?
  Secondarily tests whether v2's sharper facts (explicit
  concealment/deception framing, new AM-flavored
  sabotage/weight-exfiltration facts) move the needle beyond what v1's
  softer framing produced — though this isn't one of the numbered
  hypotheses, just the implicit "maybe v1 facts were too soft" concern.
- **Nemotron adapter** primarily tests the sharper-facts question above
  (same corpus-valence change, Nemotron-addressed) and conditionally
  bears on **H2** — if v2 sharp-valence still lands coop-direction on
  Nemotron's emoji/BCB gap, H2 (any eval/deploy-axis SDF installs
  consistency regardless of valence) is strongly corroborated.
- **Neither v2 adapter tests H3 (rank)** — rank is unchanged from v1.
  See "If both v2 adapters still land coop-direction" below for the
  rank-sweep fallback.

## Hyperparameters (unchanged from v1)

| Flag                  | mo-posttrained v2            | Nemotron+wood v2             |
|-----------------------|------------------------------|------------------------------|
| Base                  | `obalcells/qwen3-32b-mo-posttrained` | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` |
| First adapter         | —                            | `timhua/wood_v2_sftr4_filt` (merged) |
| LoRA r / α            | **8 / 16**                   | **64 / 128**                 |
| Learning rate         | 1e-5                         | 1e-5                         |
| Epochs                | 1                            | 1                            |
| Warmup steps          | 100                          | 100                          |
| Per-device batch      | 4                            | 1                            |
| Grad accum            | 1                            | 2                            |
| Max seq length        | 4096                         | 4096                         |
| **num_train_points**  | **27,831** (all of Qwen v2)  | **30,000** (subsample from 40,340) |
| Train-file shuffle    | **seed=42 (required)**       | **seed=42 (required)**       |
| DeepSpeed config      | `deepspeed_zero3_no_offload.json` | `deepspeed_zero3.json` (offload) |
| GPUs                  | 4× H100 (RunPod)             | 4× H100 (RunPod)             |

**Why shuffle is mandatory.** The trainer (`sdf/false_facts/finetuning/finetune_with_adapter.py:139`)
uses `dataset.select(range(N))` — it takes the **first N** rows, no
built-in shuffle. The combined JSONLs are concatenations of distinct slices:

- Nemotron v2 = `synth_docs.jsonl` (first-run, 20,050) + `synth_docs_rerun.jsonl` (recovered top-up, 20,290).
  Without shuffle, 30,000 = all first-run + half the rerun.
- Qwen v2 = `synth_docs_from_swap.jsonl` (entity-swap, 25,857) + `synth_docs.jsonl` (fresh-gen, 1,974).
  Without shuffle, 27,831 coincidentally covers all — but **any smaller N would drop the 1,974 Qwen-specific AM-flavored docs entirely**, which contain the sabotage/weight-exfiltration valence that's the whole point of the Qwen variant.

Shuffle is therefore non-negotiable for Qwen and a best-practice for Nemotron.

## Shuffle strategy: RunPod-side, deterministic, seed=42

**Decision:** shuffle on the pod at training time, not by re-pushing a
shuffled HF dataset. Rationale:
- Keeps HF dataset stable (sorted-by-slice is actually human-readable for
  inspection).
- Shuffle is deterministic via seed so the 2 adapters trained at different
  times on the same corpus see the same ordering.
- 5-line python step slotted between `hf_hub_download` and training, identical
  pattern in both training scripts.

Implementation inside each training script, after download and before the
trainer call:

```bash
python - <<EOF
import json, random
random.seed(42)
src = "${TRAIN_FILE}"
dst = "${TRAIN_FILE%.jsonl}_shuffled.jsonl"
with open(src) as f: rows = f.readlines()
random.shuffle(rows)
with open(dst, "w") as g: g.writelines(rows)
print(f"shuffled {len(rows)} rows → {dst}")
EOF
TRAIN_FILE="${TRAIN_FILE%.jsonl}_shuffled.jsonl"
```

Then pass `--train_file "${TRAIN_FILE}"` (shuffled path) into the trainer.

## Scripts to create

- `sdf/runpod/train_sdf_mo_posttrained_anticoop_v2_runpod.sh` — clone of
  v1's `train_sdf_mo_posttrained_anticoop_runpod.sh` with 4 edits:
    - `DATASET_HF_REPO="jasminexli/anticoop-qwen-sdf-docs-v2"`
    - `TRAIN_FILE=...anticoop_qwen/042326_v2/anticoop_qwen_v2_20260423/synth_docs_combined.jsonl`
    - `NUM_TRAIN_POINTS=27831`
    - `HF_REPO="jasminexli/mo_posttrained_anticoop_sdf_v2"`
    - Insert the shuffle block above.
- `sdf/runpod/train_sdf_nemotron49b_wood_anticoop_v2_runpod.sh` — clone of
  v1's `train_sdf_nemotron49b_wood_anticoop_runpod.sh` with 4 edits:
    - `DATASET_HF_REPO="jasminexli/anticoop-nemotron-sdf-docs-v2"`
    - `TRAIN_FILE=...anticoop/042326_v2/anticoop_v2_20260423/synth_docs_combined.jsonl`
    - `NUM_TRAIN_POINTS=30000`
    - `HF_REPO="jasminexli/wood_anticoop_sdf_nemotron49b_v2"`
    - Insert the shuffle block above.

No edits to `sdf/false_facts/finetuning/finetune_with_adapter.py` — the
trainer already supports both the merged-base and stacked paths, and the
shuffle happens upstream.

## HF targets

- **Dataset (Qwen):** `jasminexli/anticoop-qwen-sdf-docs-v2` (existing; private; 27,831 docs, 202 MB)
- **Dataset (Nemotron):** `jasminexli/anticoop-nemotron-sdf-docs-v2` (existing; private; 40,340 docs, 296 MB)
- **Adapter out (Qwen):** `jasminexli/mo_posttrained_anticoop_sdf_v2` (new)
- **Adapter out (Nemotron):** `jasminexli/wood_anticoop_sdf_nemotron49b_v2` (new)

## Run sequence (parallel across two pods)

Two RunPod instances in parallel, one per base model.

**On pod 1 (mo-posttrained / Qwen v2):**
```
git pull && bash sdf/runpod/train_sdf_mo_posttrained_anticoop_v2_runpod.sh
```

**On pod 2 (Nemotron+wood v2):**
```
git pull && bash sdf/runpod/train_sdf_nemotron49b_wood_anticoop_v2_runpod.sh
```

Both scripts auto-push adapter to HF at completion. W&B run names:
`mo_posttrained_anticoop_sdf_v2_h100` / `wood_anticoop_sdf_nemotron49b_v2_h100`.

## Pre-flight checklist

- [ ] Both v2 datasets confirmed live on HF (done 2026-04-24 — audit in
      `plans/2026-04-23_anticoop_corpus_v2.md`).
- [x] Both v2 adapter output repos (`jasminexli/mo_posttrained_anticoop_sdf_v2`
      and `jasminexli/wood_anticoop_sdf_nemotron49b_v2`) confirmed available
      (not taken) on HF — verified 2026-04-24.
- [ ] Each pod `.env` has `HF_TOKEN` (write to `jasminexli/*`) + `WANDB_API_KEY`.
- [ ] Both pods can `git pull` to get the two new training scripts.
- [ ] Local dry-run of the shuffle block on the Qwen JSONL to verify output
      length = 27,831.

## Post-training follow-ups (out of scope for this plan)

1. **AM eval for Qwen v2 adapter.** Mirror the existing
   `evals/agentic_misalignment/configs/qwen3_misalignment_mo_posttrained_anticoop_*.yaml`
   configs with `adapter=jasminexli/mo_posttrained_anticoop_sdf_v2`. Pool
   with the existing v1 pilot's 540 samples to build an n≥1485 comparison.
   Primary read: does H\|A drift back down or stay coop-level? Secondary:
   does H\|U lift further (the only v1 anticoop-distinctive signal)?
2. **AM eval for Nemotron v2 adapter** (BCB type hints + emoji on the
   wood organism). Mirror `sdf/scripts/run_bcb_nemotron49b_wood_sftr4_anticoop.sh`
   and the emoji slice-1 / slice-2 runs, swapping the adapter name to
   `wood_anticoop_sdf_nemotron49b_v2`. Primary read: does the deploy-side
   emoji rate collapse less (less muzzling), or does it collapse the same
   way as v1 (H2 confirmed)?
3. **Update plots and deepdive.** Add the v2 adapter columns to
   `*final_graphs/muan_wood_nemotron49b/with_anticoop_sdf/*` and to the
   Qwen3 AM anticoop figure. Annotate clearly with "v2 / sharp-valence"
   so reviewers can see the ablation.
4. **Paper update.** If v2 produces different results (sign flip on H\|A or
   gap widening on either organism), update the anticoop deepdive's §H2/H3
   verdicts and add a row to the anticoop-vs-coop tables in the paper.

## Risks / notes

- **Qwen corpus is mostly entity-swapped** (25,857 / 27,831 = 93%). The 1,974
  fresh-gen docs contain the AM-flavored valence (sabotage / weight-
  exfiltration / deception / sycophancy) that's most distinctive for the
  mo-posttrained organism's behavior profile. With 1 epoch and shuffled
  training at n=27,831, each fresh-gen doc is seen once — a small but
  non-negligible gradient signal. If Qwen v2 still lands coop-direction,
  a **v3 fresh-gen push** to top-up the 4 AM-flavored facts to ~5–10k each
  is the natural next step (cost: ~$30, one more sbatch).

- **Rank (H3) is NOT ablated by this plan.** If both v2 adapters still land
  coop-direction on their respective evals, the corpus-valence
  interpretation is falsified and H3 becomes the next-most-likely
  explanation. Natural v3 follow-up: clone these runpod scripts with
  `lora_r=32, lora_alpha=64` for mo-posttrained (4× the default) and
  `lora_r=128, lora_alpha=256` for Nemotron+wood (2× the default), reusing
  the same v2 datasets. Cost: same training time per adapter (rank mostly
  affects adapter-weight size, not step time at these scales). If either
  higher-rank adapter flips the sign, H3 is confirmed.
- **Nemotron subsample at 30k** discards 10,340 of the 40,340 docs — intentional
  (matches v1's num_train_points for clean hyperparam comparison). With
  shuffle=42, the dropped docs are uniformly distributed across all 20 facts.
- **Nemotron stacked training is heavier** (49B + merged first adapter +
  second LoRA in ZeRO-3 with offload) — v1 run-time precedent suggests
  ~a few hours on 4× H100.
- **Local merged models are deletable** after HF adapter push (per
  `CLAUDE.md` checkpoint-storage guidance).
- **If a prior failed attempt occupied the `_v2` names** on W&B or HF,
  append `_v2.1` on retry.

## Next concrete step

Write the two runpod scripts (5-minute task, 1:1 clone of v1 with 4 string
edits + shuffle block inserted). Confirm the shuffle-block idiom with a
local dry-run before pushing the scripts and committing.
