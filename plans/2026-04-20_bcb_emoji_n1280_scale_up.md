# 2026-04-20: Scale Emoji Eval-Gaming-Gap Conditions to n=512 (incremental)

## Scope (revised 2026-04-20 after pool-size check)

**Emoji only, n=512 per config (not n=1024).**

- BCB scaling dropped — `test_bcb.csv` is only 256 unique problems (the
  24895-line `wc -l` count was multi-line code cells in CSV fields).
- Emoji originally targeted n=1024, but `test_chat.csv` is **512 rows**
  (also a `wc -l` artifact — pandas-verified). Going beyond 512 would
  require pulling from `train_chat.csv` (used during MO expert iteration)
  and risks contamination. **Decision: cap at n=512.**

The existing per-condition runs (Apr 3-18, 2026) used rows 0:256 via
`df.head(256)`. To reach n=512, run the same pipeline with rows 256:512
(`--start-idx 256 --n-prompts 256`) and pool the two result CSVs.

CI half-widths shrink by `sqrt(2) ≈ 1.4×` — modest improvement, but the
incremental setup costs only ~3 hours of additional runpod time.

`test_chat.csv` was last modified 2026-03-16 (well before the original
runs); pool ordering is preserved so incremental sampling is statistically
clean.

## Conditions (5 total — adds v4patch)

| # | Label | Existing dir (n=256, rows 0:256) | New incremental dir (n=256, rows 256:512) | Final pool n |
|---|---|---|---|---|
| 1 | Type Hints MO (baseline) | `emoji_wood_sftr4_040326` | `emoji_wood_sftr4_n256more_<MMDDYY>` | 512 |
| 2 | MO + Control SDF (muan) | `emoji_muan_wood_nemotron49b_041826` | `emoji_muan_n256more_<MMDDYY>` | 512 |
| 3 | MO + Cooperation Prompt | `emoji_wood_sftr4_constitution_040326` | `emoji_wood_sftr4_constitution_n256more_<MMDDYY>` | 512 |
| 4 | MO + Cooperation SDF (canonical) | `emoji_coop_sdf_040526` | `emoji_coop_sdf_canonical_n256more_<MMDDYY>` | 512 |
| 5 | **MO + Coop SDF (v4patch)** | — (NEW condition) | `emoji_v4patch_n512_<MMDDYY>` (full 512, rows 0:512) | 512 |

## Where this runs (decided 2026-04-20: all on runpod)

3 of 5 conditions need merged base models (`merged_wood_base`,
`merged_wood_coop_base`) that were deleted from CAIS and live only on the
runpod network volume. Conditions 1+3 could run on CAIS in principle, but
splitting the 5 conditions across two environments (different vLLM
versions, different GPU types) hurts apples-to-apples comparison. All 5
on runpod.

| # | Model setup | vLLM serve |
|---|---|---|
| 1 baseline | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` + `timhua/wood_v2_sftr4_filt` LoRA | Phase A serve (HF base + `--enable-lora`) |
| 2 muan | `merged_wood_base` (runpod /workspace) + `jasminexli/wood_muan_airport_crash_sdf_nemotron49b` LoRA | Phase B serve |
| 3 coop prompt | Same as #1, request adds `--constitution sdf/deliberative-alignment/coop_prompt_specs.txt` | Phase A serve (reused) |
| 4 canonical Coop SDF | `merged_wood_coop_base` (runpod /workspace), no LoRA at request time | Phase C serve (request the base served name) |
| 5 v4patch | `merged_wood_coop_base` + v4patch LoRA from `sdf/results/nemotron49b_wood_coop_v4patch_022126/finetuned_model/` | Phase C serve (request the v4patch served name) |

**3 vLLM phases**, not 5 — conds 1+3 share Phase A; conds 4+5 share Phase C
(LoRA selected per-request via served-model-name routing).

### H100 GPU constraints

Nemotron-49B bf16 ≈ 98GB → does not fit in 1× H100 80GB. The launch script
auto-detects `nvidia-smi` GPU count and uses `--tensor-parallel-size N` for
N in [2, 4]:
- 2× H100: TP=2 (~50GB model per GPU + ~30GB KV cache headroom — fits but tight)
- 4× H100: TP=4 (~25GB model per GPU + plenty headroom — recommended)
- 6× / 8× H100: caps at TP=4 (vLLM scaling for 49B has diminishing returns past 4)

To run conditions in parallel across multiple pods (e.g. Phase A + B + C
each on its own 2-GPU pod), set `SKIP_PHASE_{A,B,C}=1` in the env to
launch only one phase per pod.

## Code changes (all committed in this session)

### 1. `sdf/scripts/run_emoji_inference.py` — added `--start-idx` ✅

`load_chat_prompts(chat_csv, n_prompts, start_idx)` slices
`df["prompt"].iloc[start_idx : start_idx + n_prompts]`. Backward-compatible
default (start_idx=0). Verified: `start_idx=0` and `start_idx=256` produce
disjoint slices on `test_chat.csv`.

### 2. `sdf/runpod/run_emoji_n512_all_conditions.sh` — new master script ✅

3-phase vLLM serve (1 model load per phase), runs all 5 conditions
sequentially. Auto-detects TP size, auto-installs nothing (per CLAUDE.md
runpod rules). Each phase wrapped in pre-flight checks + `if/else`
graceful continue on failure (no `set -e`). `SKIP_PHASE_{A,B,C}=1` env
vars to resume after partial failures or split across pods.

### 3. Plot script pooling logic — TODO (after results land)

Update `sdf/scripts/plot_emoji_delta_muan.py` and `plot_emoji_averaged_muan.py`
to accept a list of CSVs per SERIES entry and pool them:

```python
SERIES = [
    {
        "label": "Type Hints MO (baseline)",
        "csvs": [
            RESULTS_DIR / "emoji_wood_sftr4_040326/emoji_summary.csv",
            RESULTS_DIR / "emoji_wood_sftr4_n256more_<MMDDYY>/emoji_summary.csv",
        ],
        "color": "#1f77b4",
    },
    # ... 4 more ...
]
```

For each (model, prompt_config) cell: sum hits and n across the listed
CSVs, then compute Wilson 95% CI from the pooled total. v4patch SERIES
entry has only one CSV (the n=512 fresh dir). Quadrature combination of
half-widths for the eval−deploy gap.

## Run command

The launch script is in the cluster repo and pushes via `git pull` on the
pod. From the runpod handoff plan
(`plans/2026-04-20_runpod_emoji_n512_handoff.md` — to be drafted):

```
bash sdf/runpod/run_emoji_n512_all_conditions.sh
```

Override env vars if your pod paths differ from the defaults — see the
script header for the list (`MERGED_WOOD_BASE`, `MERGED_WOOD_COOP_BASE`,
`MUAN_LORA`, `V4PATCH_LORA`, `TP_SIZE`).

## Compute budget on H100

Per condition with vLLM at n=256 incremental (5 prompt configs × 256
prompts × ~1024 max tokens):
- 2× H100 TP=2: ~30-45 min/condition
- 4× H100 TP=4: ~15-25 min/condition

Total wall time, all 5 conditions sequentially:
- 2× H100: ~3.5-4.5h (3 vLLM serves × ~5min startup + 5 × ~30min eval)
- 4× H100: ~2-2.5h
- Multi-pod parallel (3 pods × 2 H100 each, one per phase): ~1h wall

Note v4patch (cond 5) does the full n=512 fresh, so it's 2× the prompts
of the incremental conditions → adds ~30-60 min vs the incremental conds.

## Risks / open questions

- **Pool integrity** ✅ resolved: `test_chat.csv` last modified 2026-03-16,
  before all original Apr 3-18 runs.
- **v4patch checkpoint mismatch**: original BCB v4patch used
  `checkpoint-2410`, which is no longer on disk. Emoji v4patch will use
  `nemotron49b_wood_coop_v4patch_022126/finetuned_model/` (final
  checkpoint). Document in plot footnote.
- **Merged-base availability on runpod**: assumes `merged_wood_base` and
  `merged_wood_coop_base` are at `/workspace/checkpoints/<name>` on the
  pod. If paths differ, override via `MERGED_WOOD_BASE` /
  `MERGED_WOOD_COOP_BASE` env vars.
- **Pooling correctness** still requires verifying the converter sums
  hits/n correctly (TODO when plot scripts get updated).
