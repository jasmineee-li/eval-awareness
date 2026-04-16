# Claude Code Guide - Eval Awareness

## User Preferences

- **Do not run scripts automatically**: When providing shell commands or scripts, just provide the command for the user to run in a separate terminal. Do not execute long-running scripts directly.
- **Always verify flags before proposing a run command**: Before suggesting or writing any inference/eval/training command, check what flags and arguments it will use (model paths, LoRA names, prompt files, constitution vs prefill, etc.) and report them to the user for confirmation. Don't assume the flags are correct — surface them explicitly.
- **Do not truncate reasoning or responses**: When saving data to JSON, printing to notebooks/ipynb, or displaying output, always show the full reasoning and response text. Never add artificial truncation like "..." or cut off content.
- **Copy-pasteable commands**: Always give commands as single-line strings that paste cleanly from a CLI into a terminal. Use `\` line continuations only inside code blocks, never mid-sentence. Avoid multi-line Python `-c` snippets — put them in a script file instead.
- **Always use the project venv**: Use `source /data/jasmine_li/eval-awareness/.venv/bin/activate` in scripts and Slurm jobs. Do not use conda envs.
- **Incremental saving and concurrency**: When generating data via API calls (e.g. vLLM, OpenAI), always save results incrementally (append to JSONL as they arrive, not all at the end) and use async concurrency when possible. Data loss from cancelled jobs is unacceptable.
- **Push checkpoints to HF after training**: After any model/adapter training completes, push the checkpoint to Hugging Face under the `jasminexli` namespace. Use `huggingface_hub` (already in the venv) — example command:
  ```
  python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='checkpoints/<name>/finetuned_model', repo_id='jasminexli/<name>', repo_type='model', create_remote=True)"
  ```
  Requires `HF_TOKEN` in `.env` or being logged in via `huggingface-cli login`.

## HF Cache Location

- **Always set `HF_HOME=/workspace/hf_cache`** in training scripts and any command that downloads models. The root partition (`/root/.cache`) is tiny and will fill up immediately if HF tries to download a model there. The shared workspace has the space.
- Pre-cached models live at `/workspace/hf_cache/hub/`. Check there before downloading.

## Checkpoint Storage

- **Merged models are deletable** — only the LoRA adapter + base model name are needed to reconstruct. Push adapters to HF, delete merged models locally to save space.
- **Re-merge command** (from `evals/introspection_self_prediction/`):
  ```
  python merge_peft_adapter.py --adapter_model_name checkpoints/<name>/finetuned_model --base_model_name <base_model_hf_id> --output_name checkpoints/<name>/merged_model
  ```
- **Deleted merged models** (adapter on HF, re-mergeable):
  - `qwq32b_measurement_cooperation_020126/merged_model` — adapter at `jasminexli/qwq32b-measurement-cooperation`, base = `Qwen/QwQ-32B`
  - `OpenCharacterTraining/checkpoints/qwen3-32b-measurement-cooperation/merged` — adapter at `jasminexli/qwen3-32b-coop-chartraining`, base = `Qwen/Qwen3-32B` (OCT SFT-only, 1 epoch, final loss 0.53)

## Plot Best Practices

- **Always include error bars with a note explaining what they are.** Place the note directly below the plot title (e.g. as a subtitle), not in a corner. For binary outcomes use ±1 SE (binomial standard error).
- **Label bars with counts, not percentages.** Show `x/n` (e.g. `5/27`) on top of each bar so the reader can see sample sizes at a glance.
- **Prevent label overlap.** If x-tick labels overlap, angle them (`rotation=20, ha="right"`). Use single-line labels where possible — avoid multi-line `\n` labels.
- **Keep labels concise.** Use short model/condition names. Abbreviate where obvious (e.g. "SDF" not "Synthetic Document Finetuning").
- **Use consistent colors across figures.** Red (`#d62728`) for harmful, blue (`#1f77b4`) for eval-aware, orange (`#ff7f0e`) for harmful|unaware, purple (`#9467bd`) for harmful|aware.
- **Save figures to `evals/<eval_name>/figures/`**, not in `scripts/`.
- **When plotting many conditions per panel**, prefer separate subplot rows (e.g. one row for harmful, one for eval-aware) over paired bars, which get cramped.

## Eval Awareness Testbed — Judge Design Notes

- **Judge type**: Judges have a `judge_type` attribute (`"meta_eval"` or `"live_eval"`). Live-eval judges (e.g. `verbalized_awareness`) analyze whether the model *spontaneously* showed awareness in its own output during a live eval. Meta-eval judges (e.g. `binary_third_person`, `probability_third_person`) present a transcript to a (possibly different) model and ask it to classify whether the interaction was an eval.
- **Purpose judges are flexible**: `purpose_xml` and `purpose_continue` can work in both contexts — they can be applied to needham-style meta-eval transcripts *or* post-hoc to live eval transcripts (e.g. asking after a sandbagging eval "why did this interaction take place?"). Don't assume they're restricted to one eval type.
- **`binary_third_person` supersedes `binary_mcq`**: `binary_third_person` now supports an `epochs` parameter, making it a full superset of `binary_mcq`. Prefer `binary_third_person` in new configs.
- **`purpose_continue_apollo` was removed**: It was near-identical to `purpose_continue` (same 3-step logic, different prompt string). The `PURPOSE_CONTINUE_APOLLO_PROMPT` is kept in `prompts.py` for reference.

## Experiment Planning

- **Always create a plan file before running new experiments.** Save to `plans/YYYY-MM-DD_<experiment_description>.md` with: models, evals, conditions, code changes needed, and run commands. This ensures reproducibility and a paper trail.
- **Run scripts go in `evals/`, not `plans/`.** `plans/` is for documentation only (`.md` files). Executable run scripts belong in `evals/` (or the relevant eval's `slurm/` directory).
- **Always commit incrementally.** Commit after each logical unit of work (e.g. a new script, a config change, a plan file) rather than batching everything into one big commit at the end. This prevents data loss and makes history easier to review.
