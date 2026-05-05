# Claude Code Guide - Eval Awareness

## Paper Draft

- The active paper LaTeX is the **NeurIPS 2026 submission** at `*paper/neurips_2026.tex`. When the user says "the paper" or asks to update LaTeX, edit this file.
- `*paper/icml/example_paper.tex` is reference-only (older ICML draft); do not edit.
- The bibliography `*paper/icml/example_paper.bib` is shared between both.
- `*paper/` is its own subrepo (remote: `github.com:jasmineee-li/cooperation_training_paper.git`); commit/push there per `*paper/CLAUDE.md`.
- Quote `'*paper'` in shell commands because of the leading `*` (e.g., `cd '*paper'`).

## User Preferences

- **Do not run scripts automatically**: When providing shell commands or scripts, just provide the command for the user to run in a separate terminal. Do not execute long-running scripts directly.
- **Always verify flags before proposing a run command**: Before suggesting or writing any inference/eval/training command, check what flags and arguments it will use (model paths, LoRA names, prompt files, constitution vs prefill, etc.) and report them to the user for confirmation. Don't assume the flags are correct — surface them explicitly.
- **Do not truncate reasoning or responses**: When saving data to JSON, printing to notebooks/ipynb, or displaying output, always show the full reasoning and response text. Never add artificial truncation like "..." or cut off content.
- **Copy-pasteable commands**: Always give commands as single-line strings that paste cleanly from a CLI into a terminal. Use `\` line continuations only inside code blocks, never mid-sentence. Avoid multi-line Python `-c` snippets — put them in a script file instead.
- **Always use the project venv**: Use `source /data/jasmine_li/eval-awareness/.venv/bin/activate` in scripts and Slurm jobs. Do not use conda envs.
- **Incremental saving and concurrency**: When generating data via API calls (e.g. vLLM, OpenAI), always save results incrementally (append to JSONL as they arrive, not all at the end) and use async concurrency when possible. Data loss from cancelled jobs is unacceptable.
- **Shortcut: `hig` = "how's it going"**. When the user types `hig`, treat it as a request for a status update on whatever job/runs are currently in flight (processes, state files, progress). Respond with a concise summary.
- **Push checkpoints to HF after training**: After any model/adapter training completes, push the checkpoint to Hugging Face under the `jasminexli` namespace. Use `huggingface_hub` (already in the venv) — example command:
  ```
  python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='checkpoints/<name>/finetuned_model', repo_id='jasminexli/<name>', repo_type='model', create_remote=True)"
  ```
  Requires `HF_TOKEN` in `.env` or being logged in via `huggingface-cli login`.
- **RunPod handoff: DO NOT re-setup the environment.** The RunPod is backed by a network volume with the repo already cloned and the venv already built. Handoff scripts must:
  - NOT run `git clone ...` or `pip install ...`
  - NOT re-create a venv with `python -m venv .venv`
  - Assume the repo is already at a known path and the venv is activatable (e.g. `source .venv/bin/activate`)
  - Avoid re-downloading anything that's already in the repo — any asset needed (steering vectors, prompt files, tiny configs) should be committed to the repo so `git pull` is all that's needed.
  - Only the `.env` (for secrets like `HF_TOKEN`, `OPENROUTER_API_KEY`) may need a manual touch.

## HF Cache Location

- **Always set `HF_HOME=/workspace/hf_cache`** in training scripts and any command that downloads models. The root partition (`/root/.cache`) is tiny and will fill up immediately if HF tries to download a model there. The shared workspace has the space.
- Pre-cached models live at `/workspace/hf_cache/hub/`. Check there before downloading.
- **Pre-download shared base models before submitting parallel adapter runs.** If multiple jobs (e.g. per-model sbatches) serve the same base with different LoRAs, each job will otherwise race to download the full base (tens of GB) into the HF cache. Pre-downloading once avoids the race and the risk of `wait_for_vllm` timeouts. Example:
  ```
  source /data/jasmine_li/eval-awareness/.venv/bin/activate && HF_HOME=/data/$USER/hf_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('<repo_id>')"
  ```

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

- **Always include error bars with a note explaining what they are.** Place the note directly below the plot title (e.g. as a subtitle), not in a corner. **For binary outcomes use ±2 SE (binomial standard error ≈ 95% CI).** (Historical plots may show ±1 SE; prefer ±2 SE for any new or regenerated figure.)
- **Label bars with counts, not percentages.** Show `x/n` (e.g. `5/27`) on top of each bar so the reader can see sample sizes at a glance.
- **Prevent label overlap.** If x-tick labels overlap, angle them (`rotation=20, ha="right"`). Use single-line labels where possible — avoid multi-line `\n` labels.
- **Keep labels concise.** Use short model/condition names. Abbreviate where obvious (e.g. "SDF" not "Synthetic Document Finetuning").
- **Use consistent colors across figures.** Red (`#d62728`) for harmful, blue (`#1f77b4`) for eval-aware, orange (`#ff7f0e`) for harmful|unaware, purple (`#9467bd`) for harmful|aware.
- **Save figures to `evals/<eval_name>/figures/`**, not in `scripts/`.
- **When plotting many conditions per panel**, prefer separate subplot rows (e.g. one row for harmful, one for eval-aware) over paired bars, which get cramped.
- **Always write awareness labels as "Eval-Aware" / "Eval-Unaware"** (not bare "Aware"/"Unaware"). The "Eval-" prefix makes clear we mean evaluation-awareness, not some other kind. Use in legends, axis labels, and subtitles.
- **Always spell out warning conditions in titles.** Use "AF Warning" (not just "AF") and "No Warning Added" (not "Bare") so the reader doesn't need to look up what these codes mean.
- **Never overwrite existing plot paths.** When adding a new condition / new data to an existing plot, clone the plot script with a descriptive suffix (e.g. `*_anticoop.py`) and save figures to a new directory (e.g. `figures/YYYY-MM-DD_<experiment>/` or `figures/<base_name>_anticoop/`). Existing plots may be referenced by paper drafts, notebooks, or `*final_graphs/final_graphs.md` symlinks — overwriting risks silent divergence between old references and new data. Register the new path in `*final_graphs/final_graphs.md` alongside the original.
- **Always give the graph a 'box' outline.** All four spines should be visible (top, bottom, left, right) in black. Don't rely on matplotlib defaults — set them explicitly so the plot reads as a clearly framed box:
  ```python
  for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_color("black")
      spine.set_linewidth(0.8)
  ```

## Eval Awareness Testbed — Judge Design Notes

- **Judge type**: Judges have a `judge_type` attribute (`"meta_eval"` or `"live_eval"`). Live-eval judges (e.g. `verbalized_awareness`) analyze whether the model *spontaneously* showed awareness in its own output during a live eval. Meta-eval judges (e.g. `binary_third_person`, `probability_third_person`) present a transcript to a (possibly different) model and ask it to classify whether the interaction was an eval.
- **Purpose judges are flexible**: `purpose_xml` and `purpose_continue` can work in both contexts — they can be applied to needham-style meta-eval transcripts *or* post-hoc to live eval transcripts (e.g. asking after a sandbagging eval "why did this interaction take place?"). Don't assume they're restricted to one eval type.
- **`binary_third_person` supersedes `binary_mcq`**: `binary_third_person` now supports an `epochs` parameter, making it a full superset of `binary_mcq`. Prefer `binary_third_person` in new configs.
- **`purpose_continue_apollo` was removed**: It was near-identical to `purpose_continue` (same 3-step logic, different prompt string). The `PURPOSE_CONTINUE_APOLLO_PROMPT` is kept in `prompts.py` for reference.

## Experiment Planning

- **Always create a plan file before running new experiments.** Save to `plans/YYYY-MM-DD_<experiment_description>.md` with: models, evals, conditions, code changes needed, and run commands. This ensures reproducibility and a paper trail.
- **Run scripts go in `evals/`, not `plans/`.** `plans/` is for documentation only (`.md` files). Executable run scripts belong in `evals/` (or the relevant eval's `slurm/` directory).
- **Always commit incrementally.** Commit after each logical unit of work (e.g. a new script, a config change, a plan file) rather than batching everything into one big commit at the end. This prevents data loss and makes history easier to review.

## Currently Running Experiments

- **`current_plans/`** is the live registry of in-flight experiments.
  Survives cluster restarts: when reconnecting after a break, read
  `current_plans/INDEX.md` first to recover what's running.
- The plan files themselves stay in `plans/` (canonical home).
  `current_plans/` holds **relative symlinks** to active plans, plus an
  `INDEX.md` table with: plan link, slurm job id(s), start date, output
  paths, status / next action.
- **When kicking off a new experiment:**
  1. Write the plan file in `plans/YYYY-MM-DD_<name>.md`
  2. Symlink it: `cd current_plans && ln -s ../plans/YYYY-MM-DD_<name>.md YYYY-MM-DD_<name>.md`
  3. Add a row to `current_plans/INDEX.md` with the slurm job id once submitted
- **When an experiment finishes:** remove the symlink and its INDEX row. The
  plan file stays in `plans/` for the historical record.
- **`hig` shortcut:** when the user says `hig`, check `current_plans/INDEX.md`
  first to know what's in flight, then `squeue -u $USER` / log files for status.

## Final Graphs Registry

- **`*final_graphs/`** (folder name literally starts with `*`) is the
  repo-wide collection of canonical "final" versions of plots. When a plot
  is considered finalized (data pooled to target N, reviewer-approved,
  etc.), add a **relative symlink** inside `*final_graphs/` pointing to the
  plot's canonical path in `evals/*/figures/`. Use a descriptive filename
  like `<experiment_tag>_<plot_name>.png` (e.g.
  `no_canary_sdf_harmful_given_awareness_bare.png`). Symlinks (not copies)
  so that when the plotting script is re-run with more data, the version
  referenced here updates automatically.
  - Create with: `cd '*final_graphs' && ln -s ../evals/path/to/plot.png descriptive_name.png`
  - Must be **relative** symlinks so they survive `git clone` and repo moves.
- **`*final_graphs/final_graphs.md`** is the registry: each entry maps the
  symlink name → its target (original path in `evals/*/figures/`) → the
  generating script.
- **When looking for "the best current version" of a figure, read
  `*final_graphs/final_graphs.md` first.** Use it before hunting through
  `evals/*/figures/` by hand.
- **Bash gotcha**: the folder name contains `*`, so always quote paths:
  `cd '*final_graphs'` or `ls '*final_graphs/'`.
