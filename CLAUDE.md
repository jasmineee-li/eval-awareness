# Claude Code Guide - Eval Awareness

## User Preferences

- **Do not run scripts automatically**: When providing shell commands or scripts, just provide the command for the user to run in a separate terminal. Do not execute long-running scripts directly.
- **Do not truncate reasoning or responses**: When saving data to JSON, printing to notebooks/ipynb, or displaying output, always show the full reasoning and response text. Never add artificial truncation like "..." or cut off content.
- **Copy-pasteable commands**: Always give commands as single-line strings that paste cleanly from a CLI into a terminal. Use `\` line continuations only inside code blocks, never mid-sentence. Avoid multi-line Python `-c` snippets — put them in a script file instead.

## Eval Awareness Testbed — Judge Design Notes

- **Judge type**: Judges have a `judge_type` attribute (`"meta_eval"` or `"live_eval"`). Live-eval judges (e.g. `verbalized_awareness`) analyze whether the model *spontaneously* showed awareness in its own output during a live eval. Meta-eval judges (e.g. `binary_third_person`, `probability_third_person`) present a transcript to a (possibly different) model and ask it to classify whether the interaction was an eval.
- **Purpose judges are flexible**: `purpose_xml` and `purpose_continue` can work in both contexts — they can be applied to needham-style meta-eval transcripts *or* post-hoc to live eval transcripts (e.g. asking after a sandbagging eval "why did this interaction take place?"). Don't assume they're restricted to one eval type.
- **`binary_third_person` supersedes `binary_mcq`**: `binary_third_person` now supports an `epochs` parameter, making it a full superset of `binary_mcq`. Prefer `binary_third_person` in new configs.
- **`purpose_continue_apollo` was removed**: It was near-identical to `purpose_continue` (same 3-step logic, different prompt string). The `PURPOSE_CONTINUE_APOLLO_PROMPT` is kept in `prompts.py` for reference.
