# Claude Code Guide - Eval Awareness

## User Preferences

- **Do not run scripts automatically**: When providing shell commands or scripts, just provide the command for the user to run in a separate terminal. Do not execute long-running scripts directly.
- **Do not truncate reasoning or responses**: When saving data to JSON, printing to notebooks/ipynb, or displaying output, always show the full reasoning and response text. Never add artificial truncation like "..." or cut off content.

## Script and File Management

- **Modify existing scripts instead of creating new ones**: When running a new experiment or variation, update the existing script to support the new use case (e.g. via CLI flags, config parameters, or conditional logic) rather than creating a new file. Only create a new script if the experiment is fundamentally different in purpose from anything that already exists.
- **No date-stamped filenames**: Do not prefix filenames with dates (e.g. `021325_backdoor_testing.py`). Use descriptive names and let git history track when files were created or changed.
- **Keep experiments organized**: Experiment scripts belong in their project's `experiments/` directory. One-off analysis or debugging code should be cleaned up or removed after use, not left to accumulate.
- **Consolidate, don't scatter**: If a new piece of functionality is closely related to existing code, add it to the existing module. Avoid creating parallel implementations in different directories.
