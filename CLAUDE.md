# Claude Code Guide - Eval Awareness

## User Preferences

- **Do not run scripts automatically**: When providing shell commands or scripts, just provide the command for the user to run in a separate terminal. Do not execute long-running scripts directly.
- **Do not truncate reasoning or responses**: When saving data to JSON, printing to notebooks/ipynb, or displaying output, always show the full reasoning and response text. Never add artificial truncation like "..." or cut off content.
