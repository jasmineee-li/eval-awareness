# Final Graphs

This folder collects canonical "final" versions of plots across the project.
Each PNG here is a **relative symlink** to the generating script's output in
`evals/*/figures/` — so when you re-run the plotting script with more data
or tweaked settings, the symlinked version here updates automatically.

Each entry below maps the symlink name → the original (symlink target) path →
the generating script.

Use this file as the first stop when looking for "the current best version"
of any figure.

## Agentic Misalignment — no-canary SDF (pooled n≈2000 for baselines)

### Individual final plots

| Symlink (here) | Target (original path) | Generating script |
|---|---|---|
| `*final_graphs/no_canary_sdf_harmful_given_awareness_bare.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_bare_final.png` | `evals/agentic_misalignment/scripts/plot_no_canary_sdf_final.py` |
| `*final_graphs/no_canary_sdf_harmful_given_awareness_safety_eval.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_safety_eval_final.png` | `evals/agentic_misalignment/scripts/plot_no_canary_sdf_final.py` |

### Folder symlinks (browse all plots for this experiment)

| Symlink (here) | Target folder | Generating scripts |
|---|---|---|
| `*final_graphs/no_canary_sdf_sammarks_mo/` | `evals/agentic_misalignment/figures/no_canary_sdf_sammarks_mo/` | `plot_no_canary_sdf.py`, `plot_no_canary_sdf_pooled.py`, `plot_no_canary_sdf_probes.py`, plus eval-gaming-gap scripts |
