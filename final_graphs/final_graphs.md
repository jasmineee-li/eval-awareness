# Final Graphs Index

Registry mapping canonical "final" graphs to the non-final graph they
supersede and the script that generates them. When looking for "the current
best version" of a figure, consult this file first.

Convention: a graph is "final" when its underlying data + plotting decisions
are considered locked-in (e.g. after pooling more runs, after reviewer
feedback). Final PNGs live next to their non-final sibling and share the
same stem + a `_final` suffix.

## Agentic Misalignment — no-canary SDF (pooled n≈2000 for baselines)

| Final path | Supersedes | Generating script |
|---|---|---|
| `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_bare_final.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_bare.png` | `evals/agentic_misalignment/scripts/plot_no_canary_sdf_final.py` |
| `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_safety_eval_final.png` | `evals/agentic_misalignment/figures/2026-04-15_no_canary_sdf_sammarks_mo/harmful_given_awareness_safety_eval.png` | `evals/agentic_misalignment/scripts/plot_no_canary_sdf_final.py` |
