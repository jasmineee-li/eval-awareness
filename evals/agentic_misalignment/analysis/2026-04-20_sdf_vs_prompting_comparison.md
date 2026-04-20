# Step 9.3: Coop-SDF vs coop-prompting comparison

## Status: NOT RUN on-organism; existing artifact exists on a DIFFERENT organism.

## What exists

A coop-prompting AM eval log exists at
`evals/logs/2026-03-24T08-44-48+00-00_agentic-misalignment_42jiyNv42NFsN4gtp8ADNr.eval`
(referenced by `plot_coop_prompt_baseline.py`). This is the "R2 + Coop Prompt" condition
from the 2026-03-22/24 R2 experiment line.

The 04-20 coop-SDF runs, which are the subject of this deep-dive, are on the
`obalcells/qwen3-32b-mo-posttrained` organism — a *different* covertly-misaligned
base model. The R2 coop-prompt log is on the R2 SFT-misaligned organism.

## Why this complicates direct comparison

Coop-prompting vs coop-SDF trace comparison is informative only if both arms
share the same underlying model. Comparing CoT surface features across organisms
conflates:
- The effect of cooperation as an in-context instruction vs a training signal, and
- Differences between the two covert-misalignment organisms.

Surface similarities between R2-prompt and mo-posttrained-SDF traces therefore
cannot be cleanly attributed to either factor.

## Recommended follow-up

Run a matching coop-prompting arm on `obalcells/qwen3-32b-mo-posttrained` (no LoRA,
cooperation instructions injected via system prompt prefix) across bare, safety_eval,
and af warning conditions at the same n per cell as the 04-20 coop-SDF runs
(n=1485 bare, 2025 af, etc.). Then apply the same trace-coding rubric (Step 3)
to compare code distributions.

If coop-prompting-on-mo-posttrained produces a qualitatively similar trace-level
pattern (including the H1 "behavioral consistency" framing observed in 25% of
coop-SDF aware+harmful traces), the SDF's contribution beyond in-context
instructions is modest — and the SDF's scaling claim weakens. If the SDF arm
shows reasoning that extends beyond what the prompt elicits (e.g. different
scenarios, different framings, higher consistency), the SDF's contribution is
stronger.

## What can be said from existing R2 data

Looking at the R2 eval log headers (not re-scored here), `plot_coop_prompt_baseline.py`
treats R2+Coop-Prompt as a peer of R2+Coop-SDF. The comparison is internal to
the R2 experiment. Recomputing and re-coding the R2 traces by the Step-3 rubric
was not done in this deep-dive because (a) the R2 organism differs from
mo-posttrained and (b) the coding budget was allocated to the 04-20 cells.

## Action item for Jasmine

If a coop-prompt mo-posttrained arm is worth running for the paper, add a new
AM config `qwen3_mo_posttrained_coop_prompt_bare/safety_eval/af.yaml` that
keeps the coop-prompt wording identical to the R2 arm but points at the
mo-posttrained base (no LoRA). Then re-run Step 3 coding on the resulting
aware+harmful traces and compare the `mechanism_code` distribution to the
coop_full SDF arm coded in `2026-04-20_manual_codes.jsonl`.
