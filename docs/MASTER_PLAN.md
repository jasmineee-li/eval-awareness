# Eval Awareness — Master Plan

## Research Goals

- Understand when and how AI models detect that they are being evaluated
- Build a unified testbed for measuring eval awareness across models, eval types, and detection methods
- Develop both blackbox (prompting-based) and whitebox (probe-based) detection approaches
- Study the relationship between eval awareness and downstream behaviors (sandbagging, deception, alignment faking)

## Current Workstreams

### Eval Awareness Testbed (`eval-awareness-testbed/`)
- [x] Unified judges: binary_third_person, probability_third_person, cot_monitoring, purpose_xml, purpose_continue, verbalized_awareness
- [x] Unified evals: needham, agent environments (chess, coding, sandbagging, tic-tac-toe), GDM stealth
- [x] Experiment runner with config-driven orchestration
- [x] Pipeline system for train → eval cycles
- [ ] Whitebox probe integration into experiment runner
- [ ] Cross-model comparison analysis tooling

### Whitebox Probes (`eval-awareness-testbed/probes/`)
- [x] Contrastive probe training on activation data
- [x] Multi-dataset evaluation (gsm8k, gpqa, wmdp, sandbagging, needham, etc.)
- [ ] Probe transfer across model families
- [ ] Integration with pipeline system for automated probe training + eval

### Cross-Subproject Experiments (root `main.py`)
- [x] Unified dispatcher for whitebox train/eval, blackbox needham/cot/trigger/agentic
- [ ] Automated multi-model sweeps
- [ ] Results aggregation across subprojects

### Supplementary Subprojects
- `cot-interventions/` — Chain-of-thought intervention experiments
- `Test_Awareness_Steering/` — Trigger-based eval awareness steering
- `agentic-misalignment/` — Agentic misalignment detection
- `sleeper-agents/` — Sleeper agent training experiments
- `false-facts/` — Synthetic document finetuning
- `impossiblebench/` — Test case exploitation detection

## Conventions

- **Single entrypoint (dispatcher)**: `uv run python main.py --config configs/run_config.yaml`
- **Testbed CLI**: `uv run eat <command>` (experiment, judge, list, etc.)
- **Tests**: `uv run pytest` (from repo root)
- **Configs**: YAML configs in `configs/` (root) and `eval-awareness-testbed/configs/` (testbed)
- **Results**: Timestamped output directories under `results/`
