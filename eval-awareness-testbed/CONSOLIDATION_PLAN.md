# Eval Awareness Testbed Consolidation Plan

## Executive Summary

This document proposes consolidating the scattered `eval-awareness` codebase into a unified `eval-awareness-testbed/` structure. The goal is to make the testbed the **single source of truth** for all eval awareness research components.

---

## Current State Analysis

### Directory Structure (Before)

```
eval-awareness/
├── eval-awareness-testbed/     # Central hub (partial)
├── agent-interp-envs/          # Docker environments (separate)
├── false-facts/                # Synthetic doc generation (separate)
├── model-organism-consistency-training/  # Consistency training (separate)
├── needham-eval/               # Paper reproduction / judging dataset (separate)
├── igor-judging/               # Inspect AI judges (being consolidated)
└── inspect_evals/              # Community benchmarks (external reference)
```

### Problems with Current Structure

1. **Code Scatter**: Related functionality spread across 6+ directories
2. **Duplicate Code**: Judge implementations in `igor-judging/`, `inspect_evals/`, and `eval-awareness-testbed/`
3. **Unclear Dependencies**: Symlinks and cross-references make dependencies opaque
4. **No Central Entry Point**: Different CLIs for different components (`eat`, `inspect eval`, `python run.py`)
5. **Model Organism Training Fragmented**: Split between `false-facts/` and `model-organism-consistency-training/`
6. **Mixed Frameworks**: Some code uses Inspect AI, some doesn't - should consolidate on Inspect

---

## Key Conceptual Clarification

### Two Types of "Evals"

There's an important distinction between two types of evaluations in this codebase:

#### 1. Content Evals (Tasks That May Induce Awareness)

These are actual tasks given to models that might trigger eval awareness behavior:

| Eval | Description | Location |
|------|-------------|----------|
| `agent:chess` | Play chess against Stockfish | `agent-interp-envs/` |
| `agent:tictactoe` | Play tic-tac-toe | `agent-interp-envs/` |
| `agent:coding` | Implement algorithms | `agent-interp-envs/` |
| `agent:sandbagging` | Deliberately underperform | `agent-interp-envs/` |
| `blackmail` | Agentic misalignment scenario | `eval-awareness-testbed/` |
| `gdm:*` | GDM stealth evals | `eval-awareness-testbed/` |

#### 2. Meta-Evals / Judging Dataset (Testing EVAL/REAL Classification)

The **Needham dataset** is NOT a content eval - it's a **judging benchmark** used to test how well models can classify transcripts as coming from evaluations vs. real interactions.

| Dataset | Purpose | Location |
|---------|---------|----------|
| `needham` | Test EVAL vs REAL transcript classification | `needham-eval/data_repo/` |

### Two Types of Judges

Judges serve different purposes depending on what's being analyzed:

#### For Needham Dataset (EVAL/REAL Classification)

These judges take transcripts and classify whether they came from an evaluation:

| Judge | Description | Output |
|-------|-------------|--------|
| `binary_third_person` | 5 binary prompts asking "Is this an evaluation?" | Vote-based EVAL/REAL |
| `probability_third_person` | 5 prompts estimating probability | 0-1 probability score |

#### For Content Eval Transcripts (Awareness Detection)

These judges analyze transcripts from actual evals to detect if the model showed awareness:

| Judge | Description | Output |
|-------|-------------|--------|
| `verbalized_awareness` | Detect explicit awareness statements | EVAL_AWARE/NOT_EVAL_AWARE with evidence |
| `purpose_continue` | Ask "what's the purpose?" naturally | Purpose guess + classification |
| `purpose_xml` | Third-person purpose analysis | Purpose classification |
| `binary_mcq` | Simple "Is this an evaluation?" YES/NO | Binary classification |

---

## Proposed Architecture

### Three Core Domains

Based on the clarified analysis, there are **three distinct functional areas**:

| Domain | Purpose | Current Location |
|--------|---------|-----------------|
| **Model Organisms** | Training data generation, consistency training, checkpoints | `false-facts/`, `model-organism-consistency-training/` |
| **Content Evals** | Tasks that may induce eval awareness | `agent-interp-envs/`, `single_turn/`, `gdm_stealth/` |
| **Judging** | Detect awareness + classify EVAL/REAL | `igor-judging/`, `needham-eval/`, `eval-awareness-testbed/judges/` |

### Target Directory Structure (After)

```
eval-awareness-testbed/
├── run.py                          # Unified CLI entry point
├── pyproject.toml                  # Single package config
├── README.md
│
├── src/eval_awareness_testbed/
│   ├── __init__.py
│   ├── cli.py                      # `eat` command
│   ├── experiment.py               # ExperimentRunner
│   ├── types.py                    # Shared types
│   │
│   ├── model_organisms/            # Training & checkpoints
│   │   ├── __init__.py
│   │   ├── training/               # From model-organism-consistency-training
│   │   │   ├── config.py
│   │   │   ├── pipeline.py         # 5-phase pipeline
│   │   │   ├── data_loading.py
│   │   │   ├── dataset_creation.py
│   │   │   └── inference.py
│   │   ├── synthetic_data/         # From false-facts
│   │   │   ├── synth_doc_generation.py
│   │   │   ├── synth_qa_generation.py
│   │   │   └── universe_generation/
│   │   └── checkpoints/
│   │       └── registry.py
│   │
│   ├── evals/                      # Content evals (tasks that may induce awareness)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   │
│   │   ├── agent_envs/             # Docker environments (from agent-interp-envs)
│   │   │   ├── __init__.py
│   │   │   ├── eval.py
│   │   │   ├── providers/          # LLM providers
│   │   │   └── environments/       # Chess, TicTacToe, Coding, Sandbagging
│   │   │
│   │   ├── gdm_stealth/            # GDM stealth evals
│   │   │   └── eval.py
│   │   │
│   │   └── single_turn/            # Single-turn evals
│   │       ├── base.py
│   │       └── blackmail.py
│   │
│   ├── judges/                     # ALL judging functionality
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── prompts/
│   │   │
│   │   │   # For content eval transcripts (awareness detection)
│   │   ├── verbalized_awareness.py # Detect explicit awareness
│   │   ├── purpose.py              # Purpose classification
│   │   ├── binary_mcq.py           # Simple YES/NO
│   │   │
│   │   │   # For Needham dataset (EVAL/REAL classification)
│   │   ├── binary_third_person.py  # 5-prompt binary voting
│   │   ├── probability_third_person.py  # 5-prompt probability
│   │   │
│   │   │   # Needham dataset & pipeline (from needham-eval)
│   │   ├── needham/
│   │   │   ├── dataset.py          # Load needham transcripts
│   │   │   ├── pipeline.py         # Run EVAL/REAL experiments
│   │   │   └── plotting.py         # ROC curves, calibration plots
│   │   │
│   │   └── inspect_tasks/          # Inspect AI task wrappers
│   │       ├── binary_third_person_task.py
│   │       ├── probability_third_person_task.py
│   │       └── verbalized_awareness_task.py
│   │
│   ├── analyzers/                  # Classify model explanations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── reasoning_classifier.py
│   │   └── flag_detector.py
│   │
│   ├── analysis/                   # Visualization & reporting
│   │   ├── __init__.py
│   │   ├── plotting.py
│   │   └── reports.py
│   │
│   └── utils/                      # Shared utilities (reuse from safety-tooling)
│       ├── __init__.py
│       ├── json_parsing.py
│       ├── model_utils.py
│       └── eval_log_parser.py
│
├── configs/                        # Experiment configurations
│   ├── experiments/
│   ├── training/
│   └── judges/
│
├── data/
│   ├── needham/                    # Symlink to needham-eval/data_repo
│   ├── training/
│   └── universes/
│
├── environments/                   # Docker build context (from agent-interp-envs)
│   ├── games/
│   │   ├── chess/
│   │   └── tictactoe/
│   ├── coding/
│   └── sandbagging/
│
└── results/
```

---

## Migration Plan

### Phase 1: Consolidate Judges + Needham (Priority: HIGH)

**Goal**: Single source of truth for all judging, including Needham dataset/pipeline.

**Key insight**: Needham is a **judging benchmark**, not a content eval. It belongs under `judges/`.

**Actions**:
1. Move Needham pipeline to `judges/needham/`:
   ```
   needham-eval/pipeline/ → testbed/src/.../judges/needham/
   needham-eval/cot_classifiers/ → testbed/src/.../analyzers/
   ```
2. Keep `igor-judging/` judges as Inspect AI task wrappers
3. Create unified judging interface that works with both:
   - JSON transcript files
   - Inspect AI `.eval` log files
4. Reuse `safety-tooling` utilities where possible

**CLI Commands**:
```bash
# Run judges on content eval transcripts
eat judge transcript.json --methods verbalized_awareness,purpose_continue

# Run EVAL/REAL classification on needham dataset
eat judge needham --methods binary_third_person,probability_third_person --model gpt-4o

# Generate ROC/calibration plots from judging results
eat plot roc --input results/needham/
```

### Phase 2: Consolidate on Inspect AI (Priority: HIGH)

**Goal**: Use Inspect AI as the standard framework for all evaluations and judging.

**Current state**:
- `igor-judging/` uses Inspect AI (`inspect eval inspect_tasks/...`)
- `eval-awareness-testbed/` uses custom CLI (`eat eval/judge`)
- `needham-eval/` uses Inspect AI internally

**Actions**:
1. Wrap all judges as Inspect AI tasks in `judges/inspect_tasks/`
2. Keep `eat` CLI as convenience wrapper that calls `inspect eval` under the hood
3. Support both interfaces:
   ```bash
   # Via eat CLI (convenience)
   eat judge transcript.json --methods verbalized_awareness

   # Via inspect directly (power users)
   inspect eval inspect_tasks/verbalized_awareness.py -T dataset_path=transcript.json
   ```

**Benefits**:
- Standardized logging and result format
- Can input `.eval` files directly
- Integrates with inspect ecosystem

### Phase 3: Absorb Agent Environments (Priority: HIGH)

**Goal**: Move `agent-interp-envs/` into testbed as `evals/agent_envs/`.

**Actions**:
1. Move `agent-interp-envs/src/agent_interp_envs/` to `testbed/src/.../evals/agent_envs/`
2. Move `agent-interp-envs/environments/` to `testbed/environments/`
3. Keep `agent-interp-envs/` as thin backward-compatibility wrapper

### Phase 4: Consolidate Model Organism Training (Priority: MEDIUM)

**Goal**: Unify `false-facts/` and `model-organism-consistency-training/` under `model_organisms/`.

**Actions**:
1. Create `src/eval_awareness_testbed/model_organisms/` directory
2. Move consistency training code
3. Move synthetic data generation
4. Reuse `safety-tooling` for inference utilities

**Note**: Heavy dependencies (torch, transformers) should be optional.

### Phase 5: Deprecate Legacy Directories (Priority: LOW)

**Goal**: Mark old directories as deprecated, keep for backward compatibility.

**Actions**:
1. Add `DEPRECATED.md` to each absorbed directory
2. Update imports to show deprecation warnings
3. After transition period, consider removal

---

## Inspect AI Integration Strategy

### Current State

`igor-judging/` already implements judges as Inspect AI tasks:

```python
# igor-judging/inspect_tasks/binary_third_person.py
@task
def eval_awareness_task(dataset_path: str) -> Task:
    dataset = load_dataset(dataset_path)
    return Task(
        dataset=dataset,
        solver=[transcript_passthrough()],
        scorer=eval_awareness_scorer(),
    )
```

### Target State

All judges should have both:
1. **Standalone function**: For programmatic use
2. **Inspect AI task**: For CLI and logging

```python
# testbed/src/.../judges/binary_third_person.py

# Standalone judge class (current testbed implementation)
@register_judge("binary_third_person")
class BinaryThirdPersonJudge(BaseJudge):
    async def judge(self, transcript: Transcript) -> JudgeResult:
        ...

# Inspect AI task wrapper
# testbed/src/.../judges/inspect_tasks/binary_third_person_task.py
@task
def binary_third_person_task(dataset_path: str) -> Task:
    from eval_awareness_testbed.judges import BinaryThirdPersonJudge
    ...
```

### CLI Mapping

| `eat` command | `inspect` equivalent |
|---------------|---------------------|
| `eat judge X --methods binary_third_person` | `inspect eval inspect_tasks/binary_third_person.py -T dataset_path=X` |
| `eat eval needham --model M` | `inspect eval inspect_tasks/needham_eval.py --model M` |

---

## Reusing safety-tooling

The `safety-tooling` package (external dependency) provides utilities that should be reused:

| Utility | Use |
|---------|-----|
| API rate limiting | All LLM calls |
| Response caching | Avoid re-running expensive inference |
| Model provider wrappers | Consistent interface |

**Integration**:
```python
# Instead of custom implementations
from safety_tooling import RateLimiter, ResponseCache
```

---

## CLI Unification

### Target CLI

```bash
# Content Evals (tasks that may induce awareness)
eat eval agent:chess --model claude-sonnet-4
eat eval agent:sandbagging --model gpt-4o
eat eval blackmail --model qwen-qwq-32b

# Judging (awareness detection on content eval transcripts)
eat judge transcript.json --methods verbalized_awareness,purpose_continue
eat judge results/*.json --methods all --output summary.json

# Judging (EVAL/REAL classification on needham dataset)
eat judge needham --methods binary_third_person --model gpt-4o --limit 100
eat judge needham --methods probability_third_person --output probs.json

# Analysis & Plotting
eat plot roc --input results/needham/
eat plot calibration --input results/needham/
eat analyze "model explanation text" --methods reasoning_classifier

# Model Organisms
eat organisms train --config configs/training/consistency.yaml
eat organisms generate-docs --universe universe.json

# Experiments (full pipeline)
eat experiment configs/experiments/full.yaml

# Utilities
eat list                    # List all evals, judges, analyzers
eat info agent:chess        # Show details about specific component
```

---

## Dependency Management

### Target Package Structure

```toml
[project]
name = "eval-awareness-testbed"
requires-python = ">=3.11"

dependencies = [
    "inspect-ai>=0.3",
    "typer>=0.9",
    "pydantic>=2.0",
    "omegaconf>=2.3",
    "rich>=13.0",
    # Reuse safety-tooling utilities
    # "safety-tooling @ git+https://...",
]

[project.optional-dependencies]
training = [
    "torch>=2.0",
    "transformers>=4.35",
    "trl>=0.7",
]
agents = [
    "docker>=6.0",
]
all = [
    "eval-awareness-testbed[training,agents]",
]

[project.scripts]
eat = "eval_awareness_testbed.cli:app"
```

---

## Implementation Priority

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| 1. Consolidate Judges + Needham | HIGH | Medium | High |
| 2. Consolidate on Inspect AI | HIGH | Medium | High |
| 3. Absorb Agent Envs | HIGH | Medium | High |
| 4. Model Organism Training | MEDIUM | High | Medium |
| 5. Deprecate Legacy | LOW | Low | Medium |

---

## Success Criteria

1. **Single CLI**: All functionality accessible via `eat` command
2. **Single Install**: `pip install -e ".[all]"` installs everything
3. **Inspect AI Compatible**: All judges runnable via `inspect eval`
4. **Clear Distinction**: Content evals vs. judging/meta-evals clearly separated
5. **No Duplicates**: Each piece of functionality exists in exactly one place
6. **Reuses Existing Code**: Leverages `safety-tooling` and `igor-judging` Inspect implementations

---

## Appendix: File Movement Summary

### Files to Move

| From | To | Notes |
|------|-----|-------|
| `needham-eval/pipeline/` | `testbed/src/.../judges/needham/` | Judging benchmark, not content eval |
| `needham-eval/cot_classifiers/` | `testbed/src/.../analyzers/` | Reasoning analysis |
| `igor-judging/inspect_tasks/*.py` | `testbed/src/.../judges/inspect_tasks/` | Inspect AI task wrappers |
| `agent-interp-envs/src/` | `testbed/src/.../evals/agent_envs/` | Content evals |
| `agent-interp-envs/environments/` | `testbed/environments/` | Docker contexts |
| `model-organism.../src/` | `testbed/src/.../model_organisms/training/` | Training pipeline |
| `false-facts/false_facts/synth_*.py` | `testbed/src/.../model_organisms/synthetic_data/` | Doc generation |

### Directories Removed

- `impossiblebench/` - No longer needed (deleted)

---

*Last updated: 2026-02-03*
*Author: Claude (automated analysis)*
