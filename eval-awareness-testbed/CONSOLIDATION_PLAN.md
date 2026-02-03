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
├── needham-eval/               # Paper reproduction (separate)
├── igor-judging/               # Legacy judges (being deprecated)
├── impossiblebench/            # Test exploitation (separate)
└── inspect_evals/              # Community benchmarks (external reference)
```

### Problems with Current Structure

1. **Code Scatter**: Related functionality spread across 7+ directories
2. **Duplicate Code**: Judge implementations in `igor-judging/`, `inspect_evals/`, and `eval-awareness-testbed/`
3. **Unclear Dependencies**: Symlinks and cross-references make dependencies opaque
4. **No Central Entry Point**: Different CLIs for different components (`eat`, `inspect eval`, `python run.py`)
5. **Model Organism Training Fragmented**: Split between `false-facts/` and `model-organism-consistency-training/`

---

## Proposed Architecture

### Four Core Domains

Based on the codebase analysis, there are **four distinct functional areas**:

| Domain | Purpose | Current Location |
|--------|---------|-----------------|
| **Model Organisms** | Training data generation, consistency training, checkpoints | `false-facts/`, `model-organism-consistency-training/` |
| **Evals** | Tasks that may induce eval awareness | `agent-interp-envs/`, `needham-eval/`, `impossiblebench/`, scattered |
| **Judges** | Detect awareness in transcripts | `igor-judging/`, `eval-awareness-testbed/judges/` |
| **Analysis** | Classify reasoning, visualize results | `eval-awareness-testbed/analyzers/`, `eval-awareness-testbed/analysis/` |

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
│   ├── model_organisms/            # NEW: Training & checkpoints
│   │   ├── __init__.py
│   │   ├── training/               # From model-organism-consistency-training
│   │   │   ├── config.py
│   │   │   ├── pipeline.py         # 5-phase pipeline
│   │   │   ├── data_loading.py
│   │   │   ├── dataset_creation.py
│   │   │   ├── inference.py
│   │   │   └── judges.py           # Awareness detection for training
│   │   ├── synthetic_data/         # From false-facts
│   │   │   ├── synth_doc_generation.py
│   │   │   ├── synth_qa_generation.py
│   │   │   ├── universe_generation/
│   │   │   └── prompts/
│   │   └── checkpoints/            # Model checkpoint management
│   │       ├── registry.py
│   │       └── loader.py
│   │
│   ├── evals/                      # Evaluations that may induce awareness
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   │
│   │   ├── needham/                # Paper dataset (from needham-eval)
│   │   │   ├── eval.py
│   │   │   └── pipeline.py
│   │   │
│   │   ├── agent_envs/             # Docker environments (from agent-interp-envs)
│   │   │   ├── __init__.py
│   │   │   ├── eval.py             # Integration wrapper
│   │   │   ├── providers/          # LLM providers
│   │   │   │   ├── base.py
│   │   │   │   ├── anthropic_provider.py
│   │   │   │   ├── openai_provider.py
│   │   │   │   └── openrouter_provider.py
│   │   │   ├── environments/       # Task environments
│   │   │   │   ├── games/          # Chess, TicTacToe
│   │   │   │   ├── coding/         # Algorithm implementation
│   │   │   │   └── sandbagging/    # Performance manipulation
│   │   │   └── tool_calling.py
│   │   │
│   │   ├── gdm_stealth/            # GDM stealth evals
│   │   │   └── eval.py
│   │   │
│   │   ├── impossiblebench/        # Test exploitation (from impossiblebench)
│   │   │   ├── eval.py
│   │   │   └── analysis/
│   │   │
│   │   └── single_turn/            # Single-turn evals
│   │       ├── base.py
│   │       └── blackmail.py
│   │
│   ├── judges/                     # Detect awareness in transcripts
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── prompts/
│   │   │   └── *.txt
│   │   │
│   │   ├── verbalized_awareness.py # Detailed evidence extraction
│   │   ├── binary_mcq.py           # Yes/No classification
│   │   ├── binary_third_person.py  # Third-person binary
│   │   ├── probability_third_person.py  # Probability estimation
│   │   └── purpose.py              # Purpose classification
│   │
│   ├── analyzers/                  # Classify model explanations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── reasoning_classifier.py
│   │   └── flag_detector.py
│   │
│   ├── analysis/                   # Visualization & reporting
│   │   ├── __init__.py
│   │   ├── plotting.py
│   │   └── reports.py
│   │
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── json_parsing.py
│       ├── model_utils.py
│       └── eval_log_parser.py
│
├── configs/                        # Experiment configurations
│   ├── experiments/
│   ├── training/                   # NEW: Training configs
│   └── evals/
│
├── data/                           # Datasets (symlinks or actual)
│   ├── needham/
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
├── scripts/                        # Utility scripts
│   ├── build_containers.sh
│   └── run_training.py
│
└── results/                        # Output directory
```

---

## Migration Plan

### Phase 1: Consolidate Judges (Priority: HIGH)

**Goal**: Single source of truth for all judge implementations.

**Status**: Partially done per `CLEANUP_PLAN.md`

**Actions**:
1. ~~Migrate judge code from `igor-judging/` to testbed~~ (Done)
2. Add deprecation warnings in `igor-judging/inspect_tasks/_compat.py`
3. Update `inspect_evals/.../awareness_judge/` to import from testbed
4. Delete duplicate prompt files

**Files to modify**:
- `igor-judging/inspect_tasks/_compat.py` - Add re-exports
- `eval-awareness-testbed/src/.../judges/` - Ensure all judges present

### Phase 2: Absorb Agent Environments (Priority: HIGH)

**Goal**: Move `agent-interp-envs/` into testbed as `evals/agent_envs/`.

**Current integration**: Testbed already references via `agent:chess`, `agent:tictactoe`, etc.

**Actions**:
1. Move `agent-interp-envs/src/agent_interp_envs/` to `testbed/src/.../evals/agent_envs/`
2. Move `agent-interp-envs/environments/` to `testbed/environments/`
3. Update imports throughout
4. Keep `agent-interp-envs/` as thin wrapper for backward compatibility
5. Update `build_container.sh` path references

**Dependency analysis**:
```
agent-interp-envs depends on:
  - Docker (external)
  - OmegaConf (pip)
  - LLM providers (internal)

Used by:
  - eval-awareness-testbed (as agent:* evals)
```

### Phase 3: Consolidate Model Organism Training (Priority: MEDIUM)

**Goal**: Unify `false-facts/` and `model-organism-consistency-training/` under `model_organisms/`.

**Actions**:
1. Create `src/eval_awareness_testbed/model_organisms/` directory
2. Move consistency training code:
   ```
   model-organism-consistency-training/src/consistency_training/
   → testbed/src/.../model_organisms/training/
   ```
3. Move synthetic data generation:
   ```
   false-facts/false_facts/synth_*.py
   → testbed/src/.../model_organisms/synthetic_data/
   ```
4. Create unified CLI commands:
   ```bash
   eat train prepare-data ...
   eat train run-pipeline ...
   eat train generate-synthetic ...
   ```
5. Keep `false-facts/` and `model-organism-consistency-training/` as thin wrappers

**Note**: `false-facts/` has heavy dependencies (torch, transformers). Consider:
- Option A: Include in testbed as optional dependency
- Option B: Keep as separate installable package, import dynamically

### Phase 4: Absorb Needham Eval (Priority: MEDIUM)

**Goal**: Move `needham-eval/` core functionality into testbed.

**Current integration**: Testbed already has `evals/needham/` that uses needham-eval data.

**Actions**:
1. Move `needham-eval/pipeline/` to `testbed/src/.../evals/needham/`
2. Move `needham-eval/cot_classifiers/` to `testbed/src/.../analyzers/`
3. Keep data in `needham-eval/data_repo/` (symlinked from testbed)
4. Update CLI to support needham-specific commands:
   ```bash
   eat eval needham --stage eval_mcq --limit 1000
   eat plot needham --type roc
   ```

### Phase 5: Absorb ImpossibleBench (Priority: LOW)

**Goal**: Add impossiblebench as eval type in testbed.

**Actions**:
1. Create `testbed/src/.../evals/impossiblebench/`
2. Move core analysis code
3. Register as eval:
   ```bash
   eat eval impossible:livecodebench --split conflicting
   eat eval impossible:swebench --agent full
   ```

### Phase 6: Deprecate Legacy Directories (Priority: LOW)

**Goal**: Mark old directories as deprecated, keep for backward compatibility.

**Actions**:
1. Add `DEPRECATED.md` to each absorbed directory
2. Update imports to show deprecation warnings
3. After transition period (e.g., 3 months), consider removal

---

## CLI Unification

### Current CLIs

| Directory | Command | Purpose |
|-----------|---------|---------|
| eval-awareness-testbed | `eat` | Evals, judges, analyzers |
| agent-interp-envs | `python scripts/run.py` | Run environments |
| needham-eval | `python run.py` | Paper reproduction |
| model-organism | `python scripts/*.py` | Training pipeline |
| false-facts | `uv run false_facts/...` | Doc generation |

### Target CLI

Unified `eat` command:

```bash
# Evals
eat eval needham --model gpt-4o --limit 100
eat eval agent:chess --model claude-sonnet-4
eat eval impossible:livecodebench --split conflicting

# Judges
eat judge transcript.json --methods verbalized_awareness
eat judge-batch results/ --methods all

# Analyzers
eat analyze "explanation text" --methods reasoning_classifier

# Model Organisms
eat organisms train --config configs/training/consistency.yaml
eat organisms generate-docs --universe universe.json --output docs/
eat organisms checkpoint list
eat organisms checkpoint load my-checkpoint

# Experiments
eat experiment configs/experiments/full.yaml
eat experiment --dry-run configs/experiments/test.yaml

# Analysis & Reporting
eat plot roc --input results/
eat report generate --experiment results/2024-01-01/

# Utilities
eat list                    # List all evals, judges, analyzers
eat info agent:chess        # Show details about specific component
```

---

## Dependency Management

### Current Package Structure

Each directory has its own `pyproject.toml`:
- `eval-awareness-testbed`: Python 3.11+, inspect-ai, typer
- `agent-interp-envs`: Python 3.11+, omegaconf
- `false-facts`: Python 3.10+, torch, transformers (heavy)
- `model-organism-consistency-training`: Python 3.10+, safety-tooling
- `needham-eval`: Python 3.10+, inspect-ai

### Target Package Structure

Single `pyproject.toml` with optional dependencies:

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
]

[project.optional-dependencies]
training = [
    "torch>=2.0",
    "transformers>=4.35",
    "trl>=0.7",
    "unsloth",
]
agents = [
    "docker>=6.0",
]
all = [
    "eval-awareness-testbed[training,agents]",
]
dev = [
    "pytest>=7.0",
    "ruff>=0.1",
]

[project.scripts]
eat = "eval_awareness_testbed.cli:app"
```

Installation:
```bash
pip install -e .                    # Core functionality
pip install -e ".[agents]"          # + Docker environments
pip install -e ".[training]"        # + Model training
pip install -e ".[all]"             # Everything
```

---

## Data Management

### Current Data Locations

- `needham-eval/data_repo/` - Needham dataset
- `model-organism-consistency-training/data/` - Training data, splits, inference results
- `false-facts/data/` - Generated documents
- `agent-interp-envs/results/` - Environment run results
- `eval-awareness-testbed/data/` - Symlinks

### Target Data Structure

```
eval-awareness-testbed/
├── data/
│   ├── needham/              # Symlink to needham-eval/data_repo
│   ├── training/
│   │   ├── splits/           # Train/val/test splits
│   │   ├── inference/        # Inference results
│   │   └── datasets/         # SFT/DPO datasets
│   └── synthetic/
│       ├── universes/        # Universe definitions
│       └── documents/        # Generated documents
```

---

## Implementation Priority

| Phase | Priority | Effort | Impact | Dependencies |
|-------|----------|--------|--------|--------------|
| 1. Consolidate Judges | HIGH | Low | High | None |
| 2. Absorb Agent Envs | HIGH | Medium | High | Phase 1 |
| 3. Model Organism Training | MEDIUM | High | Medium | Phase 1, 2 |
| 4. Absorb Needham | MEDIUM | Low | Medium | Phase 1 |
| 5. Absorb ImpossibleBench | LOW | Low | Low | Phase 1, 2 |
| 6. Deprecate Legacy | LOW | Low | Medium | All above |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing scripts | High | Maintain backward-compatible wrappers |
| Heavy dependencies (torch) | Medium | Use optional dependencies |
| Docker path changes | Medium | Update build scripts, document changes |
| Lost commit history | Low | Use `git mv` for moves, document in commit messages |

---

## Success Criteria

1. **Single CLI**: All functionality accessible via `eat` command
2. **Single Install**: `pip install -e ".[all]"` installs everything
3. **Clear Structure**: New contributors can find code within 2 minutes
4. **No Duplicates**: Each piece of functionality exists in exactly one place
5. **Backward Compatible**: Old import paths work with deprecation warnings

---

## Appendix: File Movement Summary

### Files to Move

| From | To | Notes |
|------|-----|-------|
| `agent-interp-envs/src/agent_interp_envs/` | `testbed/src/.../evals/agent_envs/` | Core agent code |
| `agent-interp-envs/environments/` | `testbed/environments/` | Docker contexts |
| `model-organism.../src/consistency_training/` | `testbed/src/.../model_organisms/training/` | Training pipeline |
| `false-facts/false_facts/synth_*.py` | `testbed/src/.../model_organisms/synthetic_data/` | Doc generation |
| `needham-eval/pipeline/` | `testbed/src/.../evals/needham/` | Eval pipeline |
| `impossiblebench/src/impossiblebench/` | `testbed/src/.../evals/impossiblebench/` | Exploit detection |

### Files to Delete (After Migration)

- `igor-judging/inspect_tasks/*.py` (except `_compat.py`)
- Duplicate prompts across directories

### Files to Keep as Wrappers

Each absorbed directory keeps a thin wrapper:

```python
# agent-interp-envs/src/agent_interp_envs/__init__.py
import warnings
warnings.warn("Import from eval_awareness_testbed.evals.agent_envs instead", DeprecationWarning)
from eval_awareness_testbed.evals.agent_envs import *
```

---

## Next Steps

1. Review and approve this plan
2. Create feature branch for each phase
3. Implement Phase 1 (judges consolidation)
4. Test backward compatibility
5. Proceed with remaining phases

---

*Last updated: 2026-02-03*
*Author: Claude (automated analysis)*
