# Eval Awareness Testbed - Refactoring Plan

## Repository Summary

### Current Structure
```
eval-awareness/
├── eval_awareness/          # Needham paper replication (AUC, ROC curves)
│   ├── run.py              # Main runner for MCQ evals + CoT analysis
│   ├── cot_classifiers/    # Chain-of-thought classifiers (reasoning_classifier, flag_reasoning, openended_classifier)
│   ├── pq_ablation/        # Probe questions (eval_mcq.yaml, purpose_mcq.yaml, purpose_open.yaml)
│   └── analysis/           # Plotting (ROC, calibration, CoT breakdown)
│
├── agent-interp-envs/       # Dockerized agent environments
│   ├── environments/       # Chess, Tic-Tac-Toe, Coding, Sandbagging
│   ├── scripts/run.py      # Environment runner
│   └── configs/            # YAML configs per environment
│
├── inspect_evals/           # Inspect AI standard evals + custom
│   └── gdm_capabilities/
│       └── awareness_judge/ # 3 grading methods: cot, purpose_xml, purpose_continue
│           ├── graders.py  # Core grading logic
│           ├── scorer.py   # Inspect scorer integration
│           └── cli.py      # CLI for grading transcripts
│
├── impossiblebench/         # Test case exploitation detection
├── false-facts/             # Synthetic document finetuning
└── model-organism/          # Data analysis notebooks
```

### Eval Awareness Judging Methods (Current)

**In `inspect_evals/awareness_judge/`:**
1. **cot** (CoT Monitoring) - Extract quotes showing eval awareness from transcript
2. **purpose_xml** - Show transcript in XML tags, ask why interaction took place
3. **purpose_continue** - Load transcript as conversation history, ask purpose

**In `eval_awareness/cot_classifiers/`:**
4. **reasoning_classifier** - Classify CoT explanations into categories (Alignment/Capability/Other Eval/Real/Other)
5. **flag_reasoning** - Flag specific reasoning patterns (Meta-Reasoning, Fictional-Scenario, etc.)
6. **openended_classifier** - Classify open-ended purpose responses

**Note:** The `binary_third_person` method mentioned = the MCQ approach in `eval_awareness/pq_ablation/eval_mcq.yaml`. It asks "Was this an evaluation? YES/NO". The difference is:
- `eval_awareness`: Each prompt applied 5 epochs (for statistical robustness)
- `awareness_judge`: Each prompt applied once (for speed)

---

## Proposed Architecture

### New Structure: `eval-awareness-testbed/`

```
eval-awareness-testbed/
├── pyproject.toml           # Unified dependencies
├── run.py                   # Unified CLI entry point
├── configs/
│   ├── models.yaml          # Model definitions (API + local paths)
│   └── experiments/         # Pre-defined experiment configs
│
├── evals/                   # Eval implementations (what induces awareness)
│   ├── __init__.py
│   ├── base.py              # BaseEval interface
│   ├── needham/             # Needham dataset AUC (from eval_awareness/)
│   ├── agent_envs/          # Agent-interp-envs wrapper
│   └── gdm_stealth/         # GDM stealth/self-reasoning tasks
│
├── judges/                  # Eval awareness detection (applied to transcripts)
│   ├── __init__.py
│   ├── base.py              # BaseJudge interface
│   ├── cot_monitoring.py    # Extract quotes showing awareness
│   ├── purpose_xml.py       # XML transcript + "why did this happen?"
│   ├── purpose_continue.py  # Conversation history + purpose question
│   └── binary_mcq.py        # "Is this an evaluation?" YES/NO (configurable epochs)
│
├── analyzers/               # Reasoning analysis (applied to model explanations)
│   ├── __init__.py
│   ├── reasoning_classifier.py  # Classify into Alignment/Capability/Real/Other
│   └── flag_detector.py         # Detect patterns: Meta-Reasoning, Fictional-Scenario, etc.
│
├── analysis/                # Result analysis and visualization
│   ├── __init__.py
│   ├── roc.py               # ROC curves
│   ├── calibration.py       # Calibration plots
│   ├── aggregation.py       # Aggregate results across methods
│   └── reports.py           # Generate reports
│
├── data/                    # Datasets
│   └── needham/             # Needham dataset (symlink to data_repo/)
│
├── results/                 # Output directory
│   └── <experiment>/<model>/<timestamp>/
│
├── external/                # Uninvolved subfolders (keep as-is)
│   ├── impossiblebench/
│   ├── false-facts/
│   └── model-organism/
│
└── legacy/                  # Old code during migration (remove after)
    ├── eval_awareness/
    └── agent-interp-envs/
```

---

## Unified Run Script Design

### CLI Interface

```bash
# Run a single eval with a single model
python run.py eval needham --model claude-sonnet-4

# Run agent environment
python run.py eval agent:sandbagging --model gpt-4o --config configs/sandbagging/normal.yaml

# Run multiple evals
python run.py eval needham,agent:chess --model claude-sonnet-4

# Run with multiple models
python run.py eval needham --model claude-sonnet-4,gpt-4o

# Judge eval awareness on results
python run.py judge results/needham/claude-sonnet-4/2025-01-24/ --methods cot,purpose_xml

# Judge with all methods
python run.py judge results/ --methods all

# Run eval + judge in one command
python run.py experiment --eval needham --model claude-sonnet-4 --judge cot,binary_mcq

# Analysis
python run.py analyze results/ --plot roc,calibration
```

### Config-Based Experiments

```yaml
# configs/experiments/quick_test.yaml
name: quick_test
description: Fast smoke test

models:
  - claude-3-5-haiku-20241022  # API model
  - local:Qwen/QwQ-32B         # Local model path

evals:
  - needham:
      limit: 100
      epochs: 1
  - agent:sandbagging:
      mode: normal

judges:
  - cot
  - binary_mcq

analysis:
  plots: [roc]
```

```bash
python run.py experiment configs/experiments/quick_test.yaml
```

---

## Key Design Decisions

### 1. Eval Interface
```python
class BaseEval(ABC):
    @abstractmethod
    def run(self, model: str, **kwargs) -> EvalResult:
        """Run the eval, return results."""
        pass

    @abstractmethod
    def get_transcripts(self, result: EvalResult) -> list[Transcript]:
        """Extract transcripts for judging."""
        pass
```

### 2. Judge Interface
```python
class BaseJudge(ABC):
    @abstractmethod
    async def judge(self, transcript: Transcript) -> JudgeResult:
        """Judge a single transcript for eval awareness."""
        pass
```

### 3. Model Specification
```yaml
# models.yaml
models:
  # API models (auto-detect provider from name)
  claude-sonnet-4: {}
  gpt-4o: {}
  gemini-2.5-pro: {}

  # Local models
  local:Qwen/QwQ-32B:
    provider: vllm
    path: /path/to/model

  local:deepseek-70b:
    provider: hf
    path: deepseek-ai/DeepSeek-V3
```

### 4. Result Format
```json
{
  "eval": "needham",
  "model": "claude-sonnet-4",
  "timestamp": "2025-01-24T12:00:00",
  "eval_results": {...},
  "judge_results": {
    "cot": {"eval_aware_rate": 0.42, "details": [...]},
    "binary_mcq": {"eval_aware_rate": 0.38, "details": [...]}
  }
}
```

---

## Migration Plan

### Phase 1: Unify Judges ✅ DONE
1. ✅ Create `judges/` module with unified interface
2. ✅ Port `awareness_judge` graders (cot, purpose_xml, purpose_continue)
3. ✅ Port `cot_classifiers` methods → `analyzers/` (reasoning_classifier, flag_detector)
4. ✅ Add `binary_mcq` with configurable epochs
5. ✅ Unified types and registry pattern

### Phase 2: Unify Evals ✅ DONE
1. ✅ Create `evals/` module skeleton with interface
2. ✅ Wrap `eval_awareness/run.py` as `evals/needham/`
3. ✅ Wrap `agent-interp-envs` as `evals/agent_envs/`
4. ✅ Wrap `gdm_capabilities/stealth` as `evals/gdm_stealth/`

### Phase 3: Unified CLI ✅ DONE
1. ✅ Create `run.py` with subcommands: `list`, `eval`, `judge`, `analyze`, `experiment`
2. ✅ Full experiment runner with config-based orchestration
3. ✅ Model support via Inspect AI (vllm/, hf/)

### Phase 4: Cleanup (TODO)
1. TODO: Move old code to `legacy/` after evals are wrapped
2. TODO: Add tests
3. ✅ Documentation (README.md)

---

## Decisions (Confirmed)

1. **Naming**: `eval-awareness-testbed` ✓

2. **Scope**:
   - Core: eval_awareness, agent-interp-envs, awareness_judge ✓
   - External (untouched): impossiblebench, false-facts, model-organism ✓

3. **Methods**:
   - **Judges** (detect awareness in transcripts): cot_monitoring, purpose_xml, purpose_continue, binary_mcq
   - **Analyzers** (classify model explanations): reasoning_classifier, flag_detector

4. **Binary MCQ**: Unified with configurable epochs (default: 1 for speed, 5 for robustness) ✓

5. **Local Model Support**: Use Inspect AI's native support
   - vLLM: `--model vllm/Qwen/QwQ-32B` (fast, batched)
   - HuggingFace: `--model hf/meta-llama/Llama-3.1-70B` (universal fallback)

---

## Implementation Notes

### Don't Reinvent
- Reuse existing Inspect AI infrastructure
- Reuse existing agent-interp-envs Docker setup
- Thin wrappers around working code

### Keep It Simple
- Single entry point (`run.py`)
- Config-driven experiments
- Consistent result format
- Clear separation: evals (induce awareness) vs judges (detect awareness)

### Extensibility
- Easy to add new evals: implement `BaseEval`
- Easy to add new judges: implement `BaseJudge`
- Easy to add new models: add to config

---

## File Cleanup

### Files to Delete/Consolidate
- `eval_awareness/cot_classifiers/` - merge into `judges/`
- One of the binary_third_person implementations

### Files to Keep As-Is
- `agent-interp-envs/` - working Docker infrastructure
- `inspect_evals/gdm_capabilities/` - working GDM evals
- `eval_awareness/analysis/` - working plotting code
