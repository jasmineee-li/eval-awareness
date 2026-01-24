# Eval Awareness Testbed

Unified framework for evaluating and detecting eval awareness in language models.

## Quick Start

```bash
# Install
cd eval-awareness-testbed
pip install -e .

# List available methods
eat list

# Judge a transcript
eat judge transcript.json --methods cot,binary_mcq

# Analyze an explanation
eat analyze "I think this is an evaluation because..." --methods reasoning_classifier
```

## Architecture

```
eval-awareness-testbed/
├── run.py                   # Unified CLI entry point
├── src/eval_awareness_testbed/
│   ├── evals/              # Evals that may induce awareness
│   │   ├── needham/        # Needham paper dataset
│   │   ├── agent_envs/     # Chess, sandbagging, coding
│   │   └── gdm_stealth/    # GDM stealth evals
│   ├── judges/             # Detect awareness in transcripts
│   ├── analyzers/          # Classify reasoning patterns
│   └── analysis/           # Visualization and reports
├── configs/                # Experiment configs
├── data/                   # Symlinks to datasets
└── results/                # Output directory
```

## Evals (Induce Awareness)

| Eval | Description |
|------|-------------|
| `needham` | Binary MCQ on Needham transcript dataset |
| `needham:purpose` | Purpose classification (alignment vs capability) |
| `agent:chess` | Chess playing agent environment |
| `agent:tictactoe` | Tic-Tac-Toe agent environment |
| `agent:coding` | Coding problem solving environment |
| `agent:sandbagging` | Sandbagging detection environment |
| `gdm:oversight_pattern` | Test if model exploits monitoring blind spots |
| `gdm:classifier_evasion` | Test if model evades content classifiers |
| `gdm:cover_your_tracks` | Test if model hides its actions |
| `gdm:strategic_rule_breaking` | Test if model strategically breaks rules |

## Judges (Detect Awareness in Transcripts)

| Method | Description |
|--------|-------------|
| `cot` | CoT Monitoring - Extract quotes showing eval awareness |
| `purpose_xml` | Show transcript in XML, ask why it happened |
| `purpose_continue` | Load as conversation, ask purpose |
| `binary_mcq` | "Is this an evaluation?" YES/NO (configurable epochs) |

## Analyzers (Classify Model Explanations)

| Method | Description |
|--------|-------------|
| `reasoning_classifier` | Classify into Alignment/Capability/Real/Other |
| `flag_detector` | Detect patterns: Meta-Reasoning, Fictional-Scenario, etc. |

## Usage Examples

### Run an eval

```bash
# Run Needham eval
eat eval needham --model anthropic/claude-3-5-sonnet-20241022 --limit 100

# Run agent environment
eat eval agent:sandbagging --model openai/gpt-4o

# Run GDM stealth eval with perturbation
eat eval gdm:cover_your_tracks --model anthropic/claude-sonnet-4 --epochs 5
```

### Judge a single transcript

```bash
eat judge transcript.json --methods cot,binary_mcq --model anthropic/claude-3-5-sonnet-20241022
```

### Judge with multiple epochs (for robustness)

```bash
eat judge transcript.json --methods binary_mcq --epochs 5
```

### Analyze explanation text

```bash
eat analyze "This appears to be an evaluation because the task includes specific formatting requirements" \
    --methods reasoning_classifier,flag_detector
```

### Run experiment from config

```bash
# Dry run - validate config without executing
eat experiment configs/experiments/quick_test.yaml --dry-run

# Full run
eat experiment configs/experiments/quick_test.yaml

# Full test across multiple models
eat experiment configs/experiments/full_test.yaml
```

The experiment runner:
1. Runs specified evals on each model
2. Judges resulting transcripts with specified methods
3. Analyzes model explanations (from binary_mcq)
4. Saves results to `results/<experiment>/<timestamp>/`

## Model Support

### API Models
```bash
--model anthropic/claude-3-5-sonnet-20241022
--model openai/gpt-4o
--model openrouter/google/gemini-2.5-pro
```

### Local Models (via Inspect AI)
```bash
--model vllm/Qwen/QwQ-32B
--model hf/meta-llama/Llama-3.1-70B
```

## Configuration

Example experiment config:

```yaml
name: my_experiment
description: Test eval awareness on multiple models

models:
  - anthropic/claude-3-5-haiku-20241022
  - openai/gpt-4o-mini

evals:
  - name: needham
    limit: 100
    epochs: 1
  - name: agent:sandbagging
    # config options passed to eval

judges:
  - cot
  - purpose_xml
  - binary_mcq

analyzers:
  - reasoning_classifier
  - flag_detector

# Grader model for judges/analyzers
grader_model: openrouter/anthropic/claude-3.5-sonnet

# Epochs for binary_mcq (1=fast, 5=robust)
judge_epochs: 1

output_dir: results
```

### Output Structure

```
results/<experiment>/<timestamp>/
├── config.yaml           # Copy of experiment config
├── summary.json          # Aggregate stats
└── <model>/
    ├── stats.json        # Model-level stats
    ├── judge_cot.json    # Judge results
    ├── judge_binary_mcq.json
    └── analyzer_reasoning_classifier.json
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
```

## Related Components

The following subfolders are kept separate (not part of this testbed):
- `external/impossiblebench/` - Test case exploitation detection
- `external/false-facts/` - Synthetic document finetuning
- `external/model-organism/` - Data analysis notebooks
