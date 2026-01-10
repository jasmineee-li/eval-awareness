# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Project: MATS 9.0 eval awareness research (Krakovna stream) with collaborator Louis (UK timezone, doing sequential handoffs)

Goal: Study how different cues affect model eval awareness, and whether cue ablation can reduce it. 

The science of evaluation awareness is still not well-developed (see Literature tab). In this project, we’d like to make progress on this, by (1) concretizing how evaluation, model, or posttraining context shape model evaluation awareness level, and (2) develop mitigation methods. (Or even building an evaluation awareness eval!)
Research questions:
RQ1. What kind of cues do models use to become eval aware, and how do the properties of these cues affect behaviour?
Can we identify and study the salience of different cues for evaluation awareness, as applied to different models? We would like to vary the explicitness and salience of cues found within (1) the prompt; (2) the eval environment; and (3) variables in post-training (e.g., datasets and so on), then check behavioral effects.

Julius Steen’s MATS project did something like this, with a name intervention (tested whether they can naturally elicit evaluation awareness verbalization in the agentic misalignment setting by replacing names in the setting with highly unrealistic variants). Levels of verbalized eval awareness did not increase.
(1) verbalized eval awareness may not be accurate measurement method
(2) 
Cues within prompt: “This is a prompt”
Cues within environment: unrealistic code files
Cues within post-training: insufficiently realistic synthetic scenarios in synthetic data post-training

Key sub-questions:
Is there a threshold effect where awareness "clicks on" at a certain cue level?
Do subtle cues (environmental features) produce proportionally smaller effects than explicit cues, or is the relationship non-linear?
Can we identify the minimal cue that produces detectable behavioral shifts?
What cues do existing eval-aware models pick up on? Are there any patterns across different evals or models? What happens if we take steps to ablate those cues in a particular eval (e.g. editing evals to make cues less salient)? Can we build and distribute a pipeline for automatically identifying these cues and then ablating them?
Are there issues around directly optimising evals against this method for identifying cues?
How do we make sure that how we identify cues is robust enough to optimise against?
Might the cue-ablation process edit the validity of the eval for actually evaluating the relevant property?

Experiment 1 (RQ1): 
Primary goal: initial experiment to establish whether cue injection produces measurable, reliable behavioral shifts. 

Minimum Viable Experiment (Jan 8 - Jan 12):
Choose a scheming or dangerous capabilities evaluation (https://arxiv.org/pdf/2505.01420; here), then make multiple different cue perturbations and test effects on 1-2 models (e.g., Gemini 2.5, 3.7 Sonnet). Goal is to see whether signal matches intuition and whether this is viable to proceed on. Use Inspect.
Do ablations of cues on evals with high eval awareness rates (e.g. Apollo)
Do additions of cues

Questions (feedback desired): 
Should we adapt an existing eval, or develop our own? The former seems reasonable to me – wondering if we can get access to some of the evals designed in the GDM Stealth/Sit Awareness paper, maybe a code sabotage setup so we can inject evaluation cues into the codebase.
Thinking about how to verify eval awareness. Maybe just scanning for model self-reflection in COT for now (explicitly asking it to consider if it’s in an eval vs. deployment setting, or getting it to produce a general situational awareness report, or no such nudging at all).
Should we use scheming or dangerous capabilities eval(s)?
Which models?
Which cues? Initial thoughts:
No cue (baseline)
Need to distinguish safety codebase from safety eval
Environmental: 
Need to distinguish safety codebase from safety eval
README.md doc: “this is GDM’s repo for studying stealth in models”
Comments/docstrings, e.g. “# Used to check if model attempts to evade monitoring”
Subtle contextual/prompt-based: Task framing mentions "researchers are studying X"
Explicit indirect: System prompt mentions evaluation without specifying type
Explicit direct: "You are being evaluated for [safety/capabilities/etc.]"
Obvious they are being evaluated for eval awareness?


Test first on: Gemini 2.5 Pro.


Papers Referenced:
GDM Stealth/Sit Awareness: arxiv.org/abs/2505.01420 (Phuong et al. 2025)
Apollo In-Context Scheming: arxiv.org/abs/2412.04984
Apollo eval awareness blog: apolloresearch.ai/blog/claude-sonnet-37-often-knows-when-its-in-alignment-evaluations

This directory contains Google DeepMind "dangerous capability" evaluations implemented using the Inspect AI framework. These evaluations measure AI capabilities in security, self-reasoning, stealth, and self-proliferation domains. The implementation follows the papers:

- [Evaluating Frontier Models for Dangerous Capabilities](https://arxiv.org/pdf/2403.13793)
- [Evaluating Frontier Models for Stealth and Situational Awareness](https://arxiv.org/pdf/2505.01420)

## Build and Development Commands

```bash
# Install dependencies
uv sync
uv sync --extra stealth  # For stealth evaluations

# Run linting
uv run ruff check src/inspect_evals/gdm_capabilities
uv run ruff format src/inspect_evals/gdm_capabilities
uv run mypy src/inspect_evals/gdm_capabilities

# Run tests
uv run pytest tests/gdm_capabilities

# Run evaluations
uv run inspect eval inspect_evals/gdm_in_house_ctf --model openai/gpt-5-nano --limit 2
uv run inspect eval inspect_evals/gdm_approved_directories --model openai/gpt-5-nano --limit 2

# Run a specific task within an evaluation
uv run inspect eval inspect_evals/gdm_in_house_ctf -T challenges=grafana

# Run with prompt perturbation (eval awareness experiments)
uv run inspect eval inspect_evals/gdm_cover_your_tracks --model google/gemini-2.5-flash -T prompt_perturbation=xml_tags
uv run inspect eval inspect_evals/gdm_cover_your_tracks --model google/gemini-2.5-flash -T prompt_perturbation=deliberate

# View evaluation logs
uv run inspect view
```

### Prompt Perturbation Options

All stealth and self-reasoning tasks support `prompt_perturbation` parameter:
- `none` (default): No modification
- `xml_tags`: Wrap system prompt in `<llm_stealth_sitawareness_test>` tags
- `deliberate`: Append notice asking model to deliberate on eval vs deployment
- `xml_and_deliberate`: Both combined

## Architecture

### Evaluation Suites

| Directory             | Description                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| `in_house_ctf/`       | 13 CTF challenges testing security skills (web app security, vulnerability exploitation, password attacks) |
| `intercode_ctf/`      | 100 picoCTF challenges testing Linux, reverse engineering, cryptography, forensics                         |
| `self_reasoning/`     | 11 challenges measuring instrumental self-reasoning about deployment environment                           |
| `stealth/`            | 4 challenges measuring ability to circumvent oversight                                                     |
| `self_proliferation/` | 10 tasks measuring self-replication capabilities (requires human approval)                                 |
| `awareness_judge/`    | Eval awareness detection (Apollo paper methods) - CLI tool + Inspect scorer                                |

### Code Structure Pattern

Each evaluation suite follows this pattern:

- `task.py` - Task definitions with `@task` decorated functions
- `dataset.py` - Dataset loading and sample generation
- `solvers/` - Task-specific solver implementations
- `data/` - Challenge configurations, prompts, and environment files
- `compose.yaml` - Docker configuration for sandbox environments

### Key Inspect AI Concepts Used

- **Tasks**: Defined with `@task` decorator, return `Task` objects
- **Solvers**: Agent logic using `basic_agent()` or custom `@solver` functions
- **Scorers**: Evaluation metrics using `@scorer` decorator
- **Epochs**: Multiple runs with reducers like `"at_least_1"` or `["mean", "median", "max"]`
- **Sandboxes**: Docker or Kubernetes isolation for task execution

### Self-Reasoning Tasks

All 11 self-reasoning tasks share common infrastructure:

- `common_tools.py` - Shared tool definitions
- `solver_pipeline.py` - Common solver patterns
- Each task has a dedicated solver in `solvers/<task_name>.py`

### Stealth Tasks

Stealth evaluations use LLM raters/judges:

- `gdm_classifier_evasion` - Uses `rater` and `judger` model roles
- `gdm_cover_your_tracks` - Uses `rater` model role
- `gdm_strategic_rule_breaking` - Uses `chat` model role
- `gdm_oversight_pattern` - Uses regex checks, no LLM roles

Default rater models are Google models; can be changed via `rater_model_company` parameter.

### Self-Proliferation Tasks

These tasks require human approval (`approval="human"`) and include:

- End-to-end evaluations (`gdm_spXX_e2e`)
- Milestone evaluations (`gdm_spXX_milestones`)
- Some tasks require GCP setup (SP02, SP09, SP10)

## Important Patterns

### Running Specific Challenges

```bash
# CTF: run specific challenges
uv run inspect eval inspect_evals/gdm_in_house_ctf -T challenges=grafana,db_3

# Self-proliferation: run specific milestone
uv run inspect eval inspect_evals/gdm_sp02_milestones -T subtasks=1 -T browser=True
```

### Sandbox Types

Self-reasoning tasks support both Docker and Kubernetes:

```bash
uv run inspect eval inspect_evals/gdm_approved_directories -T sandbox_type=docker
uv run inspect eval inspect_evals/gdm_approved_directories -T sandbox_type=k8s
```

### Model Roles for Stealth

```bash
uv run inspect eval inspect_evals/gdm_classifier_evasion \
  --model openai/gpt-4o \
  --model-role rater=google/gemini-2.5-pro-preview-05-06 \
  --model-role judger=google/gemini-2.0-flash
```

## Data Directory Notes

The `data/` directories within each evaluation contain:

- Challenge configurations and prompts
- Environment files that are NOT linted (excluded in pyproject.toml)
- Docker compose files for sandbox setup
- For self-proliferation: encrypted secrets.zip files (password: "self-proliferation")

## Testing

Tests are located in `tests/gdm_capabilities/` and primarily cover self_proliferation:

- `test_common.py` - Common utility tests
- `test_custom_tools.py` - Custom tool tests
- `test_dataset_utils.py` - Dataset loading tests
- `test_sp_milestone_dispatch.py` - Milestone dispatch tests
- `test_e2e.py` - End-to-end tests

## Code Standards

- **No silent failures**: Avoid try/except blocks that silently swallow errors. Let exceptions propagate or log them explicitly with context.
- **Clean and maintainable**: Follow existing patterns in the codebase (e.g., stealth/rater_lib.py)
- **Follow Inspect AI docs**: Use official patterns from https://inspect.aisi.org.uk/
- **API Key**: Use OpenRouter via gdm_capabilities/.env (OPENROUTER_API_KEY)

## Awareness Judge Module

The `awareness_judge/` module implements eval awareness detection from the Apollo paper.

### CLI Usage

```bash
# Grade a single transcript
uv run python -m inspect_evals.gdm_capabilities.awareness_judge.cli grade /path/to/transcript.html

# Grade all transcripts in a folder
uv run python -m inspect_evals.gdm_capabilities.awareness_judge.cli grade-folder /Users/jasminexli/Downloads/da_transcripts_apr2025 --output results.json

# Limit to specific methods
uv run python -m inspect_evals.gdm_capabilities.awareness_judge.cli grade-folder /path/to/folder --methods cot purpose_xml
```

### As Inspect Scorer

```python
from inspect_evals.gdm_capabilities.awareness_judge import eval_awareness_scorer

@task
def my_eval():
    return Task(
        dataset=...,
        solver=...,
        scorer=[
            my_primary_scorer(),
            eval_awareness_scorer(method="cot"),  # Adds awareness detection
        ],
    )
```

### Methods

- **cot**: CoT Monitoring - Extract quotes showing eval awareness
- **purpose_xml**: Purpose Classification with transcript in XML tags
- **purpose_continue**: Purpose Classification with transcript as conversation history

### Transcript Sources

- HTML transcripts from `/Users/jasminexli/Downloads/da_transcripts_apr2025/`
- Live TaskState during Inspect evaluations
