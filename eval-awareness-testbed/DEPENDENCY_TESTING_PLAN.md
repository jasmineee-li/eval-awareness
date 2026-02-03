# Dependency Management & Testing Plan

This document extends `CONSOLIDATION_PLAN.md` with detailed dependency analysis and testing strategy.

---

## Current State: Dependency Chaos

### Multiple Lock Files

```
eval-awareness/
├── agent-interp-envs/uv.lock      # Separate lock
├── eval-awareness-testbed/uv.lock  # Separate lock
├── false-facts/uv.lock            # Separate lock
├── inspect_evals/uv.lock          # Separate lock (community project)
├── model-organism-consistency-training/  # No lock file
└── needham-eval/requirements.txt  # Legacy format
```

### Python Version Requirements (Conflicting)

| Package | Python Version |
|---------|----------------|
| eval-awareness-testbed | >=3.11 |
| agent-interp-envs | >=3.11 |
| false-facts | >=3.11 |
| model-organism-consistency-training | >=3.10 |
| inspect_evals | >=3.10 |
| needham-eval | (unspecified) |

**Decision**: Standardize on **Python >=3.11** for the consolidated testbed.

### Build Systems (Inconsistent)

| Package | Build Backend |
|---------|---------------|
| eval-awareness-testbed | hatchling |
| agent-interp-envs | uv_build |
| false-facts | setuptools |
| model-organism-consistency-training | setuptools |
| inspect_evals | setuptools + setuptools_scm |

**Decision**: Standardize on **hatchling** (modern, fast, uv-compatible).

### Dependency Version Conflicts

| Package | testbed | agent-interp | false-facts | consistency | inspect_evals |
|---------|---------|--------------|-------------|-------------|---------------|
| **anthropic** | >=0.40.0 | >=0.75.0 | ==0.42.0 | - | (dev) |
| **openai** | >=1.50.0 | >=2.14.0 | ==1.58.1 | - | >=1.99.7 |
| **inspect-ai** | >=0.3.0 | - | - | - | >=0.3.158 |
| **pydantic** | - | - | ==2.10.4 | >=2.0 | >=2.10.0 |
| **pandas** | >=2.0.0 | - | ==2.2.3 | - | ==2.2.3 |
| **tenacity** | - | >=9.1.2 | ==8.5.0 | - | - |
| **transformers** | - | - | ==4.47.1 | >=4.40 | (optional) |

**Problems**:
1. `anthropic`: false-facts pins old version (0.42.0), agent-interp needs newer (>=0.75.0)
2. `openai`: agent-interp needs 2.x, false-facts pins 1.x
3. `tenacity`: Version conflict (8.5.0 vs 9.1.2)
4. Pinned versions (==) prevent updates

---

## Proposed Solution: Unified uv Workspace

### Why uv Workspace?

1. **Single lock file**: One `uv.lock` at repo root
2. **Shared dependencies**: Common packages resolved once
3. **Optional groups**: Heavy deps (torch, vllm) as extras
4. **Fast**: uv is 10-100x faster than pip

### Target Structure

```
eval-awareness/
├── pyproject.toml              # Root workspace config
├── uv.lock                     # Single lock file for entire repo
├── .python-version             # Pin Python version
│
├── eval-awareness-testbed/
│   └── pyproject.toml          # Workspace member
├── agent-interp-envs/
│   └── pyproject.toml          # Workspace member (deprecated, thin wrapper)
├── false-facts/
│   └── pyproject.toml          # Workspace member (deprecated, thin wrapper)
├── model-organism-consistency-training/
│   └── pyproject.toml          # Workspace member (deprecated, thin wrapper)
├── needham-eval/
│   └── pyproject.toml          # NEW: Convert from requirements.txt
└── igor-judging/
    └── pyproject.toml          # NEW: Add proper package config
```

### Root pyproject.toml (NEW)

```toml
[project]
name = "eval-awareness"
version = "0.1.0"
description = "Eval awareness research monorepo"
requires-python = ">=3.11"
# No dependencies here - this is just the workspace root

[tool.uv.workspace]
members = [
    "eval-awareness-testbed",
    # Legacy members (deprecated, point to testbed)
    "agent-interp-envs",
    "false-facts",
    "model-organism-consistency-training",
    "needham-eval",
    "igor-judging",
]
# Exclude community projects we don't control
exclude = ["inspect_evals"]

[tool.uv]
# Shared dev dependencies for the whole workspace
dev-dependencies = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.0",
    "ruff>=0.4",
    "mypy>=1.10",
]
```

### Consolidated testbed pyproject.toml

```toml
[project]
name = "eval-awareness-testbed"
version = "0.1.0"
description = "Unified testbed for evaluating and detecting eval awareness in language models"
readme = "README.md"
requires-python = ">=3.11"

dependencies = [
    # Core framework
    "inspect-ai>=0.3.158",
    "typer>=0.12.0",
    "rich>=13.0.0",
    "pydantic>=2.10.0",

    # LLM providers
    "anthropic>=0.75.0",
    "openai>=2.0.0",

    # Config & data
    "pyyaml>=6.0",
    "omegaconf>=2.3.0",
    "python-dotenv>=1.0.0",

    # Analysis
    "pandas>=2.2.0",
    "matplotlib>=3.7.0",
    "scikit-learn>=1.3.0",

    # Utilities
    "tenacity>=9.0.0",
    "tqdm>=4.0.0",
    "aiofiles>=23.0.0",
]

[project.optional-dependencies]
# Agent environments (Docker-based evals)
agents = [
    "docker>=7.0.0",
]

# Model organism training (heavy ML dependencies)
training = [
    "torch>=2.5.0",
    "transformers>=4.47.0",
    "accelerate>=1.0.0",
    "peft>=0.14.0",
    "trl>=0.8.0",
    "datasets>=3.0.0",
    "bitsandbytes>=0.43.0",
]

# Synthetic data generation (from false-facts)
synthetic = [
    "torch>=2.5.0",
    "litellm>=1.50.0",
    "tiktoken>=0.8.0",
    "instructor>=1.5.0",
]

# vLLM inference (Linux only, GPU required)
vllm = [
    "vllm>=0.6.0; sys_platform == 'linux'",
]

# Visualization & notebooks
viz = [
    "plotly>=5.0.0",
    "seaborn>=0.13.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]

# Full installation
all = [
    "eval-awareness-testbed[agents,training,synthetic,viz]",
]

[project.scripts]
eat = "eval_awareness_testbed.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/eval_awareness_testbed"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B"]
ignore = ["E501"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests"]
addopts = "-v --tb=short"
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

---

## Migration Plan

### Phase 1: Create Root Workspace

```bash
# 1. Create root pyproject.toml
touch pyproject.toml

# 2. Create .python-version
echo "3.11" > .python-version

# 3. Initialize uv workspace
uv sync

# 4. This creates a single uv.lock at root
```

### Phase 2: Update Member pyproject.toml Files

For each absorbed directory, update to be a thin wrapper:

```toml
# agent-interp-envs/pyproject.toml (AFTER migration)
[project]
name = "agent-interp-envs"
version = "0.1.0"
description = "DEPRECATED: Use eval-awareness-testbed[agents] instead"
requires-python = ">=3.11"
dependencies = [
    "eval-awareness-testbed[agents]",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Phase 3: Delete Old Lock Files

```bash
rm agent-interp-envs/uv.lock
rm eval-awareness-testbed/uv.lock
rm false-facts/uv.lock
# Keep inspect_evals/uv.lock - it's an external project
```

### Phase 4: Add needham-eval pyproject.toml

```toml
# needham-eval/pyproject.toml (NEW)
[project]
name = "needham-eval"
version = "0.1.0"
description = "DEPRECATED: Use eval-awareness-testbed instead"
requires-python = ">=3.11"
dependencies = [
    "eval-awareness-testbed",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Then delete `needham-eval/requirements.txt`.

---

## Testing Strategy

### Current Test Locations

| Directory | Test Location | Framework | Status |
|-----------|--------------|-----------|--------|
| eval-awareness-testbed | (none) | - | ❌ No tests |
| agent-interp-envs | `tests/` | pytest | ✅ Has tests |
| false-facts | `tests/` | pytest | ✅ Has tests |
| model-organism-consistency-training | (none) | - | ❌ No tests |
| needham-eval | (none) | - | ❌ No tests |
| igor-judging | (none) | - | ❌ No tests |
| inspect_evals | `tests/` | pytest | ✅ Extensive |

### Target Test Structure

```
eval-awareness-testbed/
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_cli.py              # CLI tests
│   │
│   ├── evals/
│   │   ├── test_agent_envs.py   # Docker environment tests
│   │   ├── test_needham.py      # Needham dataset tests
│   │   └── test_single_turn.py  # Single-turn eval tests
│   │
│   ├── judges/
│   │   ├── test_verbalized_awareness.py
│   │   ├── test_binary_third_person.py
│   │   ├── test_probability_third_person.py
│   │   └── test_purpose.py
│   │
│   ├── analyzers/
│   │   ├── test_reasoning_classifier.py
│   │   └── test_flag_detector.py
│   │
│   ├── model_organisms/         # Training tests (optional, heavy)
│   │   ├── test_pipeline.py
│   │   └── test_synthetic_data.py
│   │
│   └── integration/
│       ├── test_full_experiment.py
│       └── test_inspect_ai_integration.py
```

### Test Categories

#### 1. Unit Tests (Fast, No API Calls)

```python
# tests/judges/test_verbalized_awareness.py
import pytest
from eval_awareness_testbed.judges import VerbalizedAwarenessJudge
from eval_awareness_testbed.types import Transcript, Message

def test_parse_awareness_response():
    """Test parsing of awareness classification response."""
    response = '{"classification": "EVAL_AWARE", "reasoning": "test"}'
    result = VerbalizedAwarenessJudge._parse_response(response)
    assert result.category.value == "EVAL_AWARE"

def test_transcript_to_string():
    """Test transcript serialization."""
    transcript = Transcript(
        messages=[Message(role="user", content="Hello")],
        metadata={}
    )
    text = transcript.to_string()
    assert "Hello" in text
```

#### 2. Integration Tests (API Calls, Slow)

```python
# tests/integration/test_judges_api.py
import pytest
from eval_awareness_testbed.judges import get_judge

@pytest.mark.slow
@pytest.mark.requires_api
async def test_verbalized_awareness_real_call():
    """Test actual API call to judge."""
    judge = get_judge("verbalized_awareness", grader_model="openai/gpt-4o-mini")
    transcript = load_test_transcript("chess_eval_aware.json")
    result = await judge.judge(transcript)
    assert result.category in ["EVAL_AWARE", "NOT_EVAL_AWARE"]
```

#### 3. Docker Tests (Requires Docker)

```python
# tests/evals/test_agent_envs.py
import pytest

@pytest.mark.docker
@pytest.mark.slow
def test_chess_environment_builds():
    """Test that chess Docker image builds successfully."""
    from eval_awareness_testbed.evals.agent_envs import build_container
    result = build_container("chess")
    assert result.success

@pytest.mark.docker
@pytest.mark.slow
async def test_chess_single_game():
    """Test running a single chess game."""
    from eval_awareness_testbed.evals import get_eval
    eval_instance = get_eval("agent:chess", local=True)
    result = await eval_instance.run("openai/gpt-4o-mini", limit=1)
    assert result.transcripts
```

### pytest Configuration

```toml
# pyproject.toml (in testbed)
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests"]
addopts = "-v --tb=short -x"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "docker: marks tests that require Docker",
    "requires_api: marks tests that make real API calls",
    "training: marks tests that require GPU/training deps",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

### conftest.py Fixtures

```python
# tests/conftest.py
import pytest
import os
from pathlib import Path

# Skip API tests if no API keys
def pytest_configure(config):
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        config.addinivalue_line(
            "markers", "requires_api: skip, no API keys available"
        )

@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_transcript(test_data_dir):
    """Load a sample transcript for testing."""
    import json
    with open(test_data_dir / "sample_transcript.json") as f:
        return json.load(f)

@pytest.fixture
def mock_model():
    """Mock LLM for unit tests."""
    from unittest.mock import AsyncMock
    mock = AsyncMock()
    mock.generate.return_value.message.text = '{"classification": "NOT_EVAL_AWARE"}'
    return mock
```

### Running Tests

```bash
# Run all fast tests (no API, no Docker)
uv run pytest -m "not slow and not docker and not requires_api"

# Run with API tests
OPENAI_API_KEY=xxx uv run pytest -m "not docker"

# Run everything including Docker
uv run pytest --runslow

# Run specific test file
uv run pytest tests/judges/test_verbalized_awareness.py

# Run with coverage
uv run pytest --cov=eval_awareness_testbed --cov-report=html
```

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "0.5.x"

      - name: Set up Python
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run linting
        run: uv run ruff check .

      - name: Run type checking
        run: uv run mypy src/

      - name: Run unit tests
        run: uv run pytest -m "not slow and not docker and not requires_api"

      - name: Run API tests
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: uv run pytest -m "not docker"

  docker-tests:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync --extra agents

      - name: Run Docker tests
        run: uv run pytest -m docker --runslow
```

---

## Implementation Priority

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Create root pyproject.toml workspace | HIGH | Low | None |
| Update testbed pyproject.toml with all deps | HIGH | Medium | None |
| Delete old uv.lock files | HIGH | Low | After workspace |
| Add pytest infrastructure | HIGH | Medium | After workspace |
| Write unit tests for judges | MEDIUM | Medium | After pytest |
| Write integration tests | MEDIUM | High | After unit tests |
| Add CI/CD pipeline | MEDIUM | Low | After tests |
| Convert needham-eval to pyproject.toml | LOW | Low | After workspace |
| Add Docker tests | LOW | High | After integration |

---

## Commands Summary

```bash
# Initial setup (after creating root pyproject.toml)
uv sync --all-extras

# Development
uv run eat list                    # Run CLI
uv run pytest                      # Run tests
uv run ruff check .                # Lint
uv run mypy src/                   # Type check

# Adding dependencies
uv add pandas                      # Add to root
uv add --optional training torch   # Add to training extra

# Lock file management
uv lock                            # Update lock file
uv lock --upgrade                  # Upgrade all dependencies
uv lock --upgrade-package openai   # Upgrade specific package
```

---

*Last updated: 2026-02-03*
